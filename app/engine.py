import gc
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.metrics import Metrics
from app.runtime import defer_forecaster_gc


@dataclass
class Prediction:
    forecast: np.ndarray
    quantiles: np.ndarray


class MockEngine:
    """Last-price baseline for API/scheduler testing; explicitly labelled mock."""

    backend = "mock"
    model_name = "last-price-baseline-NOT-TimesFM"
    device = "cpu"

    def load(self):
        pass

    def predict(self, contexts, horizon):
        return [
            Prediction(
                forecast=np.full(horizon, float(context[-1]), dtype=np.float32),
                quantiles=np.full((horizon, 9), float(context[-1]), dtype=np.float32),
            )
            for context in contexts
        ]

    def close(self):
        pass


class TimesFMEngine:
    backend = "timesfm3"

    def __init__(self, settings: Settings, metrics: Metrics):
        self.settings = settings
        self.metrics = metrics
        self.model_name = settings.checkpoint
        self.device = settings.device
        self.model = None

    def load(self):
        if not self.settings.license_accepted:
            raise RuntimeError(
                "Read the TimesFM 3.0 non-commercial/non-production license and set "
                "TF_LICENSE_ACCEPTED=true for permitted evaluation use."
            )
        import torch
        from timesfm3 import ModelConfig, TimesFM3Forecaster

        self.torch = torch
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install compatible CUDA-enabled PyTorch/drivers "
                "or explicitly set TF_DEVICE=cpu."
            )
        kwargs = {}
        if self.settings.model_revision:
            kwargs["revision"] = self.settings.model_revision
        self.model = TimesFM3Forecaster(
            ModelConfig(
                checkpoint_path=self.settings.checkpoint,
                per_core_batch_size=self.settings.max_batch_series,
                device=self.device,
                **kwargs,
            )
        )
        if self.device == "cuda":
            defer_forecaster_gc(
                self.model, torch, self.device, self.settings.gc_interval_seconds, self.metrics
            )
        context = np.linspace(1, 2, self.settings.max_context, dtype=np.float32)
        for _ in range(self.settings.warmup_runs):
            self.predict([context] * self.settings.max_batch_series, self.settings.max_horizon)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def _predict(self, contexts, horizon):
        with self.torch.inference_mode():
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            outputs = list(
                self.model.predict_batch(
                    contexts=contexts,
                    horizon=horizon,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                )
            )
            if self.device == "cuda":
                self.torch.cuda.synchronize()
        if len(outputs) != len(contexts):
            raise RuntimeError("Model returned an unexpected number of series")
        predictions = []
        for output in outputs:
            point = np.asarray(output.forecast, dtype=np.float64).reshape(-1)
            quantiles = np.asarray(output.quantiles, dtype=np.float64)
            if point.shape != (horizon,) or quantiles.shape != (horizon, 9):
                raise RuntimeError("Unexpected TimesFM output shape")
            if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
                raise RuntimeError("TimesFM returned non-finite predictions")
            predictions.append(Prediction(point, quantiles))
        return predictions

    def predict(self, contexts, horizon):
        try:
            return self._predict(contexts, horizon)
        except self.torch.cuda.OutOfMemoryError:
            gc.collect()
            self.torch.cuda.empty_cache()
            if len(contexts) == 1:
                raise
            self.metrics.oom_retries.inc()
            # Single-series retry trades throughput for lower peak memory.
            predictions = []
            for context in contexts:
                predictions.extend(self._predict([context], horizon))
            return predictions
        finally:
            if self.device == "cuda":
                for name, value in {
                    "allocated": self.torch.cuda.memory_allocated(),
                    "reserved": self.torch.cuda.memory_reserved(),
                    "peak_allocated": self.torch.cuda.max_memory_allocated(),
                }.items():
                    self.metrics.gpu_bytes.labels(name).set(value)

    def close(self):
        self.model = None
        if hasattr(self, "torch") and self.device == "cuda":
            # Release cycles from the instance-bound prediction method before
            # returning allocator blocks at shutdown or after failed startup.
            gc.collect()
            self.torch.cuda.empty_cache()
