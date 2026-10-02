import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.engine import TimesFMEngine
from app.metrics import Metrics


class FakeCuda:
    class OutOfMemoryError(Exception):
        pass

    @staticmethod
    def empty_cache():
        pass


def test_license_checked_before_model_import():
    engine = TimesFMEngine(Settings(_env_file=None), Metrics())
    with pytest.raises(RuntimeError, match="license"):
        engine.load()


def test_pinned_adapter_contract_and_maximum_warmup(monkeypatch):
    config_kwargs, calls = {}, []

    def model_config(**kwargs):
        config_kwargs.update(kwargs)
        return kwargs

    class Forecaster:
        def __init__(self, config):
            assert config == config_kwargs

        def predict_batch(self, **kwargs):
            calls.append(kwargs)
            horizon = kwargs["horizon"]
            return [
                SimpleNamespace(forecast=np.ones(horizon), quantiles=np.ones((horizon, 9)))
                for _ in kwargs["contexts"]
            ]

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=FakeCuda, inference_mode=nullcontext)
    )
    monkeypatch.setitem(
        sys.modules,
        "timesfm3",
        SimpleNamespace(ModelConfig=model_config, TimesFM3Forecaster=Forecaster),
    )
    settings = Settings(
        device="cpu",
        license_accepted=True,
        max_context=128,
        max_horizon=16,
        max_batch_series=2,
        model_revision="commit",
        _env_file=None,
    )
    engine = TimesFMEngine(settings, Metrics())
    engine.load()
    assert config_kwargs["per_core_batch_size"] == 2
    assert config_kwargs["revision"] == "commit"
    assert len(calls) == settings.warmup_runs
    assert all(len(call["contexts"]) == 2 for call in calls)
    assert all(len(call["contexts"][0]) == 128 for call in calls)
    assert all(call["horizon"] == 16 for call in calls)
    assert all(call["use_symmetric_averaging"] is False for call in calls)
    assert all(call["return_quantiles"] is True for call in calls)
    engine.close()
    assert engine.model is None


def test_batch_oom_retries_singly():
    class OOMForecaster:
        def predict_batch(self, contexts, horizon, **_kwargs):
            if len(contexts) > 1:
                raise FakeCuda.OutOfMemoryError
            return [SimpleNamespace(forecast=np.ones(horizon), quantiles=np.ones((horizon, 9)))]

    metrics = Metrics()
    engine = TimesFMEngine(Settings(device="cpu", _env_file=None), metrics)
    engine.torch = SimpleNamespace(cuda=FakeCuda, inference_mode=nullcontext)
    engine.model = OOMForecaster()
    assert len(engine.predict([np.ones(32), np.ones(32)], 4)) == 2
    assert metrics.oom_retries._value.get() == 1


def test_rejects_invalid_model_output():
    engine = TimesFMEngine(Settings(device="cpu", _env_file=None), Metrics())
    engine.torch = SimpleNamespace(cuda=FakeCuda, inference_mode=nullcontext)
    engine.model = SimpleNamespace(
        predict_batch=lambda **_kwargs: [
            SimpleNamespace(forecast=np.ones(4), quantiles=np.ones((9, 4)))
        ]
    )
    with pytest.raises(RuntimeError, match="shape"):
        engine.predict([np.ones(32)], 4)
