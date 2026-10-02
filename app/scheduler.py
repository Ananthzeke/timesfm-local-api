import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from app.schemas import ForecastResponse, SeriesForecast, Timing

log = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


class DeadlineExceeded(Exception):
    pass


class ServiceUnavailable(Exception):
    pass


@dataclass(eq=False)
class Job:
    payload: object
    future: asyncio.Future
    submitted: float
    started: bool = False

    @property
    def key(self):
        return self.payload.horizon, tuple(len(s.values) for s in self.payload.series)


class Scheduler:
    """One GPU owner; bounded outstanding work and FIFO compatible batching."""

    def __init__(self, settings, engine, metrics):
        self.settings, self.engine, self.metrics = settings, engine, metrics
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.pending = deque()
        self.outstanding = 0
        self.wake = asyncio.Event()
        self.stopping = False
        self.worker = None

    async def start(self):
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.engine.load)
        except BaseException:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.engine.close)
            self.executor.shutdown(wait=True)
            raise
        self.worker = asyncio.create_task(self._run(), name="forecast-scheduler")

    @property
    def ready(self):
        return not self.stopping and self.worker is not None and not self.worker.done()

    def _gauges(self):
        self.metrics.outstanding.set(self.outstanding)
        self.metrics.queue_depth.set(len(self.pending))

    async def submit(self, payload):
        if not self.ready:
            raise ServiceUnavailable("Inference worker is unavailable")
        if self.outstanding >= self.settings.queue_capacity:
            self.metrics.rejected.labels("capacity").inc()
            raise QueueFull("Inference capacity exhausted")
        job = Job(payload, asyncio.get_running_loop().create_future(), time.perf_counter())
        self.pending.append(job)
        self.outstanding += 1
        self.metrics.accepted.inc()
        self._gauges()
        self.wake.set()
        try:
            return await asyncio.wait_for(
                asyncio.shield(job.future), self.settings.request_timeout_seconds
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            job.future.cancel()
            if not job.started:
                try:
                    self.pending.remove(job)
                except ValueError:
                    pass  # Shutdown may already have removed this queued job.
                else:
                    self.outstanding -= 1
                    self._gauges()
            if isinstance(exc, asyncio.CancelledError):
                raise
            self.metrics.rejected.labels("deadline").inc()
            raise DeadlineExceeded("Forecast deadline exceeded") from exc

    async def _run(self):
        while True:
            if not self.pending:
                if self.stopping:
                    return
                self.wake.clear()
                await self.wake.wait()
                continue
            if self.settings.batch_window_ms:
                await asyncio.sleep(self.settings.batch_window_ms / 1000)
            if not self.pending:
                continue
            first = self.pending.popleft()
            jobs = [first]
            count = len(first.payload.series)
            while self.pending:
                next_job = self.pending[0]
                n = len(next_job.payload.series)
                if next_job.key != first.key or count + n > self.settings.max_batch_series:
                    break
                jobs.append(self.pending.popleft())
                count += n
            dispatched = time.perf_counter()
            for job in jobs:
                job.started = True
                self.metrics.queue_seconds.observe(dispatched - job.submitted)
            self._gauges()
            contexts = [
                np.asarray(series.values, dtype=np.float32)
                for job in jobs
                for series in job.payload.series
            ]
            try:
                outputs = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.engine.predict, contexts, first.payload.horizon
                )
                finished = time.perf_counter()
                inference_ms = (finished - dispatched) * 1000
                self.metrics.inference_seconds.observe(inference_ms / 1000)
                self.metrics.batch_size.observe(count)
                self.metrics.completed_series.inc(count)
                if len(outputs) != count:
                    raise RuntimeError("Unexpected prediction count")
                offset = 0
                for job in jobs:
                    results = []
                    for series in job.payload.series:
                        prediction = outputs[offset]
                        offset += 1
                        results.append(
                            SeriesForecast(
                                id=series.id,
                                forecast=prediction.forecast.tolist(),
                                quantiles={
                                    f"{(i + 1) / 10:.1f}": prediction.quantiles[:, i].tolist()
                                    for i in range(9)
                                }
                                if job.payload.return_quantiles
                                else None,
                            )
                        )
                    response = ForecastResponse(
                        request_id=str(uuid.uuid4()),
                        backend=self.engine.backend,
                        model=self.engine.model_name,
                        device=self.engine.device,
                        horizon=job.payload.horizon,
                        results=results,
                        timing=Timing(
                            queue_ms=(dispatched - job.submitted) * 1000,
                            inference_batch_ms=inference_ms,
                            service_ms=(finished - job.submitted) * 1000,
                            batch_series=count,
                        ),
                    )
                    if not job.future.done():
                        job.future.set_result(response)
            except Exception:
                log.exception("Inference batch failed")
                self.metrics.inference_errors.inc()
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(ServiceUnavailable("Inference failed"))
            finally:
                self.outstanding -= len(jobs)
                self._gauges()

    async def close(self):
        self.stopping = True
        while self.pending:
            job = self.pending.popleft()
            self.outstanding -= 1
            if not job.future.done():
                job.future.set_exception(ServiceUnavailable("Service is stopping"))
        self._gauges()
        self.wake.set()
        if self.worker:
            await self.worker  # Drain an active GPU call; do not spawn a second owner.
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.engine.close)
        finally:
            self.executor.shutdown(wait=True)
