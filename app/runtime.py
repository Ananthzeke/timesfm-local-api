"""Instance-scoped cleanup policy for the pinned TimesFM PyTorch adapter."""

import time
from functools import update_wrapper
from importlib.metadata import version
from types import FunctionType, MethodType


class PeriodicGC:
    def __init__(self, upstream_gc, torch, device, interval_seconds, metrics, clock=None):
        self.upstream_gc, self.torch = upstream_gc, torch
        self.interval_seconds, self.metrics = interval_seconds, metrics
        self.clock = clock or time.monotonic
        self.last_collection = self.clock()
        self.total_memory = torch.cuda.get_device_properties(device).total_memory

    def __call__(self, device=None, gc_memory_threshold=0.9):
        if device is None or self.torch.device(device).type != "cuda":
            return self.upstream_gc(device, gc_memory_threshold)
        allocated = self.torch.cuda.memory_allocated(device)
        pressure = self.total_memory > 0 and allocated / self.total_memory > gc_memory_threshold
        if pressure or self.clock() - self.last_collection >= self.interval_seconds:
            self.upstream_gc(device, gc_memory_threshold)
            self.last_collection = self.clock()
            self.metrics.gc_collections.labels("memory_pressure" if pressure else "interval").inc()


def defer_forecaster_gc(model, torch, device, interval_seconds, metrics):
    """Keep upstream prediction code, replacing only this instance's cleanup hook.

    TimesFM 3.0.2 calls a full Python collection on every prediction. Rebinding
    its method with a private globals dictionary lets us pace that collection
    without editing the dependency or changing other forecasters in this process.
    Automatic Python GC stays enabled; the upstream 90% GPU-pressure cleanup
    still runs immediately. Setting the interval to zero leaves upstream intact.
    """
    if interval_seconds == 0:
        return
    method = model.predict_batch
    function = getattr(method, "__func__", None)
    if (
        function is None
        or version("timesfm") != "3.0.2"
        or function.__module__ != "timesfm3.torch.timesfm3_forecaster"
        or "try_gc" not in function.__code__.co_names
        or not callable(function.__globals__.get("try_gc"))
    ):
        raise RuntimeError(
            "Periodic cleanup requires the pinned TimesFM 3.0.2 PyTorch adapter; "
            "set TF_GC_INTERVAL_SECONDS=0 to use upstream cleanup."
        )
    namespace = dict(function.__globals__)
    namespace["try_gc"] = PeriodicGC(namespace["try_gc"], torch, device, interval_seconds, metrics)
    optimized = FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    optimized.__kwdefaults__ = function.__kwdefaults__
    update_wrapper(optimized, function)
    model.predict_batch = MethodType(optimized, model)
