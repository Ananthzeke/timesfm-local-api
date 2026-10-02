from types import SimpleNamespace

import pytest

from app.metrics import Metrics
from app.runtime import PeriodicGC, defer_forecaster_gc


@pytest.fixture
def fake_torch():
    cuda = SimpleNamespace(
        allocated=100,
        get_device_properties=lambda _: SimpleNamespace(total_memory=1000),
    )
    cuda.memory_allocated = lambda _: cuda.allocated
    return SimpleNamespace(cuda=cuda, device=lambda d: SimpleNamespace(type=str(d)))


def test_cleanup_interval_and_immediate_memory_pressure(fake_torch):
    now, calls, metrics = [0.0], [], Metrics()
    policy = PeriodicGC(
        lambda *args: calls.append(args), fake_torch, "cuda", 30, metrics, clock=lambda: now[0]
    )
    now[0] = 29
    policy("cuda")
    assert calls == []
    now[0] = 30
    policy("cuda")
    assert calls == [("cuda", 0.9)]
    now[0] = 31
    policy("cuda")
    assert len(calls) == 1
    fake_torch.cuda.allocated = 950
    policy("cuda")
    assert len(calls) == 2
    assert metrics.gc_collections.labels("interval")._value.get() == 1
    assert metrics.gc_collections.labels("memory_pressure")._value.get() == 1
    fake_torch.cuda.allocated = 100
    now[0] = 60
    policy("cuda")
    assert len(calls) == 2
    now[0] = 61
    policy("cuda")
    assert len(calls) == 3


def test_cpu_cleanup_and_custom_pressure_threshold_preserved(fake_torch):
    calls = []
    policy = PeriodicGC(lambda *args: calls.append(args), fake_torch, "cuda", 30, Metrics())
    policy("cpu")
    policy()
    assert calls == [("cpu", 0.9), (None, 0.9)]
    fake_torch.cuda.allocated = 600
    policy("cuda", 0.5)
    assert calls[-1] == ("cuda", 0.5)


def forecasters(calls):
    namespace = {
        "__name__": "timesfm3.torch.timesfm3_forecaster",
        "try_gc": lambda *args: calls.append(args),
    }
    exec(
        "class Forecaster:\n"
        "    def predict_batch(self, contexts, horizon=4, *, return_quantiles=True):\n"
        "        try_gc('cuda')\n"
        "        yield contexts, horizon, return_quantiles\n",
        namespace,
    )
    return namespace["Forecaster"](), namespace["Forecaster"]()


def test_rebinding_is_scoped_to_one_forecaster(fake_torch, monkeypatch):
    monkeypatch.setattr("app.runtime.version", lambda _: "3.0.2")
    calls = []
    optimized, untouched = forecasters(calls)
    original = optimized.predict_batch.__func__
    original_hook = original.__globals__["try_gc"]
    defer_forecaster_gc(optimized, fake_torch, "cuda", 30, Metrics())
    assert list(optimized.predict_batch([1, 2], return_quantiles=False)) == [([1, 2], 4, False)]
    assert calls == []
    assert list(untouched.predict_batch([1, 2], return_quantiles=False)) == [([1, 2], 4, False)]
    assert calls == [("cuda",)]
    assert original.__globals__["try_gc"] is original_hook
    assert optimized.predict_batch.__func__.__code__ is original.__code__
    assert untouched.predict_batch.__func__ is original


def test_zero_interval_preserves_original_method(fake_torch):
    calls = []
    model, _ = forecasters(calls)
    defer_forecaster_gc(model, fake_torch, "cuda", 0, Metrics())
    assert "predict_batch" not in model.__dict__
    list(model.predict_batch([1]))
    assert calls == [("cuda",)]


def test_unverified_adapter_rejected_without_modifying_model(fake_torch, monkeypatch):
    monkeypatch.setattr("app.runtime.version", lambda _: "other-version")
    model, _ = forecasters([])
    with pytest.raises(RuntimeError, match="TF_GC_INTERVAL_SECONDS=0"):
        defer_forecaster_gc(model, fake_torch, "cuda", 30, Metrics())
    assert "predict_batch" not in model.__dict__
