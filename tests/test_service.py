import asyncio
import threading

import httpx
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.engine import MockEngine
from app.main import create_app


def payload(length=32, horizon=4):
    return {"series": [{"id": "BTC", "values": list(range(1, length + 1))}], "horizon": horizon}


@pytest.fixture
def config():
    return Settings(backend="mock", _env_file=None)


@pytest.mark.asyncio
async def test_forecast_validation_auth_metrics(config):
    config.api_key = SecretStr("test-key")
    app = create_app(config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            assert (await c.get("/readyz")).json()["backend"] == "mock"
            assert (await c.post("/v1/forecast", json=payload())).status_code == 401
            headers = {"X-API-Key": "test-key"}
            response = await c.post("/v1/forecast", json=payload(), headers=headers)
            result = response.json()
            assert response.status_code == 200
            assert result["results"][0]["forecast"] == [32.0] * 4
            assert result["model"] == "last-price-baseline-NOT-TimesFM"
            assert len(result["results"][0]["quantiles"]) == 9
            assert result["timing"]["service_ms"] >= result["timing"]["queue_ms"]
            for bad in [payload(31), payload(513), payload(horizon=65)]:
                assert (await c.post("/v1/forecast", json=bad, headers=headers)).status_code == 422
            bad = payload()
            bad["series"][0]["values"][0] = True
            assert (await c.post("/v1/forecast", json=bad, headers=headers)).status_code == 422
            assert (await c.get("/metrics")).status_code == 401
            metrics = await c.get("/metrics", headers=headers)
            assert "tf_completed_series_total 1.0" in metrics.text


@pytest.mark.asyncio
async def test_body_limit(config):
    app = create_app(config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            response = await c.post("/v1/forecast", content=b"x" * (config.max_body_bytes + 1))
            assert response.status_code == 413


class GateEngine(MockEngine):
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def predict(self, contexts, horizon):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("Test did not release the inference gate")
        return super().predict(contexts, horizon)


async def entered(engine):
    assert await asyncio.to_thread(engine.entered.wait, 2)


@pytest.mark.asyncio
async def test_overload_and_health_responsive(config):
    config.queue_capacity = 1
    engine = GateEngine()
    app = create_app(config, engine_factory=lambda *_: engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            first = asyncio.create_task(c.post("/v1/forecast", json=payload()))
            try:
                await entered(engine)
                assert (await asyncio.wait_for(c.get("/healthz"), 1)).status_code == 200
                assert (await c.post("/v1/forecast", json=payload())).status_code == 429
            finally:
                engine.release.set()
            assert (await first).status_code == 200


@pytest.mark.asyncio
async def test_running_deadline_keeps_capacity_until_gpu_finishes(config):
    config.queue_capacity = 1
    config.request_timeout_seconds = 0.05
    engine = GateEngine()
    app = create_app(config, engine_factory=lambda *_: engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            try:
                first = asyncio.create_task(c.post("/v1/forecast", json=payload()))
                await entered(engine)
                assert (await first).status_code == 504
                assert (await c.post("/v1/forecast", json=payload())).status_code == 429
            finally:
                engine.release.set()
        # Shutdown waits for the underlying call even after the client deadline.
    assert engine.calls == 1
    assert app.state.scheduler.outstanding == 0


@pytest.mark.asyncio
async def test_compatible_requests_are_batched(config):
    config.max_batch_series = 2
    config.batch_window_ms = 10
    app = create_app(config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            responses = await asyncio.gather(
                *[c.post("/v1/forecast", json=payload()) for _ in range(2)]
            )
            assert all(r.status_code == 200 for r in responses)
            assert all(r.json()["timing"]["batch_series"] == 2 for r in responses)


@pytest.mark.asyncio
async def test_inference_failure_recovers(config):
    class Flaky(MockEngine):
        failed = False

        def predict(self, contexts, horizon):
            if not self.failed:
                self.failed = True
                raise RuntimeError("intentional test failure")
            return super().predict(contexts, horizon)

    app = create_app(config, engine_factory=lambda *_: Flaky())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            assert (await c.post("/v1/forecast", json=payload())).status_code == 503
            assert (await c.post("/v1/forecast", json=payload())).status_code == 200


@pytest.mark.asyncio
async def test_queued_timeout_removed_without_inference(config):
    config.queue_capacity = 2
    config.request_timeout_seconds = 0.05
    engine = GateEngine()
    app = create_app(config, engine_factory=lambda *_: engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            try:
                first = asyncio.create_task(c.post("/v1/forecast", json=payload()))
                await entered(engine)
                second = asyncio.create_task(c.post("/v1/forecast", json=payload()))
                results = await asyncio.gather(first, second)
                assert all(r.status_code == 504 for r in results)
                assert app.state.scheduler.outstanding == 1
                assert len(app.state.scheduler.pending) == 0
            finally:
                engine.release.set()
    assert engine.calls == 1


@pytest.mark.asyncio
async def test_incompatible_horizons_not_combined(config):
    config.max_batch_series = 2
    config.batch_window_ms = 10
    app = create_app(config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            responses = await asyncio.gather(
                c.post("/v1/forecast", json=payload(horizon=4)),
                c.post("/v1/forecast", json=payload(horizon=5)),
            )
            assert all(r.json()["timing"]["batch_series"] == 1 for r in responses)
            assert [len(r.json()["results"][0]["forecast"]) for r in responses] == [4, 5]
