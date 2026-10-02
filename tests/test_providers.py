import asyncio
import time

import httpx
import pytest

from app.candles import iso_time
from app.config import Settings
from app.main import create_app
from app.providers import MarketProvider


def supplied_candles(count=33):
    return {
        "id": "temperature",
        "interval_ms": 60_000,
        "candles": [
            {"timestamp_ms": 1_700_000_000_000 + i * 60_000, "close": i - 40} for i in range(count)
        ],
        "horizon": 2,
        "return_quantiles": False,
    }


@pytest.mark.asyncio
async def test_supplied_candles_sort_select_context_and_keep_non_market_values():
    app = create_app(Settings(backend="mock", max_context=32, _env_file=None))
    body = supplied_candles()
    body["candles"].reverse()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            # Selection is explicit, rather than silently truncating excess history.
            assert (await client.post("/v1/forecast/candles", json=body)).status_code == 422
            body["context"] = 32
            response = await client.post("/v1/forecast/candles", json=body)
            assert response.status_code == 200
            result = response.json()
            end = 1_700_000_000_000 + 33 * 60_000
            assert result["context_end"] == iso_time(end)
            assert result["forecast_close_times"] == [
                iso_time(end + 60_000),
                iso_time(end + 120_000),
            ]
            assert result["prediction"]["results"][0]["forecast"] == [-8.0, -8.0]
            assert result["prediction"]["results"][0]["quantiles"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem", ["gap", "duplicate", "timestamp", "boolean", "magnitude", "nan", "stale"]
)
async def test_invalid_supplied_candles(problem):
    body = supplied_candles()
    if problem == "gap":
        del body["candles"][10]
    elif problem == "duplicate":
        body["candles"].append({**body["candles"][10], "close": 999})
    elif problem == "timestamp":
        body["candles"][0]["timestamp_ms"] = True
    elif problem == "boolean":
        body["candles"][0]["close"] = True
    elif problem == "magnitude":
        body["candles"][0]["close"] = 1e21
    elif problem == "nan":
        body["candles"][0]["close"] = "NaN"
    elif problem == "stale":
        body["require_fresh"] = True
    app = create_app(Settings(backend="mock", _env_file=None))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            assert (await client.post("/v1/forecast/candles", json=body)).status_code == 422
            assert app.state.scheduler.outstanding == 0


@pytest.mark.asyncio
async def test_forming_supplied_candle_does_not_enter_forecast():
    end = int(time.time() * 1000) // 60_000 * 60_000
    body = supplied_candles()
    body["candles"] = [{"timestamp_ms": end - i * 60_000, "close": 10 + i} for i in range(33)]
    body["candles"][0]["close"] = 999
    body["require_fresh"] = True
    app = create_app(Settings(backend="mock", _env_file=None))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/forecast/candles", json=body)
            assert response.status_code == 200
            assert response.json()["context_end"] == iso_time(end)
            assert response.json()["prediction"]["results"][0]["forecast"] == [11.0, 11.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["coindcx", "binance"])
async def test_generic_market_fetches_through_provider_adapter(provider):
    end = int(time.time() * 1000) // 60_000 * 60_000
    symbol = "BTCUSDT" if provider == "binance" else "B-BTC_USDT"

    def handler(request):
        assert request.url.params["interval"] == "1m"
        assert request.url.params["limit"] == "34"
        if provider == "binance":
            assert request.url == httpx.URL(
                "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=34"
            )
            rows = [[end - i * 60_000, "0", "0", "0", str(100 + i)] for i in range(33)]
        else:
            assert request.url.params["pair"] == symbol
            rows = [{"time": end - i * 60_000, "close": 100 + i} for i in range(33)]
        return httpx.Response(200, json=rows)

    app = create_app(
        Settings(backend="mock", _env_file=None), transport=httpx.MockTransport(handler)
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/forecast/market",
                params={"provider": provider, "symbol": symbol, "context": 32, "horizon": 2},
            )
            assert response.status_code == 200
            result = response.json()
            assert (result["provider"], result["symbol"]) == (provider, symbol)
            assert result["context_end"] == iso_time(end)
            assert result["prediction"]["results"][0]["forecast"] == [101.0, 101.0]
            assert result["data_fetch_ms"] >= 0
            if provider == "coindcx":
                old = await client.get(
                    "/v1/forecast/coindcx", params={"pair": symbol, "context": 32, "horizon": 2}
                )
                assert old.status_code == 200
                assert (old.json()["source"], old.json()["pair"]) == (provider, symbol)
                assert old.json()["prediction"]["results"] == result["prediction"]["results"]


@pytest.mark.asyncio
async def test_market_rejects_invalid_requests_before_network_access():
    def no_network(_):
        pytest.fail("Invalid request should not fetch upstream data")

    app = create_app(
        Settings(backend="mock", max_context=1024, _env_file=None),
        transport=httpx.MockTransport(no_network),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            providers = (await client.get("/v1/providers")).json()["providers"]
            assert {p["name"] for p in providers} == {"coindcx", "binance"}
            valid = {"provider": "binance", "symbol": "BTCUSDT", "context": 32}
            for invalid in [
                {"provider": "https://example.com"},
                {"symbol": "../../invalid"},
                {"interval": "1M"},
                {"context": 999},
            ]:
                assert (
                    await client.get("/v1/forecast/market", params={**valid, **invalid})
                ).status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{"error": "bad response"}, [[1]], [[1, 2, 3, 4, "NaN"]]])
async def test_malformed_market_data_returns_502_and_releases_capacity(data):
    app = create_app(
        Settings(backend="mock", _env_file=None),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data)),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/forecast/market",
                params={"provider": "binance", "symbol": "BTCUSDT", "context": 32},
            )
            assert response.status_code == 502
            assert app.state.service.upstream_active == 0
            assert app.state.scheduler.outstanding == 0


@pytest.mark.asyncio
async def test_custom_provider_and_shared_upstream_capacity():
    started, release = asyncio.Event(), asyncio.Event()

    async def fetch(*_):
        started.set()
        await release.wait()
        return [42.0] * 32, 1_700_000_000_000

    custom = MarketProvider("sensor", {"1m": 60_000}, r"sensor-\d+", "sensor-1", fetch)
    app = create_app(
        Settings(backend="mock", max_upstream_requests=1, _env_file=None),
        providers={"sensor": custom},
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            params = {"provider": "sensor", "symbol": "sensor-1", "context": 32}
            first = asyncio.create_task(client.get("/v1/forecast/market", params=params))
            try:
                await asyncio.wait_for(started.wait(), 1)
                response = await client.get("/v1/forecast/market", params=params)
                assert response.status_code == 429
                assert response.headers["Retry-After"] == "1"
            finally:
                release.set()
            assert (await first).status_code == 200
            assert app.state.service.upstream_active == 0
