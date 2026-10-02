import time

import httpx
import pytest

from app.coindcx import CandleError, closed_candles
from app.config import Settings
from app.main import create_app


def candles(end=6_000_000, count=33):
    return [{"time": end - i * 60_000, "close": 100 + i} for i in range(count)]


def test_reverse_order_forming_candle_removed():
    values, end = closed_candles(candles(), "1m", 32, now_ms=6_030_000)
    assert len(values) == 32
    assert values == list(range(132, 100, -1))
    assert end == 6_000_000


@pytest.mark.parametrize("problem", ["gap", "stale", "invalid", "insufficient", "duplicate"])
def test_rejects_bad_candles(problem):
    data = candles(count=34)
    now = 6_030_000
    if problem == "gap":
        del data[10]
    if problem == "stale":
        now += 600_000
    if problem == "invalid":
        data[5]["close"] = float("nan")
    if problem == "insufficient":
        data = data[:10]
    if problem == "duplicate":
        data.append({"time": data[5]["time"], "close": 1})
    with pytest.raises(CandleError):
        closed_candles(data, "1m", 32, now_ms=now)


@pytest.mark.asyncio
async def test_candle_api_with_mocked_upstream():
    end = int(time.time() * 1000) // 60_000 * 60_000

    def handler(request):
        assert request.url.host == "api.coindcx.com"
        assert request.url.params["pair"] == "B-BTC_USDT"
        return httpx.Response(200, json=candles(end))

    app = create_app(
        Settings(backend="mock", _env_file=None), transport=httpx.MockTransport(handler)
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            response = await c.get(
                "/v1/forecast/coindcx", params={"pair": "B-BTC_USDT", "context": 32}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"]["backend"] == "mock"
            assert len(data["forecast_close_times"]) == 4
            assert data["data_fetch_ms"] >= 0
