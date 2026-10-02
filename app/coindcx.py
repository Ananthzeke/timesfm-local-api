import httpx

from app.candles import CandleError, closed_observations
from app.candles import iso_time as iso_time

INTERVALS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}


def closed_candles(data, interval, context, now_ms=None):
    """Require recent, regularly spaced completed candles; never fill missing prices."""
    if not isinstance(data, list):
        raise CandleError("Upstream candle response is not a list")
    try:
        rows = [{"timestamp_ms": row["time"], "close": row.get("close")} for row in data]
    except (KeyError, TypeError, AttributeError) as exc:
        raise CandleError("Upstream candle data is malformed") from exc
    return closed_observations(
        rows, INTERVALS[interval], context, now_ms=now_ms, positive_only=True
    )


async def fetch_candles(client, pair, interval, context):
    try:
        response = await client.get(
            "https://api.coindcx.com/market_data/candles",
            params={"pair": pair, "interval": interval, "limit": min(1000, context + 2)},
        )
        response.raise_for_status()
        return closed_candles(response.json(), interval, context)
    except (httpx.HTTPError, ValueError) as exc:
        raise CandleError("CoinDCX candles could not be fetched") from exc
