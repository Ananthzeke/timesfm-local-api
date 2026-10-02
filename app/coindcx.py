import math
import time
from datetime import UTC, datetime

import httpx

INTERVALS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}


class CandleError(Exception):
    pass


def closed_candles(data, interval, context, now_ms=None):
    """Require recent, regularly spaced completed candles; never fill missing prices."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    duration = INTERVALS[interval]
    if not isinstance(data, list):
        raise CandleError("Upstream candle response is not a list")
    rows = {}
    try:
        for row in data:
            timestamp = int(row["time"])
            if timestamp + duration > now_ms:
                continue
            close = float(row["close"])
            if not math.isfinite(close) or not 0 < close <= 1e20:
                raise ValueError("invalid close")
            if timestamp in rows and rows[timestamp] != close:
                raise ValueError("conflicting duplicate candle")
            rows[timestamp] = close
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CandleError("Upstream candle data is malformed") from exc
    selected = sorted(rows.items())[-context:]
    if len(selected) < context:
        raise CandleError("Not enough completed historical candles")
    if any(b[0] - a[0] != duration for a, b in zip(selected, selected[1:], strict=False)):
        raise CandleError("Historical candles contain a gap")
    end_ms = selected[-1][0] + duration
    if now_ms - end_ms > duration * 2:
        raise CandleError("Latest completed candle is stale")
    return [close for _, close in selected], end_ms


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


def iso_time(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat()
