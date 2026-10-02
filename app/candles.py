"""Provider-neutral normalization of completed, regularly spaced observations."""

import math
import time
from datetime import UTC, datetime


class CandleError(Exception):
    pass


def closed_observations(
    data, interval_ms, context=None, *, now_ms=None, require_fresh=True, positive_only=False
):
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if not isinstance(data, list):
        raise CandleError("Candle data must be a list")
    rows = {}
    try:
        for row in data:
            timestamp = row["timestamp_ms"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                raise ValueError("invalid timestamp")
            if timestamp + interval_ms > now_ms:
                continue
            raw_close = row["close"]
            if isinstance(raw_close, bool):
                raise ValueError("invalid close")
            close = float(raw_close)
            if not math.isfinite(close) or abs(close) > 1e20 or (positive_only and close <= 0):
                raise ValueError("invalid close")
            if timestamp in rows and rows[timestamp] != close:
                raise ValueError("conflicting duplicate candle")
            rows[timestamp] = close
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CandleError("Candle data is malformed") from exc
    selected = sorted(rows.items())
    if context is not None:
        selected = selected[-context:]
    if len(selected) < (context or 32):
        raise CandleError("Not enough completed historical candles")
    if any(b[0] - a[0] != interval_ms for a, b in zip(selected, selected[1:], strict=False)):
        raise CandleError("Historical candles contain a gap")
    end_ms = selected[-1][0] + interval_ms
    if require_fresh and now_ms - end_ms > interval_ms * 2:
        raise CandleError("Latest completed candle is stale")
    return [close for _, close in selected], end_ms


def iso_time(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat()
