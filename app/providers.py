"""Fixed public-data adapters; forecast inputs remain independent of providers."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from app.candles import CandleError, closed_observations
from app.coindcx import INTERVALS, fetch_candles

BINANCE_INTERVALS = {
    "1s": 1000,
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1h": 3600000,
    "2h": 7200000,
    "4h": 14400000,
    "6h": 21600000,
    "8h": 28800000,
    "12h": 43200000,
    "1d": 86400000,
    "3d": 259200000,
    "1w": 604800000,
}


async def fetch_binance(client, symbol, interval, context):
    try:
        response = await client.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": context + 2},
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("invalid response")
        rows = [{"timestamp_ms": row[0], "close": row[4]} for row in data]
        return closed_observations(rows, BINANCE_INTERVALS[interval], context, positive_only=True)
    except (httpx.HTTPError, ValueError, TypeError, IndexError, KeyError) as exc:
        raise CandleError("Binance candles could not be fetched or parsed") from exc


@dataclass(frozen=True)
class MarketProvider:
    name: str
    intervals: dict[str, int]
    symbol_pattern: str
    example_symbol: str
    fetch: Callable[..., Awaitable[tuple[list[float], int]]]
    max_context: int = 998

    def describe(self):
        return {
            "name": self.name,
            "intervals": list(self.intervals),
            "symbol_pattern": self.symbol_pattern,
            "example_symbol": self.example_symbol,
            "max_context": self.max_context,
        }


PROVIDERS = {
    "coindcx": MarketProvider(
        "coindcx", INTERVALS, r"[A-Z0-9]+-[A-Z0-9]+_[A-Z0-9]+", "B-BTC_USDT", fetch_candles
    ),
    "binance": MarketProvider(
        "binance", BINANCE_INTERVALS, r"[A-Z0-9]{2,40}", "BTCUSDT", fetch_binance
    ),
}
