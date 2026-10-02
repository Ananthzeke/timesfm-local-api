"""Shared forecast operations for HTTP and MCP, using one scheduler."""

import re
import time

from fastapi import HTTPException

from app.candles import CandleError, closed_observations, iso_time
from app.providers import PROVIDERS
from app.scheduler import DeadlineExceeded, QueueFull, ServiceUnavailable
from app.schemas import (
    CandleForecastResponse,
    ForecastRequest,
    MarketForecastResponse,
    SeriesInput,
)


class ForecastService:
    def __init__(self, settings, scheduler, http, metrics, providers=None):
        self.settings, self.scheduler, self.http, self.metrics = settings, scheduler, http, metrics
        self.providers = dict(PROVIDERS if providers is None else providers)
        self.upstream_active = 0

    def check_limits(self, context, horizon):
        if context > self.settings.max_context or horizon > self.settings.max_horizon:
            raise HTTPException(422, "Input exceeds configured context/horizon limits")

    async def forecast(self, payload):
        if (
            len(payload.series) > self.settings.max_batch_series
            or payload.horizon > self.settings.max_horizon
            or any(len(s.values) > self.settings.max_context for s in payload.series)
            or any(abs(v) > 1e20 for s in payload.series for v in s.values)
        ):
            raise HTTPException(422, "Input exceeds configured series/context/horizon/value limits")
        try:
            return await self.scheduler.submit(payload)
        except QueueFull as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except DeadlineExceeded as exc:
            raise HTTPException(504, str(exc)) from exc
        except ServiceUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    async def candles(self, payload):
        self.check_limits(payload.context or 0, payload.horizon)
        try:
            values, end = closed_observations(
                [c.model_dump() for c in payload.candles],
                payload.interval_ms,
                payload.context,
                require_fresh=payload.require_fresh,
            )
        except CandleError as exc:
            raise HTTPException(422, str(exc)) from exc
        self.check_limits(len(values), payload.horizon)
        result = await self.forecast(
            ForecastRequest(
                series=[SeriesInput(id=payload.id, values=values)],
                horizon=payload.horizon,
                return_quantiles=payload.return_quantiles,
            )
        )
        return CandleForecastResponse(
            id=payload.id,
            context_end=iso_time(end),
            forecast_close_times=[
                iso_time(end + (i + 1) * payload.interval_ms) for i in range(payload.horizon)
            ],
            prediction=result,
        )

    async def market(self, payload):
        provider = self.providers.get(payload.provider)
        if provider is None:
            raise HTTPException(422, "Unknown provider; see /v1/providers")
        if payload.interval not in provider.intervals:
            raise HTTPException(422, "Unsupported interval for this provider")
        if not re.fullmatch(provider.symbol_pattern, payload.symbol):
            raise HTTPException(422, "Invalid symbol format for this provider")
        self.check_limits(payload.context, payload.horizon)
        if payload.context > provider.max_context:
            raise HTTPException(422, "Context exceeds provider history limit")
        if self.upstream_active >= self.settings.max_upstream_requests:
            self.metrics.rejected.labels("upstream_capacity").inc()
            raise HTTPException(
                429, "Candle fetch capacity exhausted", headers={"Retry-After": "1"}
            )
        self.upstream_active += 1
        start = time.perf_counter()
        try:
            values, end = await provider.fetch(
                self.http, payload.symbol, payload.interval, payload.context
            )
        except CandleError as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            self.upstream_active -= 1
        fetch_ms = (time.perf_counter() - start) * 1000
        result = await self.forecast(
            ForecastRequest(
                series=[SeriesInput(id=payload.symbol, values=values)],
                horizon=payload.horizon,
                return_quantiles=payload.return_quantiles,
            )
        )
        duration = provider.intervals[payload.interval]
        return MarketForecastResponse(
            provider=payload.provider,
            symbol=payload.symbol,
            interval=payload.interval,
            context_end=iso_time(end),
            data_fetch_ms=fetch_ms,
            forecast_close_times=[
                iso_time(end + (i + 1) * duration) for i in range(payload.horizon)
            ],
            prediction=result,
        )

    async def list_providers(self):
        return {"providers": [provider.describe() for provider in self.providers.values()]}

    async def status(self):
        if not self.scheduler.ready:
            raise HTTPException(503, "Inference worker is unavailable")
        return {
            "status": "ready",
            "backend": self.scheduler.engine.backend,
            "device": self.scheduler.engine.device,
            "max_context": self.settings.max_context,
            "max_horizon": self.settings.max_horizon,
            "max_batch_series": self.settings.max_batch_series,
        }
