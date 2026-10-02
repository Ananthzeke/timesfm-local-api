import hmac
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.coindcx import INTERVALS, CandleError, fetch_candles, iso_time
from app.config import Settings
from app.engine import MockEngine, TimesFMEngine
from app.metrics import Metrics
from app.scheduler import DeadlineExceeded, QueueFull, Scheduler, ServiceUnavailable
from app.schemas import ForecastRequest, ForecastResponse, SeriesInput


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if b"content-length" in headers:
            try:
                length = int(headers[b"content-length"])
                if length < 0:
                    raise ValueError
            except ValueError:
                response = JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
                return await response(scope, receive, send)
            if length > self.max_bytes:
                response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                return await response(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings=None, engine_factory=None, transport=None):
    settings = settings or Settings()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app):
        engine = (
            engine_factory(settings, metrics)
            if engine_factory
            else MockEngine()
            if settings.backend == "mock"
            else TimesFMEngine(settings, metrics)
        )
        scheduler = Scheduler(settings, engine, metrics)
        await scheduler.start()
        app.state.scheduler = scheduler
        app.state.upstream_active = 0
        try:
            async with httpx.AsyncClient(
                timeout=settings.upstream_timeout_seconds,
                limits=httpx.Limits(max_connections=settings.max_upstream_requests),
                transport=transport,
            ) as client:
                app.state.http = client
                yield
        finally:
            await scheduler.close()

    app = FastAPI(title="Local TimesFM forecast API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)

    @app.middleware("http")
    async def observe_http(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        if request.url.path in {"/v1/forecast", "/v1/forecast/coindcx"}:
            metrics.http_seconds.labels(request.url.path, str(response.status_code)).observe(
                time.perf_counter() - start
            )
        return response

    async def auth(x_api_key: Annotated[str | None, Header()] = None):
        if settings.api_key and (
            x_api_key is None
            or not hmac.compare_digest(
                x_api_key.encode(), settings.api_key.get_secret_value().encode()
            )
        ):
            raise HTTPException(401, "Invalid API key")

    async def forecast(payload, request):
        if (
            len(payload.series) > settings.max_batch_series
            or payload.horizon > settings.max_horizon
            or any(len(s.values) > settings.max_context for s in payload.series)
            or any(abs(v) > 1e20 for s in payload.series for v in s.values)
        ):
            raise HTTPException(422, "Input exceeds configured series/context/horizon/value limits")
        try:
            return await request.app.state.scheduler.submit(payload)
        except QueueFull as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except DeadlineExceeded as exc:
            raise HTTPException(504, str(exc)) from exc
        except ServiceUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/healthz")
    async def health():
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready(request: Request):
        scheduler = request.app.state.scheduler
        if not scheduler.ready:
            raise HTTPException(503, "Inference worker is unavailable")
        return {
            "status": "ready",
            "backend": scheduler.engine.backend,
            "device": scheduler.engine.device,
            "max_context": settings.max_context,
            "max_horizon": settings.max_horizon,
            "max_batch_series": settings.max_batch_series,
        }

    @app.get("/metrics", dependencies=[Depends(auth)])
    async def prometheus():
        return Response(
            generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @app.post("/v1/forecast", response_model=ForecastResponse, dependencies=[Depends(auth)])
    async def direct_forecast(payload: ForecastRequest, request: Request):
        return await forecast(payload, request)

    @app.get("/v1/forecast/coindcx", dependencies=[Depends(auth)])
    async def coindcx_forecast(
        request: Request,
        pair: Annotated[str, Query(pattern=r"^[A-Z0-9]+-[A-Z0-9]+_[A-Z0-9]+$", max_length=60)],
        interval: Literal["1m", "15m", "1h", "1d"] = "1m",
        context: Annotated[int, Query(ge=32, le=998)] = 128,
        horizon: Annotated[int, Query(ge=1, le=256)] = 4,
    ):
        if context > settings.max_context or horizon > settings.max_horizon:
            raise HTTPException(422, "Input exceeds configured context/horizon limits")
        if request.app.state.upstream_active >= settings.max_upstream_requests:
            metrics.rejected.labels("upstream_capacity").inc()
            raise HTTPException(
                429, "Candle fetch capacity exhausted", headers={"Retry-After": "1"}
            )
        request.app.state.upstream_active += 1
        start = time.perf_counter()
        try:
            values, end = await fetch_candles(request.app.state.http, pair, interval, context)
        except CandleError as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            request.app.state.upstream_active -= 1
        fetch_ms = (time.perf_counter() - start) * 1000
        result = await forecast(
            ForecastRequest(series=[SeriesInput(id=pair, values=values)], horizon=horizon), request
        )
        return {
            "source": "coindcx",
            "pair": pair,
            "interval": interval,
            "context_end": iso_time(end),
            "data_fetch_ms": fetch_ms,
            "forecast_close_times": [
                iso_time(end + (i + 1) * INTERVALS[interval]) for i in range(horizon)
            ],
            "prediction": result,
        }

    return app


app = create_app()
