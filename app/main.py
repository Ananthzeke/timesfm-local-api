import hmac
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import Settings
from app.engine import MockEngine, TimesFMEngine
from app.mcp_tools import build_mcp_server
from app.metrics import Metrics
from app.scheduler import Scheduler
from app.schemas import (
    CandleForecastRequest,
    CandleForecastResponse,
    ForecastRequest,
    ForecastResponse,
    MarketForecastRequest,
    MarketForecastResponse,
)
from app.service import ForecastService


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


def create_app(settings=None, engine_factory=None, transport=None, providers=None):
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
        try:
            async with httpx.AsyncClient(
                timeout=settings.upstream_timeout_seconds,
                limits=httpx.Limits(max_connections=settings.max_upstream_requests),
                transport=transport,
            ) as client:
                app.state.http = client
                app.state.service = ForecastService(settings, scheduler, client, metrics, providers)
                async with mcp.session_manager.run():
                    yield
        finally:
            await scheduler.close()

    app = FastAPI(title="Local TimesFM forecast API", version="0.2.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)

    def valid_key(value):
        return not settings.api_key or (
            value is not None
            and hmac.compare_digest(value.encode(), settings.api_key.get_secret_value().encode())
        )

    @app.middleware("http")
    async def observe_http(request, call_next):
        start = time.perf_counter()
        if request.url.path.rstrip("/") == "/mcp" and not valid_key(
            request.headers.get("X-API-Key")
        ):
            response = JSONResponse({"detail": "Invalid API key"}, status_code=401)
        else:
            response = await call_next(request)
        if request.url.path in {
            "/v1/forecast",
            "/v1/forecast/coindcx",
            "/v1/forecast/candles",
            "/v1/forecast/market",
            "/mcp",
        }:
            metrics.http_seconds.labels(request.url.path, str(response.status_code)).observe(
                time.perf_counter() - start
            )
        return response

    async def auth(x_api_key: Annotated[str | None, Header()] = None):
        if not valid_key(x_api_key):
            raise HTTPException(401, "Invalid API key")

    @app.get("/healthz")
    async def health():
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready(request: Request):
        return await request.app.state.service.status()

    @app.get("/metrics", dependencies=[Depends(auth)])
    async def prometheus():
        return Response(
            generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @app.post("/v1/forecast", response_model=ForecastResponse, dependencies=[Depends(auth)])
    async def direct_forecast(payload: ForecastRequest, request: Request):
        return await request.app.state.service.forecast(payload)

    @app.post(
        "/v1/forecast/candles", response_model=CandleForecastResponse, dependencies=[Depends(auth)]
    )
    async def candle_forecast(payload: CandleForecastRequest, request: Request):
        return await request.app.state.service.candles(payload)

    @app.get("/v1/providers", dependencies=[Depends(auth)])
    async def list_providers(request: Request):
        return await request.app.state.service.list_providers()

    @app.get(
        "/v1/forecast/market", response_model=MarketForecastResponse, dependencies=[Depends(auth)]
    )
    async def market_forecast(
        request: Request,
        provider: Annotated[str, Query(min_length=1, max_length=40)],
        symbol: Annotated[str, Query(min_length=1, max_length=80)],
        interval: Annotated[str, Query(min_length=1, max_length=10)] = "1m",
        context: Annotated[int, Query(ge=32, le=1024)] = 128,
        horizon: Annotated[int, Query(ge=1, le=256)] = 4,
        return_quantiles: bool = True,
    ):
        return await request.app.state.service.market(
            MarketForecastRequest(
                provider=provider,
                symbol=symbol,
                interval=interval,
                context=context,
                horizon=horizon,
                return_quantiles=return_quantiles,
            )
        )

    @app.get("/v1/forecast/coindcx", dependencies=[Depends(auth)])
    async def coindcx_forecast(
        request: Request,
        pair: Annotated[str, Query(pattern=r"^[A-Z0-9]+-[A-Z0-9]+_[A-Z0-9]+$", max_length=60)],
        interval: Literal["1m", "15m", "1h", "1d"] = "1m",
        context: Annotated[int, Query(ge=32, le=998)] = 128,
        horizon: Annotated[int, Query(ge=1, le=256)] = 4,
    ):
        result = await request.app.state.service.market(
            MarketForecastRequest(
                provider="coindcx", symbol=pair, interval=interval, context=context, horizon=horizon
            )
        )
        return {
            "source": "coindcx",
            "pair": pair,
            "interval": interval,
            "context_end": result.context_end,
            "data_fetch_ms": result.data_fetch_ms,
            "forecast_close_times": result.forecast_close_times,
            "prediction": result.prediction,
        }

    async def dispatch(operation, payload):
        method = getattr(app.state.service, operation)
        return await method(payload) if payload is not None else await method()

    mcp = build_mcp_server(dispatch)
    mcp_http = mcp.streamable_http_app(
        json_response=True, stateless_http=True, max_request_body_size=settings.max_body_bytes
    )
    app.state.mcp = mcp
    # Register last: existing API routes win, with the SDK serving /mcp exactly.
    app.mount("/", mcp_http)
    return app


app = create_app()
