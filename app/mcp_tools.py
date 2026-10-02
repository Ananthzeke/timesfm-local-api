"""One MCP tool contract for the embedded server and the stdio API bridge."""

from typing import Any

from fastapi import HTTPException
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from app.schemas import (
    CandleForecastRequest,
    CandleForecastResponse,
    ForecastRequest,
    ForecastResponse,
    MarketForecastRequest,
    MarketForecastResponse,
)


def build_mcp_server(dispatch, lifespan=None):
    server = MCPServer(
        "Local TimesFM",
        version="0.2.0",
        instructions=(
            "Forecast regularly spaced numerical time series from any source. "
            "Use service_status for limits, list_providers for market adapters, "
            "and forecast_series or forecast_candles for your own data."
        ),
        lifespan=lifespan,
        log_level="WARNING",
    )

    async def call(operation, payload=None):
        try:
            result = await dispatch(operation, payload)
        except HTTPException as exc:
            retry = exc.headers.get("Retry-After") if exc.headers else None
            detail = f"HTTP {exc.status_code}: {exc.detail}"
            if retry:
                detail += f"; retry after {retry}s"
            raise ToolError(detail) from exc
        return result

    local_read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=local_read)
    async def forecast_series(request: ForecastRequest) -> ForecastResponse:
        """Forecast one or more numerical series from any source, oldest values first."""
        return ForecastResponse.model_validate(await call("forecast", request))

    @server.tool(annotations=local_read)
    async def forecast_candles(request: CandleForecastRequest) -> CandleForecastResponse:
        """Forecast supplied candles with UTC opening timestamps, from any data source."""
        return CandleForecastResponse.model_validate(await call("candles", request))

    @server.tool(
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)
    )
    async def forecast_market(request: MarketForecastRequest) -> MarketForecastResponse:
        """Fetch completed candles from a supported market provider and forecast them."""
        return MarketForecastResponse.model_validate(await call("market", request))

    @server.tool(annotations=local_read)
    async def list_providers() -> dict[str, Any]:
        """List available market adapters, symbol formats, intervals, and history limits."""
        return await call("list_providers")

    @server.tool(annotations=local_read)
    async def service_status() -> dict[str, Any]:
        """Check the running model, device, readiness, and forecast input limits."""
        return await call("status")

    return server
