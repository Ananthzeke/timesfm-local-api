import asyncio
import threading
import time

import httpx
import pytest
from mcp import Client
from pydantic import SecretStr

from app.config import Settings
from app.engine import MockEngine
from app.main import create_app
from app.mcp_server import create_stdio_server

TOOLS = {
    "forecast_series",
    "forecast_candles",
    "forecast_market",
    "list_providers",
    "service_status",
}
SERIES = {"series": [{"id": "temperature", "values": list(range(-32, 0))}], "horizon": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge", [False, True])
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_native_and_stdio_tool_contracts_share_service(bridge, mode):
    settings = Settings(backend="mock", api_key=SecretStr("test-key"), _env_file=None)
    end = int(time.time() * 1000) // 60_000 * 60_000

    def upstream(_):
        rows = [[end - i * 60_000, "0", "0", "0", str(100 + i)] for i in range(33)]
        return httpx.Response(200, json=rows)

    app = create_app(settings, transport=httpx.MockTransport(upstream))
    async with app.router.lifespan_context(app):
        server = (
            create_stdio_server(settings, transport=httpx.ASGITransport(app))
            if bridge
            else app.state.mcp
        )
        async with Client(server, mode=mode) as client:
            tools = (await client.list_tools()).tools
            assert {t.name for t in tools} == TOOLS
            forecast_tool = next(t for t in tools if t.name == "forecast_series")
            assert forecast_tool.output_schema is not None
            result = await client.call_tool("forecast_series", {"request": SERIES})
            assert not result.is_error
            assert result.structured_content["results"][0]["forecast"] == [-1.0, -1.0]
            candles = {
                "id": "temperature",
                "candles": [{"timestamp_ms": end - i * 60_000, "close": i - 40} for i in range(33)],
                "interval_ms": 60_000,
                "horizon": 2,
            }
            result = await client.call_tool("forecast_candles", {"request": candles})
            assert not result.is_error
            assert result.structured_content["prediction"]["results"][0]["forecast"] == [-39.0] * 2
            result = await client.call_tool(
                "forecast_market",
                {"request": {"provider": "binance", "symbol": "BTCUSDT", "context": 32}},
            )
            assert not result.is_error
            assert result.structured_content["provider"] == "binance"
            assert (await client.call_tool("service_status")).structured_content[
                "status"
            ] == "ready"
            result = await client.call_tool("list_providers")
            assert {p["name"] for p in result.structured_content["providers"]} == {
                "coindcx",
                "binance",
            }
            invalid = {**SERIES, "horizon": 65}
            assert (await client.call_tool("forecast_series", {"request": invalid})).is_error
            invalid = {"series": [{"id": "bad", "values": [True] * 32}]}
            assert (await client.call_tool("forecast_series", {"request": invalid})).is_error
            assert app.state.scheduler.outstanding == 0


@pytest.mark.asyncio
async def test_mcp_http_protocol_auth_body_limit_and_host_check():
    settings = Settings(backend="mock", api_key=SecretStr("test-key"), _env_file=None)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8000"
        ) as client:
            message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            headers = {"Accept": "application/json, text/event-stream"}
            assert (await client.post("/mcp", json=message, headers=headers)).status_code == 401
            assert (await client.get("/v1/providers")).status_code == 401
            headers["X-API-Key"] = "test-key"
            init = await client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                },
            )
            assert init.status_code == 200
            assert init.json()["result"]["protocolVersion"] == "2025-11-25"
            headers["MCP-Protocol-Version"] = "2025-11-25"
            response = await client.post("/mcp", json=message, headers=headers)
            assert response.status_code == 200
            assert {t["name"] for t in response.json()["result"]["tools"]} == TOOLS
            result = await client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "forecast_series", "arguments": {"request": SERIES}},
                },
            )
            assert result.status_code == 200
            assert result.json()["result"]["structuredContent"]["device"] == "cpu"
            response = await client.post(
                "/mcp", headers=headers, content=b"x" * (settings.max_body_bytes + 1)
            )
            assert response.status_code == 413
            headers["Host"] = "evil.example"
            assert (await client.post("/mcp", json=message, headers=headers)).status_code == 421


@pytest.mark.asyncio
async def test_mcp_and_http_use_same_inference_capacity():
    entered, release = threading.Event(), threading.Event()

    class GateEngine(MockEngine):
        def predict(self, contexts, horizon):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release the inference gate")
            return super().predict(contexts, horizon)

    app = create_app(
        Settings(backend="mock", queue_capacity=1, _env_file=None),
        engine_factory=lambda *_: GateEngine(),
    )
    async with app.router.lifespan_context(app):
        async with Client(app.state.mcp, mode="legacy") as mcp:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as http:
                first = asyncio.create_task(http.post("/v1/forecast", json=SERIES))
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    result = await mcp.call_tool("forecast_series", {"request": SERIES})
                    assert result.is_error
                    assert "HTTP 429" in result.content[0].text
                    assert "retry after 1s" in result.content[0].text
                    assert app.state.scheduler.outstanding == 1
                finally:
                    release.set()
                assert (await first).status_code == 200


@pytest.mark.asyncio
async def test_stdio_bridge_reports_api_connectivity_failure():
    def unavailable(_):
        raise httpx.ConnectError("test unavailable")

    server = create_stdio_server(
        Settings(backend="mock", _env_file=None), transport=httpx.MockTransport(unavailable)
    )
    async with Client(server, mode="legacy") as client:
        result = await client.call_tool("forecast_series", {"request": SERIES})
        assert result.is_error
        assert "HTTP 503" in result.content[0].text
