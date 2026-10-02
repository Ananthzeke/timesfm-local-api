"""Stdio MCP bridge: reuse a running API without loading another model."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import HTTPException

from app.config import Settings
from app.mcp_tools import build_mcp_server

OPERATIONS = {
    "forecast": ("POST", "/v1/forecast"),
    "candles": ("POST", "/v1/forecast/candles"),
    "market": ("GET", "/v1/forecast/market"),
    "list_providers": ("GET", "/v1/providers"),
    "status": ("GET", "/readyz"),
}


def create_stdio_server(settings=None, api_url=None, transport=None):
    settings = settings or Settings(_env_file=Path(__file__).resolve().parents[1] / ".env")
    client = None

    @asynccontextmanager
    async def lifespan(_):
        nonlocal client
        headers = {"X-API-Key": settings.api_key.get_secret_value()} if settings.api_key else {}
        async with httpx.AsyncClient(
            base_url=(api_url or str(settings.api_url)).rstrip("/"),
            headers=headers,
            timeout=settings.request_timeout_seconds + settings.upstream_timeout_seconds + 5,
            transport=transport,
            trust_env=False,
        ) as active:
            client = active
            try:
                yield {}
            finally:
                client = None

    async def dispatch(operation, payload):
        if client is None:
            raise HTTPException(503, "MCP API bridge is not running")
        method, path = OPERATIONS[operation]
        data = payload.model_dump(mode="json", exclude_none=True) if payload else None
        try:
            response = await client.request(
                method, path, **({"json": data} if method == "POST" else {"params": data})
            )
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Forecast API is unreachable; start it first") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", "Forecast API request failed")
            except ValueError:
                detail = "Forecast API request failed"
            retry = response.headers.get("Retry-After")
            raise HTTPException(
                response.status_code, detail, headers={"Retry-After": retry} if retry else None
            )
        return response.json()

    return build_mcp_server(dispatch, lifespan=lifespan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url", help="Running forecast API; default TF_API_URL or localhost:8000"
    )
    args = parser.parse_args()
    create_stdio_server(api_url=args.api_url).run(transport="stdio")


if __name__ == "__main__":
    main()
