"""Call this from your MCP tool after it obtains ordered, completed candle closes."""

import os

import httpx


async def forecast_closes(closes: list[float], symbol: str, horizon: int = 4):
    headers = {"X-API-Key": os.environ["TF_API_KEY"]} if os.getenv("TF_API_KEY") else {}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            os.getenv("TF_API_URL", "http://127.0.0.1:8000") + "/v1/forecast",
            headers=headers,
            json={"series": [{"id": symbol, "values": closes}], "horizon": horizon},
        )
        response.raise_for_status()
        return response.json()
