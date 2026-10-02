# Generalized API and MCP verification — 2026-10-02

The updated GPU application was tested locally on the existing GTX 1650 (4 GB)
with TimesFM 3.0.2, PyTorch 2.6.0+cu118, and MCP SDK 2.2.0. The model, precision,
inference thread, queue, batching, and CUDA cleanup optimization were retained.

Configuration: context limit 256, horizon limit 16, batch size up to 4, batching
window 2 ms, outstanding capacity 8, explicit cleanup interval 30 seconds.

## Functional verification

- All 47 tests passed; Ruff lint and formatting checks passed.
- Supplied numerical arrays and historical timestamped sensor observations
  produced real CUDA forecasts through the generic HTTP endpoints.
- Live Binance `BTCUSDT` and CoinDCX `B-BTC_USDT` requests returned valid forecasts.
  The legacy CoinDCX route also returned its original response fields.
- Native Streamable HTTP MCP negotiated both automatic/current and legacy modes.
  All five tools were listed and called against the running GPU application.
- A separate stdio client launched the installed `timesfm-mcp` command from `/tmp`,
  discovered the tools, and successfully forwarded forecasting and status calls.
- Forecast arrays and quantiles matched exactly between HTTP, native MCP, and
  the stdio bridge for identical inputs.
- Importing the stdio bridge did not import PyTorch. NVIDIA process inspection
  showed one inference process using 1436 MiB, with total GPU usage of 1444 MiB.
- Tests verified input validation, API-key protection, localhost Host checking,
  request-body limits, provider errors, custom provider registration, and shared
  inference/upstream capacity. Expected MCP errors retain status and retry guidance.

Details are recorded in [gpu-smoke.json](gpu-smoke.json). The first numeric call
in this smoke test took about 812 ms of service time after startup and idle time;
it is not included in the warmed benchmark below. This test does not establish
forecast accuracy for a particular application.

## Warmed HTTP performance check

`scripts/benchmark.py` was run sequentially against `POST /v1/forecast`, with
128 synthetic observations, horizon 4, quantiles omitted, and five warm-up requests
before each measurement. These short checks verify the existing batching behavior;
they are not a sustained capacity or market-fetch benchmark.

| Concurrent clients | Requests | Successful requests/s | p50 latency | p95 latency | p99 latency |
| --- | --- | --- | --- | --- | --- |
| 1 | 100 | 17.86 | 54.24 ms | 66.36 ms | 71.67 ms |
| 8 | 200 | 68.77 | 110.64 ms | 144.27 ms | 180.83 ms |

All 300 measured requests returned HTTP 200. Raw results are in
[benchmark-c1.json](benchmark-c1.json) and [benchmark-c8.json](benchmark-c8.json).
The concurrent result is comparable to the earlier optimized measurements of
about 66–67 requests/s. The single-client result is slightly slower than the earlier
approximately 20 requests/s sample; timing varies with workload and GPU conditions.
MCP throughput and latency were not separately benchmarked.

Market latency additionally includes provider fetching. During the smoke test,
fetching took about 194 ms for Binance and 88 ms for CoinDCX; these are individual
samples, not provider latency distributions.

## Connection

HTTP API documentation: `http://127.0.0.1:8000/docs`.
Native MCP endpoint: `http://127.0.0.1:8000/mcp` (Streamable HTTP).
Stdio bridge: `.venv/bin/timesfm-mcp --api-url http://127.0.0.1:8000`.
See the project README for payloads and MCP client configuration.
