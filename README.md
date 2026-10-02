# Local TimesFM API

A Python API for local **TimesFM 3.0 inference**, using PyTorch and FastAPI. Designed
to start conservatively on a GTX 1650 with 4 GB VRAM, with controls you can benchmark
and adjust. It downloads pretrained weights; you do not need to train the model.

The wrapper uses the forecasting API in `timesfm==3.0.2`. CUDA inference has been
verified locally on a GTX 1650 with 4 GB VRAM. Hardware-specific measurements and
their workload details are in `reports/performance-2026-10-02/REPORT.md` and
`reports/optimization-2026-10-02/REPORT.md`. Verification of the generalized API,
MCP transports, and live providers is in `reports/api-mcp-2026-10-02/REPORT.md`.

## What it does

- Loads and warms up the model once, before the service becomes ready.
- Owns the model through one inference thread, keeping HTTP handling responsive.
- Caps outstanding requests, including a running batch; returns `429` when full.
- Supports optional FIFO batching of compatible requests without running parallel GPU calls.
- Returns queue, batch inference, and service timings; exports Prometheus metrics.
- Applies input/body limits, optional API-key authentication, and request deadlines.
- Forecasts numerical series and timestamped candles from any source.
- Fetches completed market candles through CoinDCX and Binance adapters.
- Exposes native MCP tools over Streamable HTTP and a stdio bridge to the running API.
- Includes a client benchmark, interactive API docs, and CPU-only scheduler tests.

## 1. Try the API without a GPU or model download

### Linux setup launchers

From the extracted project folder, run:

```bash
bash scripts/setup_linux.sh mock
bash scripts/start_linux.sh mock
```

For real GPU inference, run `bash scripts/setup_linux.sh cuda`. This creates a
project-local `.venv`, installs the official PyTorch 2.6.0 CUDA 11.8 wheel and
TimesFM, and checks a CUDA allocation. It requires x86_64 Linux, Python 3.11/3.12,
and a working NVIDIA driver. It does not install system packages or drivers.
The launcher uses an older explicit CUDA build, not the latest PyTorch release;
the model adapter still needs validation on your GPU.

If `.env` does not exist, setup creates a smaller initial profile: context 256,
horizon 16, batch size 1, and queue capacity 4. Existing `.env` settings are
preserved. Review the model license below and set `TF_LICENSE_ACCEPTED=true` in
`.env` only for permitted use, then run:

```bash
bash scripts/start_linux.sh cuda
```

Open http://127.0.0.1:8000/docs after startup. If your default Python is another
version, select the interpreter explicitly, for example:
`TF_SETUP_PYTHON=python3.12 bash scripts/setup_linux.sh cuda`.

To have Work perform these steps on your computer, use a **new local Work task**
with this extracted folder selected. The cloud task that produced this project
has no access to your local terminal or GPU.

### Manual setup

Use Python **3.11 or 3.12**. In a terminal inside this folder:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or Linux:

```bash
source .venv/bin/activate
```

Then install the service:

```bash
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
$env:TF_BACKEND="mock"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 64 --timeout-keep-alive 5
```

Linux:

```bash
TF_BACKEND=mock python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 64 --timeout-keep-alive 5
```

Open **http://127.0.0.1:8000/docs**. This is an interactive interface to try requests.
Mock mode repeats the last observed price. It explicitly labels every result
`last-price-baseline-NOT-TimesFM`; use it to check connectivity only.

## 2. Run the real model on your GTX 1650

Stop the mock server. Install a **CUDA-enabled PyTorch build** using the command
from the [official PyTorch installer](https://pytorch.org/get-started/locally/).
Choose Windows/Linux, pip, and a CUDA option supported by your installed NVIDIA
driver. Then install the pinned TimesFM adapter dependency:

```bash
python -m pip install -e ".[model]"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The last line must show `CUDA available: True` to use `TF_DEVICE=cuda`.

Copy `.env.example` to `.env`. Read the
[TimesFM 3.0 model license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
The checkpoint is restricted to permitted non-commercial, non-production use.
Only set `TF_LICENSE_ACCEPTED=true` if your use satisfies its terms. This flag
records your choice; it does not grant a license. The model weights are not bundled.

In `.env`, keep:

```dotenv
TF_BACKEND=timesfm3
TF_DEVICE=cuda
TF_MAX_CONTEXT=512
TF_MAX_HORIZON=64
TF_MAX_BATCH_SERIES=1
TF_BATCH_WINDOW_MS=0
```

Remove the mock environment override (environment variables override `.env`):

```powershell
Remove-Item Env:TF_BACKEND -ErrorAction SilentlyContinue
```

Linux: `unset TF_BACKEND`.

Start:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 64 --timeout-keep-alive 5
```

The first startup downloads the checkpoint to the Hugging Face cache and can take
time. Later starts reuse it. Warm-up exercises the configured maximum context,
horizon, and batch size before accepting traffic. If warm-up runs out of VRAM,
close other GPU applications and lower `TF_MAX_CONTEXT` to `256` and
`TF_MAX_HORIZON` to `16`. Restart after each configuration change.

**Use exactly one Uvicorn worker.** Each process would load its own model and
compete for the 4 GB of VRAM. CPU mode is explicit (`TF_DEVICE=cpu`); the service
does not silently switch from CUDA to CPU.

## 3. Your first forecast

In `/docs`, expand `POST /v1/forecast`, select **Try it out**, and send:

```json
{
  "series": [{
    "id": "BTC-example",
    "values": [100,101,100,102,103,102,104,105,104,106,107,106,108,109,108,110,111,110,112,113,112,114,115,114,116,117,116,118,119,118,120,121]
  }],
  "horizon": 4,
  "return_quantiles": true
}
```

The endpoint accepts any numerical time series: sensor readings, demand, sales,
or market prices. The `id` is your label; it does not select a provider or model.
`values` must be oldest to newest, finite, and equally spaced. Use at least 32
observations. `horizon: 4` asks for the next four observations: with one-minute
candles, that means four future candle closes. Numbers above are synthetic demo
data. More history is not automatically better; compare context sizes using held-out data.

The response includes `results[0].forecast`, quantiles `0.1` through `0.9`, and:

| Field | Meaning |
| --- | --- |
| `queue_ms` | Wait until the scheduler dispatches the request, including any batching window |
| `inference_batch_ms` | Entire batch call, including model preprocessing, synchronized CUDA work, and output conversion |
| `service_ms` | Time from admission to prediction preparation; excludes HTTP transfer and JSON serialization |
| `batch_series` | Total series processed in that batch |

Quantiles are model predictions, not guaranteed coverage for future observations.

## 4. Data sources and market providers

| Endpoint | Input and purpose |
| --- | --- |
| `POST /v1/forecast` | Ordered numerical arrays from any data source; one or more series |
| `POST /v1/forecast/candles` | Timestamped observations supplied by your application |
| `GET /v1/providers` | Available adapters, supported intervals, symbol formats, history limits |
| `GET /v1/forecast/market` | Fetch market data using `provider` and `symbol`, then forecast |
| `GET /v1/forecast/coindcx` | Compatible legacy route using `pair`; retains its original response |

For supplied candles, use UTC **opening timestamps in milliseconds** and `close`
for the observed value. Values may be zero or negative, so this also works for
non-market data. This complete example forecasts a historical sensor series:

```python
import httpx

payload = {
    "id": "temperature",
    "candles": [
        {"timestamp_ms": 1700000000000 + i * 60000, "close": -10 + i * 0.2} for i in range(32)
    ],
    "interval_ms": 60000,
    "horizon": 4,
    "return_quantiles": True,
    "require_fresh": False,
}
response = httpx.post("http://127.0.0.1:8000/v1/forecast/candles", json=payload)
response.raise_for_status()
print(response.json())
```

Candles are sorted chronologically, identical duplicates are collapsed, unfinished
candles are excluded, and gaps or conflicting duplicates are rejected with `422`.
Without `context`, all completed observations must fit the configured limit.
Set `context` explicitly to select the most recent observations. `require_fresh`
defaults to false for historical evaluation; enable it for live inputs. The response
includes `context_end`, `forecast_close_times`, and the standard `prediction` result.

Try the generic market endpoint in a browser:

```text
http://127.0.0.1:8000/v1/forecast/market?provider=binance&symbol=BTCUSDT&interval=1m&context=128&horizon=4
http://127.0.0.1:8000/v1/forecast/market?provider=coindcx&symbol=B-BTC_USDT&interval=1m&context=128&horizon=4
```

The legacy request remains available:

```text
http://127.0.0.1:8000/v1/forecast/coindcx?pair=B-BTC_USDT&interval=1m&context=128&horizon=4
```

Symbol formats and intervals vary by provider; discover them through `/v1/providers`.
CoinDCX supports `1m`, `15m`, `1h`, `1d`; Binance supports fixed-duration intervals
from `1s` to `1w`. Variable calendar-month intervals are excluded. Both public feeds
have a 1000-row request limit; the adapters cap context at 998 to allow extra candles.
Configured model limits also apply. Use currently available symbols; provider outages,
regional restrictions, stale data, gaps, or malformed upstream responses return `502`.
Market requests always require fresh completed candles with positive closes and return
`data_fetch_ms` separately. They fetch history per request rather than subscribe to a stream.
Provider references: [CoinDCX](https://docs.coindcx.com/) and
[Binance klines](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market).

To add another source, implement an async adapter in `app/providers.py` with the
signature `(http_client, symbol, interval, context) -> (values, context_end_ms)`.
Use `closed_observations` for sorting and validation, raise `CandleError` for upstream
failures, and register a `MarketProvider` in `PROVIDERS`. HTTP and MCP discover it
automatically. Adapter URLs are fixed in code; clients cannot supply arbitrary fetch URLs.

## 5. MCP access

The API includes a native [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
server. Start the API as usual, then connect a client supporting **Streamable HTTP** to:

```text
http://127.0.0.1:8000/mcp
```

If `TF_API_KEY` is configured, send it in the `X-API-Key` header on every MCP HTTP
request. The endpoint uses localhost Host/Origin checks from the SDK. MCP tools are
listed through the MCP protocol; the HTTP routes remain available in `/docs`.

| Tool | Purpose |
| --- | --- |
| `forecast_series` | Forecast one or more arrays from any source |
| `forecast_candles` | Forecast supplied timestamped candles |
| `forecast_market` | Fetch candles through a supported provider and forecast |
| `list_providers` | Discover adapters and their input formats |
| `service_status` | Check readiness, device, and configured model limits |

Forecast tools accept a `request` object matching the corresponding HTTP payload.
For example, arguments for `forecast_market` are:

```json
{
  "request": {
    "provider": "binance",
    "symbol": "BTCUSDT",
    "interval": "1m",
    "context": 128,
    "horizon": 4,
    "return_quantiles": true
  }
}
```

Successful calls provide structured results and readable JSON. Expected failures
set MCP `isError` and include the HTTP status and retry delay where applicable.
Tools share the API's validation, upstream limits, bounded queue, batching,
deadlines, and **single GPU worker**.

For clients that launch **stdio** servers, install this project and configure:

```json
{
  "mcpServers": {
    "timesfm": {
      "command": "/absolute/path/to/timesfm-local-api/.venv/bin/timesfm-mcp",
      "args": ["--api-url", "http://127.0.0.1:8000"]
    }
  }
}
```

Replace the executable path with your installed entry point (Windows:
`.venv\\Scripts\\timesfm-mcp.exe`). The command can also be run as
`python -m app.mcp_server --api-url http://127.0.0.1:8000` in the installed environment.
Start the API first: this stdio bridge forwards calls to it and does not load another
model or reserve GPU memory. It reads the project's `.env`; environment variables
override it. `TF_API_URL` sets the target when `--api-url` is omitted, and `TF_API_KEY`
is forwarded as the API credential. Keep keys out of committed client configuration.

For an existing custom MCP server, [`examples/mcp_bridge.py`](examples/mcp_bridge.py)
still provides `forecast_closes()`. Maintain your own buffer of completed observations
and send numerical arrays directly to avoid per-call market fetch latency.

## 6. Measure latency and throughput

With the server running, use a second terminal with the virtual environment activated:

```bash
python scripts/benchmark.py --context 128 --horizon 4 --concurrency 1 --requests 100
python scripts/benchmark.py --context 128 --horizon 4 --concurrency 4 --requests 100
```

The script warms up the service and reports successful requests/second, p50/p95/p99
client latency, p95 queue and inference times, and all error counts. Results from
mock mode measure the API wrapper, not TimesFM performance. No GTX performance
numbers are assumed or promised.

| Setting | Starting value | When to change it |
| --- | --- | --- |
| `TF_MAX_BATCH_SERIES` | `1` | Try `2` only after measuring VRAM headroom |
| `TF_BATCH_WINDOW_MS` | `0` | Try `2` with batch size `2` to trade a short wait for throughput |
| `TF_GC_INTERVAL_SECONDS` | `30` | CUDA-only explicit cleanup interval; set `0` to restore upstream per-request collection |
| `TF_MAX_CONTEXT` | `512` | Lower it to reduce memory/computation; benchmark 128/256/512 |
| `TF_MAX_HORIZON` | `64` | Lower it when only a few future candles are needed |
| `TF_QUEUE_CAPACITY` | `32` | Lower it for earlier overload rejection and shorter queues |
| `TF_REQUEST_TIMEOUT_SECONDS` | `15` | Adjust to the measured acceptable queue + inference deadline |
| `TF_MAX_UPSTREAM_REQUESTS` | `4` | Limit simultaneous fetches across all market providers |

A larger queue does not make the GPU faster. Keep arrival rate below measured
capacity and watch p95 latency. Batching combines only adjacent requests with the
same horizon and context-length tuple; incompatible requests keep FIFO order.
Requests must contain no more than the configured batch size.

The CUDA adapter paces TimesFM 3.0.2's forced full Python garbage collection to
once per 30 seconds of active inference. Automatic Python GC remains enabled,
and the upstream cleanup still runs immediately when GPU allocation exceeds
90% of device memory. OOM recovery and shutdown also perform collection. This
optimization is scoped to this model instance and requires the pinned adapter;
`TF_GC_INTERVAL_SECONDS=0` restores its original behavior. FP32 inference and
forecast/quantile calculations are retained.

On this GTX 1650, batches of up to four compatible series, a 2 ms batching window,
and an outstanding-request capacity of eight were tested. These values are saved
in the local evaluation `.env`; the example profile keeps batch size one as a
conservative starting point for other machines. The Linux launcher reuses a
project-local `.cache/huggingface` download when present and `HF_HOME` is unset.

On CUDA batch OOM, the adapter attempts a single-series retry. If a single series
does not fit, it returns `503`; reducing configuration requires a restart.
The implementation keeps upstream FP32 inference, without unverified FP16,
quantization, or compilation changes.

## Operations and limits

- `/healthz`: process is alive. `/readyz`: warmed model and live scheduler.
- `/metrics`: Prometheus HTTP latency, queue depth, outstanding requests, inference
  duration, batch size, completion/error counts, and CUDA allocated/reserved/peak bytes.
- Optional `TF_API_KEY` protects forecast endpoints, provider discovery, MCP, and
  metrics via the `X-API-Key` header. In `/docs`, supply the header field when trying
  protected endpoints.
  Health/readiness/docs remain public; default binding is localhost.
- `TF_API_KEY` in the caller's environment is used by the benchmark and bridge.
- `429`: capacity full, with `Retry-After: 1`. `504`: inference deadline exceeded.
  `503`: inference failure/stopping. Uvicorn can also return `503` at its HTTP
  concurrency limit. Caller retries should be bounded and use backoff with jitter.
- Deadlines cover scheduler wait and inference. Market fetching has a separate
  timeout (default 5 seconds); end-to-end client latency also includes that fetch.
- A client deadline cannot preempt a running CUDA kernel. A timed-out running
  request retains its slot until inference finishes. Queued timed-out requests are removed.
- Shutdown stops admission, rejects waiting jobs, and drains active inference.
- Body limit defaults to 128 KiB. Context: 32–1024 values, additionally capped by
  configuration. Horizon: 1–256, additionally capped. Values with magnitude above
  `1e20` are rejected. Server does not silently truncate inputs.
- `TF_MODEL_REVISION` optionally pins a Hugging Face commit SHA. Dependencies other
  than TimesFM have version ranges; this is not a fully locked environment.

## Checks

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

Tests cover validation/authentication, HTTP responsiveness during inference,
backpressure, deadlines, batching, failure recovery, neutral candle normalization,
provider routing and mocked feeds, MCP discovery/tool calls, HTTP protocol handling,
and stdio bridge forwarding. MCP and HTTP are tested against the same inference capacity.
They run without downloading weights or installing PyTorch. Real checkpoint and
GPU validation must be performed locally using the run and benchmark instructions.

## Repository

The private GitHub repository is
[Ananthzeke/timesfm-local-api](https://github.com/Ananthzeke/timesfm-local-api),
with `main` as the default branch. With access to the repository, clone it using:

```bash
git clone https://github.com/Ananthzeke/timesfm-local-api.git
cd timesfm-local-api
```

Follow the setup instructions above to create your environment and obtain the model.
Local `.env` settings, API keys, model checkpoints, caches, and virtual environments
are excluded from Git. Model license acceptance remains a local configuration step.
