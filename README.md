# Local TimesFM API

A Python API for local **TimesFM 3.0 inference**, using PyTorch and FastAPI. Designed
to start conservatively on a GTX 1650 with 4 GB VRAM, with controls you can benchmark
and adjust. It downloads pretrained weights; you do not need to train the model.

The wrapper uses the forecasting API in `timesfm==3.0.2`. CUDA inference has been
verified locally on a GTX 1650 with 4 GB VRAM. Hardware-specific measurements and
their workload details are in `reports/performance-2026-10-02/REPORT.md` and
`reports/optimization-2026-10-02/REPORT.md`.

## What it does

- Loads and warms up the model once, before the service becomes ready.
- Owns the model through one inference thread, keeping HTTP handling responsive.
- Caps outstanding requests, including a running batch; returns `429` when full.
- Supports optional FIFO batching of compatible requests without running parallel GPU calls.
- Returns queue, batch inference, and service timings; exports Prometheus metrics.
- Applies input/body limits, optional API-key authentication, and request deadlines.
- Fetches completed CoinDCX candles or accepts closes supplied by your MCP server.
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

Quantiles are model predictions, not guaranteed probabilities for future crypto prices.

## 4. CoinDCX and your MCP server

Try this read-only request in a browser:

```text
http://127.0.0.1:8000/v1/forecast/coindcx?pair=B-BTC_USDT&interval=1m&context=128&horizon=4
```

Use a currently valid `pair` from CoinDCX market details; `B-BTC_USDT` is an example,
not a guarantee that a market is available. Supported candle intervals are
`1m`, `15m`, `1h`, and `1d`. This endpoint fetches historical candles per request;
it is not a streaming subscription.

The service sorts candles chronologically, excludes the unfinished candle,
and rejects insufficient, stale, conflicting, or gapped data. It returns fetch
time separately and timestamps for forecast candle closes. Invalid upstream data
returns `502` instead of a misleading prediction.

For lower latency with your MCP server, keep a rolling buffer of completed
candles in that server and pass their closes directly to `POST /v1/forecast`.
[`examples/mcp_bridge.py`](examples/mcp_bridge.py) provides `forecast_closes()`
for a Python caller. Call it from the MCP tool handler after sorting and validating
your candles. MCP is the tool protocol; TimesFM consumes the numeric array.
The bridge is an integration helper, not a tested modification of the external
`ayagup/coindcx-mcp` repository. No order-placement endpoint is included.

## 5. Measure latency and throughput

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
| `TF_MAX_UPSTREAM_REQUESTS` | `4` | Limit simultaneous CoinDCX HTTP fetches |

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
- Optional `TF_API_KEY` protects forecast endpoints and metrics via the `X-API-Key`
  header. In `/docs`, supply the header field when trying protected endpoints.
  Health/readiness/docs remain public; default binding is localhost.
- `TF_API_KEY` in the caller's environment is used by the benchmark and bridge.
- `429`: capacity full, with `Retry-After: 1`. `504`: inference deadline exceeded.
  `503`: inference failure/stopping. Uvicorn can also return `503` at its HTTP
  concurrency limit. Caller retries should be bounded and use backoff with jitter.
- Deadlines cover scheduler wait and inference. CoinDCX fetch has a separate
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
backpressure, deadlines, batching, failure recovery, and mocked CoinDCX responses.
They run without downloading weights or installing PyTorch. Real checkpoint and
GPU validation must be performed locally using the run and benchmark instructions.

## Publish to GitHub

For a **new empty GitHub repository**, replace the URL below with your own URL,
then run these commands inside this folder using your normal Git authentication:

```bash
git init -b timesfm-api
git add app tests scripts examples .github .gitignore .env.example pyproject.toml README.md
git commit -m "Add local TimesFM API with bounded inference scheduling"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin timesfm-api
```

For an existing repository, copy these files into a new branch of its checkout,
respect its instructions and existing files, then commit and push that branch.
Do not commit `.env`, API keys, model checkpoints, or virtual environments.

The application was not pushed from the development session because authenticated
GitHub access and a destination repository were unavailable.
