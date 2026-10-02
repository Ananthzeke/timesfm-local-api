# TimesFM optimization comparison — 2 October 2026

The real GPU API is faster while retaining the FastAPI service, one Uvicorn worker, one inference thread, FIFO bounded scheduler, and FP32 TimesFM 3.0.2 model. It remains running at http://127.0.0.1:8000; docs: http://127.0.0.1:8000/docs.

## Before and after

The main workload is one synthetic 128-observation series per request, a four-step forecast, and no quantiles in the response. All ranges below show the two separate runs, rather than pooled percentiles.

| Concurrent callers | Before: successful requests/s | After: successful requests/s | Before: median latency (ms) | After: median latency (ms) | Before: p95 latency (ms) | After: p95 latency (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.7–9.1 | 19.7–20.4 | 109.1–110.1 | 48.6–49.9 | 116.4–122.1 | 52.9–56.1 |
| 4 | 9.0–9.2 | 39.1–41.6 | 427.7–428.0 | 97.8–100.1 | 463.6–537.4 | 106.2–113.4 |

At four concurrent callers, throughput improved about **4.4×** and median latency fell from roughly 428 ms to 98–100 ms. With a single caller, median latency fell from 109–110 ms to 49–50 ms (about **55% lower**).

At eight concurrent callers the optimized service handled **66.2–67.3 requests/s**, with 113.5–115.9 ms median latency and 148.2–158.9 ms p95 latency. All 2,000 measured requests succeeded. This comparison also includes increasing outstanding capacity from four to eight: the old eight-caller bursts rejected 99.35% of requests, so their latency and throughput are not directly comparable sustained-load estimates.

For 256 observations and a 16-step horizon, one caller reached 16.2–16.9 requests/s with 58.0–59.1 ms median latency and 66.5–77.9 ms p95 latency. The earlier profile handled 8.6 requests/s with 113–115 ms median latency.

All **5,600 measured requests** across the eight optimized runs returned HTTP 200. No transport errors, HTTP 429/503/504 responses, inference errors, or OOM retries occurred. Warm-up requests are excluded from this count and the reported timings.

## What changed

- The CUDA adapter now schedules TimesFM’s forced full Python collection at 30-second intervals during active inference. Automatic Python collection stays enabled. The upstream memory-pressure cleanup still runs immediately above its 90% allocation threshold. OOM recovery and shutdown explicitly collect and clear the allocator cache.
- The cleanup hook is replaced only in the model instance’s private prediction-function globals. The upstream forecast function code, GPU computations, quantile calculations, and other model instances retain their behavior. No installed dependency files or process-wide GC functions are patched. The optimization checks the exact TimesFM 3.0.2 adapter contract; `TF_GC_INTERVAL_SECONDS=0` restores upstream behavior.
- The local evaluation configuration enables the scheduler’s existing batching: maximum four compatible series, a 2 ms collection window, and an outstanding-request capacity of eight. Input caps remain context 256 / horizon 16. The example configuration keeps a conservative maximum batch size of one for other machines.
- Prometheus exports `tf_gc_collections_total{reason="interval"}` and a corresponding memory-pressure count when triggered. The Linux launcher reuses the existing project-local model cache when `HF_HOME` is unset; the cache is ignored by Git.

No precision reduction, model changes, parallel GPU calls, response caching, input truncation, or request/response schema changes were introduced.

## Why this helped

Profiling ten original GPU predictions took 1.090 seconds in total; `gc.collect()` consumed 0.596 seconds (55%). Removing that forced collection from every call brought inference-only latency to approximately 40–45 ms in the initial experiments. Existing batching amortizes model work across compatible requests. CPU thread counts 1, 2, 4, and the existing default of 6 were compared; no stable benefit justified changing the thread setting.

The direct inference experiments reached roughly 24 series/s at batch size one and 89–91 series/s at batch size four. These are profiling results that exclude HTTP handling and scheduler behavior; they are not the API throughput figures. Under four API callers, the 2 ms window formed batches of two in the measured runs, which explains why live throughput was below the four-series inference-only estimate.

## Correctness and memory validation

- The real GPU comparison tested four input types (constant, trend, sinusoid, and small signed random values), lengths 32, 128, 255, and 256, and horizons 4 and 16. That covered 32 individual series/horizon cases and all nine quantiles.
- Periodic cleanup produced bitwise-identical forecasts and quantiles versus the original prediction method in all cases. Four-series batches matched individual predictions within `rtol=1e-5, atol=1e-5`; the largest absolute difference was 7.62939453e-06.
- Six periodic collections were observed during the API matrix. Allocated memory returned to 1,331,397,120 bytes in all per-run snapshots; final reserved memory was 1,432,354,816 bytes. Peak allocation after startup warm-up was 1,332,812,800 bytes (approximately 1,271 MiB). GPU memory stayed around 1.4 GiB including the CUDA context. Loading-time peak memory is outside this reset metric.
- `/healthz`, `/readyz`, and `/docs` returned HTTP 200 after the complete matrix; the queue and outstanding work count were zero.
- `pytest -q`: **24 passed** (existing tests plus five cleanup-policy tests). Lint, formatting, and Linux launcher syntax checks passed. The unchanged asynchronous baseline suite stalled in the restricted environment; running the suite outside that restriction completed normally.

## Measurement method and limits

Hardware and model revision match the earlier report: GTX 1650 4,096 MiB, NVIDIA driver 580.178.04, AMD Ryzen 5 4600H, Python 3.11.0rc1, PyTorch 2.6.0+cu118, TimesFM 3.0.2, checkpoint revision `43046b85ec22d584a13f8098c2ed39c889e129c2`. Client and server ran on the same machine alongside other applications. Measurements are a snapshot, not a guaranteed capacity.

The previous successful-load runs measured 130 requests each. The optimized runs measured 400 requests at concurrency one and 1,000 at concurrency four/eight, repeated twice, with ten excluded warm-up requests each. The request shape, numerical data, model precision, and model revision were retained. Results include localhost serialization and transfer, exclude startup/model downloading, and do not include CoinDCX fetches or external MCP latency. The adapter still computes quantiles internally even when they are excluded from the response.

Batching benefits depend on overlapping requests with matching horizons and context-length tuples. More varied request shapes may batch less often. If the service is idle beyond the cleanup interval, its next model call may include the deferred collection; the reported steady-load percentiles do not describe every cold or idle request.

## Reproduce or roll back

Restart with `bash scripts/start_linux.sh cuda`. The evaluation `.env` records batch size 4, window 2 ms, queue capacity 8, cleanup interval 30 seconds, the accepted evaluation license, and the pinned checkpoint revision.

Benchmark each command twice, sequentially:

```bash
.venv/bin/python scripts/benchmark.py --context 128 --horizon 4 --concurrency 1 --requests 400 --warmup 10
.venv/bin/python scripts/benchmark.py --context 128 --horizon 4 --concurrency 4 --requests 1000 --warmup 10
.venv/bin/python scripts/benchmark.py --context 128 --horizon 4 --concurrency 8 --requests 1000 --warmup 10
.venv/bin/python scripts/benchmark.py --context 256 --horizon 16 --concurrency 1 --requests 400 --warmup 10
```

To return to the previous inference behavior and initial profile, set the following in `.env` and restart:

```dotenv
TF_GC_INTERVAL_SECONDS=0
TF_MAX_BATCH_SERIES=1
TF_BATCH_WINDOW_MS=0
TF_QUEUE_CAPACITY=4
```

Evidence: `before-profile.txt`, `profile-experiments.json`, `gpu-equivalence.json`, `api-results.json`, per-run `metrics-*.txt`, `final-metrics.txt`, `final-checks.json`, `final-hardware.txt`, `environment.json`, `dependencies.json`, and `server.log`. The earlier baseline remains in `../performance-2026-10-02/`.
