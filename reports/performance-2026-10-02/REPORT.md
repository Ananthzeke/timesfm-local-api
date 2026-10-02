# Local TimesFM performance — 2 October 2026

The real TimesFM 3.0 model loaded, warmed up, and completed GPU forecasts on the NVIDIA GeForce GTX 1650. The application remains running at http://127.0.0.1:8000; interactive docs: http://127.0.0.1:8000/docs.

## GPU benchmark results

For a 128-observation context and four-step forecast, this batch-size-one configuration sustained approximately **9 successful requests per second**. Concurrency increased queueing and latency while throughput stayed similar.

| Context | Horizon | Concurrent callers | Successful requests/s, range | p50 latency (ms), range | p95 latency (ms), range | HTTP 429 / total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4 | 1 | 8.7–9.1 | 109.1–110.1 | 116.4–122.1 | 0 / 260 |
| 128 | 4 | 2 | 9.0–9.2 | 214.6–217.0 | 230.3–264.8 | 0 / 260 |
| 128 | 4 | 4 | 9.0–9.2 | 427.7–428.0 | 463.6–537.4 | 0 / 260 |
| 256 | 16 | 1 | 8.6–8.6 | 113.0–115.4 | 123.4–129.7 | 0 / 260 |

Ranges show the two individual runs; percentiles were not pooled. All 1,040 measured requests in the four scenarios above succeeded. No transport errors, inference failures, deadline failures, or out-of-memory retries occurred.

## Overload behavior

At eight concurrent callers, the outstanding-work capacity of four was exceeded. The client submits a replacement request immediately after every response, including fast rejections, and performs no retries or backoff. The two 1,000-request bursts rejected 1,987 of 2,000 requests (99.35%) with HTTP 429; 13 succeeded. Each burst lasted about 2.2 seconds. This is an aggressive overload case, not a sustainable throughput estimate. Successful-response p95 latency was 1.77–2.00 seconds, based on only six or seven successes per run, so these percentiles are descriptive and have a small sample. Rejection traffic competes with inference work on the same machine.

For this tested configuration, one caller gives the lowest latency; two to four callers mainly increase waiting time. A caller receiving HTTP 429 should honor `Retry-After: 1` and use bounded backoff, as the application README describes. Higher batch sizes and batching windows were not tested.

## Environment and method

- Hardware: AMD Ryzen 5 4600H, six cores / 12 threads; 7.2 GiB system memory; NVIDIA GTX 1650 with 4,096 MiB GPU memory; driver 580.178.04.
- Runtime: Python 3.11.0rc1, PyTorch 2.6.0+cu118, TimesFM 3.0.2.
- Model: `google/timesfm-3.0-pytorch`, revision `43046b85ec22d584a13f8098c2ed39c889e129c2`; FP32 inference, upstream SDPA configuration, symmetric averaging disabled.
- Service: one Uvicorn worker, HTTP concurrency limit 64, max context 256, max horizon 16, max batch series 1, batch window 0 ms, outstanding-work capacity 4, request deadline 15 seconds. Access logging disabled.
- Workload: direct `POST /v1/forecast`, one synthetic series `100 + sin(i / 10)`, quantiles excluded from the response. The adapter still computes model quantiles internally. No CoinDCX fetch or external MCP round trip is included.
- Model startup included two maximum-shape warm-up runs. Calibration used five client warm-up requests and ten measured requests. Each reported run used ten excluded client warm-up requests. Non-overload runs measured 130 requests each, taking approximately 14–15 seconds; every scenario was repeated twice.
- End-to-end client latency includes JSON serialization and transfer over localhost, and is reported only for successful responses. Throughput is successful requests divided by the entire measured run duration. Closed-loop clients submit new requests when earlier responses complete.
- Client and server shared the machine with other active applications. Results are a local snapshot for these settings, not forecast-quality measurements or a maximum hardware-capacity claim.

## Memory and final verification

During inference, NVIDIA-SMI reported 1,444 MiB GPU memory in use. The final PyTorch metrics recorded 1269.7 MiB allocated, 1366.0 MiB reserved, and 1270.9 MiB peak allocation since startup warm-up. The peak counter is reset after warm-up and therefore does not describe loading-time peak memory.

After all measured runs, `/healthz`, `/readyz`, and `/docs` returned HTTP 200. The queue and outstanding request count were zero; inference errors and OOM retries were zero. A separate smoke request returned four finite forecast values and all nine quantile arrays.

## Reproduce

The project-local `.venv` and evaluation `.env` are prepared. The latter records the explicit user acceptance of the TimesFM license for this non-commercial, non-production evaluation and pins the tested model revision. The model is cached under `.cache/huggingface`.

Start the service after stopping the current instance:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 64 --timeout-keep-alive 5 --no-access-log
```

Repeat each command twice (run sequentially):

```bash
.venv/bin/python scripts/benchmark.py --requests 130 --warmup 10 --context 128 --horizon 4 --concurrency 1
.venv/bin/python scripts/benchmark.py --requests 130 --warmup 10 --context 128 --horizon 4 --concurrency 2
.venv/bin/python scripts/benchmark.py --requests 130 --warmup 10 --context 128 --horizon 4 --concurrency 4
.venv/bin/python scripts/benchmark.py --requests 130 --warmup 10 --context 256 --horizon 16 --concurrency 1
.venv/bin/python scripts/benchmark.py --requests 1000 --warmup 10 --context 128 --horizon 4 --concurrency 8
```

Files: `gpu-results.json` (all ten runs including p99 and queue/inference timings); `gpu-calibration.json`; `gpu-smoke.json`; `gpu-environment.json`; `gpu-dependencies.json`; `gpu-metrics-before.txt`, `gpu-metrics-after.txt`, `gpu-final-metrics.txt`; `gpu-hardware.txt`, `gpu-final-hardware.txt`; `gpu-server.log`; and `final-checks.json`.

## API baseline (mock backend)

These results measure the HTTP API and scheduler using `last-price-baseline-NOT-TimesFM`. They do not measure TimesFM inference.

Hardware: AMD Ryzen 5 4600H (6 cores / 12 threads), 7.2 GiB RAM; NVIDIA GTX 1650 (4,096 MiB), driver 580.178.04. Client and server ran on the same machine while other applications were active.

Configuration: one Uvicorn worker, HTTP concurrency limit 64, maximum context 256, maximum horizon 16, maximum batch size 1, no batching window, outstanding request capacity 4, deadline 15 seconds. Access logs disabled.

Workload: POST `/v1/forecast`; one synthetic series with 128 observations; forecast horizon 4; quantiles excluded. Each run used 20 excluded warm-up requests followed by 3,000 measured requests. Two runs per concurrency. Closed-loop clients submitted a replacement request on completion; no retries. Client latency percentiles include successful responses only.

| Concurrent callers | Run | Successful requests/s | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | HTTP 200 | HTTP 429 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 324.1 | 2.64 | 5.76 | 6.29 | 3000 | 0 |
| 1 | 2 | 367.1 | 2.43 | 4.56 | 5.84 | 3000 | 0 |
| 4 | 1 | 781.8 | 4.91 | 6.87 | 8.40 | 3000 | 0 |
| 4 | 2 | 740.1 | 5.08 | 7.74 | 9.26 | 3000 | 0 |
| 8 | 1 | 766.5 | 9.15 | 17.82 | 31.83 | 2961 | 39 |
| 8 | 2 | 772.7 | 9.06 | 16.75 | 34.95 | 2973 | 27 |

At concurrency 4 both runs succeeded without errors. At concurrency 8 throughput changed little, latency rose, and 66 of 6,000 requests (1.1%) returned HTTP 429. The small outstanding-work cap intentionally rejects excess work.

Raw results: `mock-results.json`; Prometheus snapshot: `mock-metrics.txt`.
