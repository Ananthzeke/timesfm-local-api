# Validation — 2026-10-02

## Initial cloud validation

- Python 3.12; `timesfm==3.0.2` installed source inspected for the forecaster,
  model configuration, inference signature, and forecast/quantile shapes.
- `ruff check .`: passed.
- `ruff format --check .`: passed.
- `pytest -q`: **19 passed**. Tests use a labelled mock backend and fake model
  adapter; no checkpoint downloads or actual PyTorch inference were performed.
- Real Uvicorn HTTP smoke test with mock mode: readiness and `/docs` succeeded;
  benchmark completed **50/50 HTTP 200 responses at concurrency 4** after two
  warm-up requests. This validates server/benchmark connectivity, not GPU speed.
- CoinDCX integration tested with mocked upstream HTTP responses. Real market
  availability and the external CoinDCX MCP server were not validated.
- No GTX 1650 or CUDA device was available. Model loading, peak VRAM, forecast
  quality, and latency/throughput must be checked on the target computer.
- GitHub push was not performed: authenticated access was declined and no
  destination repository URL was supplied. README includes publishing commands.
- Linux setup/start launchers added after the initial build. Bash syntax and
  early-exit paths checked in the cloud. CUDA installation/allocation and local
  setup were not executed on the user's computer.

## Local GPU validation and optimization

- Real TimesFM 3.0.2 inference verified on NVIDIA GTX 1650 (4 GB), PyTorch
  2.6.0+cu118, driver 580.178.04, with explicit user acceptance for evaluation.
- The initial API profile handled approximately 9 forecasts/second. Profiling
  showed full Python garbage collection consumed 55% of inference time.
- Instance-scoped periodic cleanup, four-series batching, a 2 ms batching window,
  and capacity eight reduced single-caller median latency to 49–50 ms and reached
  39–42 requests/second at four callers or 66–67 at eight callers.
- All 5,600 measured optimized requests succeeded. Six periodic collections
  occurred with stable GPU memory; no inference errors or OOM retries occurred.
- GPU output comparisons covered 32 series/horizon cases. Cleanup changes were
  bitwise equivalent; batched results matched within FP32 tolerance, with maximum
  absolute forecast/quantile difference below 0.000008.
- `pytest -q`: **24 passed**. Lint, formatting, and launcher syntax checks passed.
- Full methods, results, limits, and rollback instructions are in
  `reports/optimization-2026-10-02/REPORT.md`; the earlier measurements are in
  `reports/performance-2026-10-02/REPORT.md`.

## Generalized API and native MCP validation

- `pytest -q`: **47 passed**. Lint and formatting checks passed.
- Provider-neutral numerical arrays and timestamped observations produced real
  CUDA forecasts. Live Binance and CoinDCX feeds and the legacy route succeeded.
- All five native MCP tools were verified over Streamable HTTP in current and
  legacy negotiation modes. A separate stdio bridge process successfully forwarded
  calls to the same GPU application without loading another model.
- Forecast arrays and quantiles matched exactly between HTTP and MCP for identical
  inputs. Tests covered shared capacity, validation, authentication, and errors.
- A short warmed HTTP check completed 300/300 requests successfully and reached
  68.77 requests/second at eight concurrent callers. MCP performance was not
  separately benchmarked; full conditions and results are in
  `reports/api-mcp-2026-10-02/REPORT.md`.
