"""Measure the running service on YOUR hardware. No TimesFM dependency here."""

import argparse
import asyncio
import json
import os
import time
from collections import Counter

import httpx
import numpy as np


async def benchmark(args):
    headers = {"X-API-Key": os.environ["TF_API_KEY"]} if os.getenv("TF_API_KEY") else {}
    payload = {
        "series": [
            {"id": "benchmark", "values": (100 + np.sin(np.arange(args.context) / 10)).tolist()}
        ],
        "horizon": args.horizon,
        "return_quantiles": False,
    }
    async with httpx.AsyncClient(
        base_url=args.url,
        headers=headers,
        timeout=120,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:
        for _ in range(args.warmup):
            (await client.post("/v1/forecast", json=payload)).raise_for_status()
        remaining = args.requests
        latencies, queues, inference = [], [], []
        statuses = Counter()

        async def worker():
            nonlocal remaining
            while remaining:
                remaining -= 1
                start = time.perf_counter()
                try:
                    response = await client.post("/v1/forecast", json=payload)
                    statuses[str(response.status_code)] += 1
                    if response.is_success:
                        latencies.append((time.perf_counter() - start) * 1000)
                        timing = response.json()["timing"]
                        queues.append(timing["queue_ms"])
                        inference.append(timing["inference_batch_ms"])
                except httpx.HTTPError:
                    statuses["transport_error"] += 1

        start = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(args.concurrency)])
        elapsed = time.perf_counter() - start
    result = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "context": args.context,
        "horizon": args.horizon,
        "elapsed_seconds": elapsed,
        "statuses": dict(statuses),
        "successful_requests_per_second": len(latencies) / elapsed,
    }
    if latencies:
        result.update(
            {
                "client_latency_ms": dict(
                    zip(
                        ["p50", "p95", "p99"],
                        np.percentile(latencies, [50, 95, 99]).tolist(),
                        strict=True,
                    )
                ),
                "queue_p95_ms": float(np.percentile(queues, 95)),
                "inference_batch_p95_ms": float(np.percentile(inference, 95)),
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or not 32 <= args.context <= 1024:
        parser.error("requests/concurrency must be positive; context must be 32..1024")
    if not 1 <= args.horizon <= 256 or args.warmup < 0:
        parser.error("horizon must be 1..256; warmup cannot be negative")
    asyncio.run(benchmark(args))
