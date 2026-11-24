#!/usr/bin/env python3
"""Load test for the Java ML Inference Engine.

Sends N POST /api/v1/inference/predict requests with C concurrent
workers against a running server (default http://localhost:8082),
then prints throughput, average/p50/p95/p99 latency, and error rate.

Each worker thread keeps one persistent HTTP connection (keep-alive),
which is how a real client behaves — without it, every request pays
for a fresh TCP connection and the numbers measure connection setup,
not inference.

Usage:
    python3 scripts/load_test.py [--url URL] [--requests N] [--concurrency C]
"""
import argparse
import concurrent.futures
import http.client
import json
import random
import statistics
import threading
import time
import urllib.parse

CATEGORIES = ["GROCERIES", "RETAIL", "TRAVEL", "ELECTRONICS", "CRYPTO"]
COUNTRIES = ["US", "US", "US", "GB", "CA", "NG"]

LOCAL = threading.local()


def make_payload(i, rng):
    fraud_like = rng.random() < 0.08
    return {
        "id": f"TXN-LOAD-{i}",
        "customerId": f"CUST-{rng.randint(1, 50):03d}",
        "amount": round(3000 + rng.random() * 9000, 2) if fraud_like else round(5 + rng.random() * 2000, 2),
        "currency": "USD",
        "merchant": "Load Test Merchant",
        "merchantCategory": "CRYPTO" if fraud_like else rng.choice(CATEGORIES),
        "country": "NG" if fraud_like else rng.choice(COUNTRIES),
        "deviceId": f"DEV-{rng.randint(1000, 9999)}",
        "hourOfDay": rng.randint(0, 23),
        "transactionsLastHour": rng.randint(0, 5),
        "customerAvgAmount": 250.0,
        "newDevice": rng.random() < 0.15,
        "foreignCountry": fraud_like,
    }


def connection(host, port):
    conn = getattr(LOCAL, "conn", None)
    if conn is None:
        conn = http.client.HTTPConnection(host, port, timeout=30)
        LOCAL.conn = conn
    return conn


def one(url, payload):
    parsed = urllib.parse.urlparse(url)
    start = time.perf_counter()
    try:
        conn = connection(parsed.hostname, parsed.port or 80)
        conn.request("POST", parsed.path, body=json.dumps(payload),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        return (time.perf_counter() - start) * 1000, resp.status
    except Exception:
        LOCAL.conn = None  # drop a broken connection; next call opens a fresh one
        return (time.perf_counter() - start) * 1000, 0


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(p / 100 * len(sorted_values))))
    return sorted_values[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8082/api/v1/inference/predict")
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()

    rng = random.Random(7)
    payloads = [make_payload(i, rng) for i in range(args.requests)]
    latencies = []
    errors = 0
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for latency, status in pool.map(lambda p: one(args.url, p), payloads):
            latencies.append(latency)
            if status != 200:
                errors += 1
    wall = time.perf_counter() - start
    latencies.sort()
    print(f"requests:     {args.requests}")
    print(f"concurrency:  {args.concurrency}")
    print(f"wall time:    {wall:.2f} s")
    print(f"throughput:   {args.requests / wall:.1f} req/s")
    print(f"avg latency:  {statistics.mean(latencies):.2f} ms")
    print(f"p50 latency:  {percentile(latencies, 50):.2f} ms")
    print(f"p95 latency:  {percentile(latencies, 95):.2f} ms")
    print(f"p99 latency:  {percentile(latencies, 99):.2f} ms")
    print(f"errors:       {errors} ({errors / args.requests * 100:.2f}%)")


if __name__ == "__main__":
    main()
