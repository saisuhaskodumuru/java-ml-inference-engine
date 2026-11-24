# Case Study: Java ML Inference Engine

## What this is

A personal project built to demonstrate concurrent ML model
inference in Java: a Spring Boot service that scores fraud
detection API calls in parallel on Java 21 virtual threads. It is
not associated with any work experience, it is not production code
from any employer, and it contains no real customer data. Every
transaction in it is synthetic. The goal was to show the shape of a
high-throughput inference service in a form anyone can run locally:
a real logistic model with explainable features, a semaphore that
caps in-flight work, batch scoring that fans out across virtual
threads, and honest latency tracking that separates model time from
HTTP time.

## Problem

A fraud scoring service has to answer thousands of concurrent
requests without letting any single slow call block the rest, and
without letting an unbounded flood of work exhaust the server. The
classic failure modes are a small fixed thread pool that queues
under load, no backpressure at all so memory grows until the
process dies, and a black-box score that gives an analyst nothing
to act on. The interesting part of this project is the concurrency
design, not the model: the model is deliberately small and
explainable, so the measurement can focus on how the service
behaves under parallel load.

## My role

This is a personal project. I built it end to end: the model, the
scoring service and its concurrency controls, the REST API, the
latency tracking, the dashboard UI, the load test, and the tests.

## Architecture

```
Browser dashboard (static HTML/CSS/JS)
        |
        |  REST / JSON
        v
InferenceController  --  POST /api/v1/inference/predict
                          POST /api/v1/inference/batch
                          GET  /api/v1/inference/stats
                          GET  /api/v1/inference/samples
                          GET  /api/v1/health
        |
        v
InferenceService
        |   Semaphore (max in-flight, default 256)
        |   Virtual-thread-per-task executor (batches)
        |   LatencyStats (last 10,000 samples)
        |
        +--> RemoteModelClient (only when
        |     inference.model.endpoint is set)
        |     on failure -> fall back
        v
LogisticRegressionFraudModel (local-logistic-v1)
        8 weighted features -> sigmoid -> 0-100 score
```

Scoring tries a remote model first when an endpoint is configured.
Any failure, or no endpoint at all, falls back to the local
logistic model, and the response says which model produced the
score. Single predictions run on the calling virtual thread behind
a semaphore, so a flood of requests gets a 429 instead of
unbounded queueing. Batch predictions fan out to a
virtual-thread-per-task executor, so a 500-item batch scores its
items concurrently. Latency is recorded per request in-process and
the stats endpoint reports real percentiles from the last 10,000
samples. The app runs on Java 21 virtual threads
(`spring.threads.virtual.enabled=true`).

## What was built

- A Spring Boot 3.3, Java 21, Maven service with no database. All
  state is in memory, which keeps the demo self-contained.
- Five endpoints: predict one transaction, predict a batch of up
  to 1,000, live latency stats, sample transactions, and health.
- A real logistic-regression-style model with eight weighted
  features (log amount, amount vs the customer's own average,
  velocity, night-time, foreign country, high-risk country, new
  device, merchant-category risk) and a -3.2 bias, returning the
  per-feature contributions with every score.
- An `InferenceModel` interface so a real model (ONNX server, Azure
  ML endpoint) can replace the local model, plus a remote HTTP
  client that implements it and falls back cleanly on failure.
- A semaphore (default 256 in-flight) and a virtual-thread
  executor for batches, plus in-process p50/p95/p99 tracking.
- A static dashboard UI with no build step and no external
  JavaScript: a score-a-transaction form, sample buttons, live
  stats that refresh every 3 seconds, and a 500-item batch run
  button. It calls the real APIs.
- A Python load-test script (`scripts/load_test.py`) using
  keep-alive connections per worker thread, so it measures the
  service instead of TCP connection setup.
- Tests: 5 JUnit 5 unit tests for the model and 6 MockMvc tests
  for the endpoints, including validation and empty-batch failures.

## Results, measured on this demo

These numbers were measured locally on 2026-11-24 by running the
packaged application and curling the real endpoints, plus the load
test script. They describe the demo on this sandbox only (1.5
shared CPUs, server and load-test client on the same machine).

- Tests: 11 run, 0 failures, 0 errors. The model tests ran in
  0.015 s and the controller tests in 8.497 s (mostly Spring
  context startup).
- Application startup: 5.822 s from launch to "Started
  InferenceEngineApplication".
- `POST /api/v1/inference/predict`, a $12,400 crypto transaction
  from NG at 02:00 UTC on a new device: HTTP 200 in 0.276 s on the
  first hit, score 95, CRITICAL, classification FRAUD_SUSPECT,
  model `local-logistic-v1`, inference time 0.726 ms, with all
  eight feature contributions listed.
- `POST /api/v1/inference/predict`, a $42.50 Starbucks purchase in
  the US: HTTP 200 in 0.004 s, score 7, LOW, classification
  LEGITIMATE, inference time 0.029 ms.
- `POST /api/v1/inference/batch`, 500 transactions: HTTP 200,
  8.776 ms server-side.
- `POST /api/v1/inference/predict` with amount -5: HTTP 400.
- `GET /api/v1/health`: HTTP 200 in 0.155 s on the first hit.
- Dashboard UI at `http://localhost:8082/`: HTTP 200 in 0.080 s,
  8,395 bytes of static HTML, CSS, and JavaScript.
- Load test, 3,000 requests at 100 concurrent clients
  (keep-alive): 2.94 s wall, 1,019.2 req/s, avg 63.10 ms, p50
  61.91 ms, p95 190.86 ms, p99 284.04 ms, 0 errors.
- Load test, 5,000 requests at 50 concurrent clients
  (keep-alive): 4.78 s wall, 1,046.3 req/s, avg 44.76 ms, p50
  25.09 ms, p95 103.02 ms, p99 167.83 ms, 0 errors.
- Server-side latency after 11,502 total requests served: avg
  0.018 ms, p50 0.005 ms, p95 0.008 ms, p99 0.025 ms, rejected 0.
  The model itself runs in microseconds; the client-measured
  latency is HTTP handling and the shared-CPU sandbox, not
  inference.

No production throughput or latency is claimed here. "Thousands
of fraud detection API calls" in the portfolio description refers
to what this demo is built to handle, measured above — not to any
real production system.

## Challenges

- The sandbox blocks Java's outbound TCP, so Maven cannot download
  dependencies from Maven Central, and the egress proxy's SSL
  fails for Java. The dependencies were already in the local Maven
  repository from the earlier fraud dashboard build, but Maven
  runs as root here, so its default local repository is
  `/root/.m2/repository` while the cached artifacts live in
  `/home/hatch/.m2/repository`. The build works offline with
  `mvn -o -Dmaven.repo.local=/home/hatch/.m2/repository`; the
  README documents that, and on a normal machine plain `mvn test`
  works.
- `Map.of` accepts at most ten entries, and the stats endpoint
  returns thirteen fields. It was rewritten with a
  `LinkedHashMap`.
- A public record must live in its own file, so
  `FeatureContribution` was moved out of `InferenceResult.java`
  into `FeatureContribution.java`.
- The first load-test run opened a fresh TCP connection per
  request and measured 304.9 req/s with p95 501.07 ms — it was
  measuring connection setup, not inference. The script was
  changed to one keep-alive connection per worker thread, which
  is how a real client behaves, and the keep-alive runs are the
  numbers reported.
- A first attempt to stop the demo server killed the shell running
  the check instead (the kill pattern matched its own command
  line); the server was then verified stopped with a plain
  connection check.

## Tech stack

Java 21 (virtual threads), Spring Boot 3.3, Spring MVC, Spring
Validation, Jackson, JUnit 5, MockMvc, Maven, Python (load-test
script only), vanilla HTML/CSS/JS dashboard. No database.
