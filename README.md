# Java ML Inference Engine

A real working demo of optimized Java 21 services for concurrent ML
model inference: it scores fraud detection API calls in parallel on
virtual threads. It is a **personal project** built to demonstrate
concurrent ML inference in Java. It is not associated with any work
experience, it is not production code from any employer, and it
contains no real customer data. Every transaction in it is synthetic.

## What it does

- Scores one transaction 0-100 with a risk band (LOW, MEDIUM, HIGH,
  CRITICAL), a classification, the per-feature contributions that
  produced the score, and the model that scored it.
- Scores a batch of up to 1,000 transactions in parallel on virtual
  threads, returning each score plus the server-side total time.
- Tracks in-process latency: total requests served, average, p50, p95,
  and p99 latency from real measured samples, and how many requests
  were rejected when saturated.
- Serves a dashboard UI at the root URL that calls the real APIs:
  a score-a-transaction form, sample transactions, live stats, and a
  500-item batch run.

## Architecture

```
Browser / curl / load test
      |
      v
Static dashboard (src/main/resources/static/index.html)
      |
      v
Spring Boot 3.3, Java 21, virtual threads
  InferenceController  (REST, /api/v1/...)
      |
      +--> InferenceService
      |       |
      |       +--> RemoteModelClient (only when
      |       |     inference.model.endpoint is set; on any
      |       |     failure it falls back to the local model)
      |       |
      |       +--> LogisticRegressionFraudModel
      |       |     (local-logistic-v1, deterministic)
      |       |
      |       +--> Semaphore (max in-flight inference,
      |       |     default 256) + virtual-thread-per-task
      |       |     executor for batches
      |       |
      |       +--> LatencyStats (last 10,000 samples,
      |             real p50/p95/p99)
      |
      +--> SeedDataGenerator (synthetic transactions,
              fixed seed, for the dashboard and load test)
```

No database. No persistence. All state lives only for the life of
the process. Java 21 virtual threads are enabled with
`spring.threads.virtual.enabled=true`, so each request runs on its
own virtual thread instead of a small platform-thread pool.

## The model (local-logistic-v1)

A real logistic-regression-style model, not a stub: it extracts eight
features from a transaction, applies fixed weights, and passes the
weighted sum through a sigmoid to get a fraud probability. The
weights are hand-set for this demo (not fitted on real data — there
is no real data here), and every contribution is returned with the
score so a result is explainable.

| Feature | Definition | Weight |
| --- | --- | --- |
| logAmount | log10(amount + 1) / 5 | 1.4 |
| amountRatio | min(amount / customerAvgAmount, 10) / 10 | 1.2 |
| velocity | min(transactionsLastHour, 10) / 10 | 1.0 |
| night | 1 when hourOfDay is 00:00-04:59 UTC | 0.5 |
| foreignCountry | 1 when the country is foreign to the customer | 0.7 |
| highRiskCountry | 1 for NG, RU, KP, IR, MM | 0.6 |
| newDevice | 1 when the device is new for the customer | 0.8 |
| merchantRisk | 1.0 for CRYPTO/GAMBLING, 0.4 for ELECTRONICS/TRAVEL | 1.1 |

Bias is -3.2, so an ordinary transaction scores low and several risk
signals have to combine before a score turns high. Risk bands: 0-29
LOW, 30-54 MEDIUM, 55-74 HIGH, 75-100 CRITICAL. Classification: LOW
is LEGITIMATE, MEDIUM is REVIEW, HIGH and CRITICAL are FRAUD_SUSPECT.

## How to run

Requirements: Java 21 and Maven 3.9 or later.

```bash
mvn spring-boot:run
```

Then open http://localhost:8082 in a browser.

Or build and run the jar:

```bash
mvn package
java -jar target/java-ml-inference-engine-1.0.0.jar
```

Note for this sandbox: the Maven dependencies were fetched earlier
into `/home/hatch/.m2/repository`, but the shell here runs Maven as
root, so Maven's default local repository is `/root/.m2/repository`
and it cannot see them. Build offline with the explicit local repo:

```bash
mvn -o -Dmaven.repo.local=/home/hatch/.m2/repository test
mvn -o -Dmaven.repo.local=/home/hatch/.m2/repository package
```

On a normal machine with a populated `~/.m2`, plain `mvn test` and
`mvn package` work.

## API examples

Score one transaction (a high-value foreign crypto transaction,
which should score CRITICAL):

```bash
curl -s -X POST http://localhost:8082/api/v1/inference/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "TXN-1",
    "customerId": "CUST-001",
    "amount": 12400.00,
    "currency": "USD",
    "merchant": "Crypto Exchange X",
    "merchantCategory": "CRYPTO",
    "country": "NG",
    "deviceId": "DEV-UNKNOWN",
    "hourOfDay": 2,
    "transactionsLastHour": 0,
    "customerAvgAmount": 250.0,
    "newDevice": true,
    "foreignCountry": true
  }'
```

Score a batch (up to 1,000 transactions, scored in parallel):

```bash
curl -s -X POST http://localhost:8082/api/v1/inference/batch \
  -H 'Content-Type: application/json' \
  -d '[ { ...transaction... }, { ...transaction... } ]'
```

Live stats and health:

```bash
curl -s http://localhost:8082/api/v1/inference/stats
curl -s http://localhost:8082/api/v1/health
curl -s http://localhost:8082/api/v1/inference/samples
```

## How a real model would plug in

The scoring service tries a remote model first, and only when an
endpoint is configured. Set these in `application.properties` or as
environment properties:

```bash
inference.model.endpoint=https://<model-server>/score
inference.model.api-key=<key>
```

The remote client (`RemoteModelClient`) posts the transaction as
JSON and expects a JSON response with a numeric `probability` field
from 0 to 1. When the endpoint returns a probability, the response
says `"model": "remote-model"`. When no endpoint is configured, or
the call fails for any reason, the service falls back to the local
logistic model and the response says
`"model": "local-logistic-v1"`. The API key is sent only as an
Authorization header to the configured endpoint and is never logged.
An ONNX model server would plug in the same way: implement the
`InferenceModel` interface and return a probability in [0, 1].

## Benchmarks, measured on this demo

Load test script: `scripts/load_test.py` — a Python client that
sends POSTs to `/api/v1/inference/predict` with N concurrent
workers, each holding one persistent (keep-alive) HTTP connection.
Measured locally on 2026-11-24 on this sandbox (1.5 shared CPUs,
server and load-test client on the same machine, so the client
numbers include HTTP and client-side CPU contention).

Single-predict load test, 3,000 requests at 100 concurrent clients:

- Wall time 2.94 s, throughput **1,019.2 req/s**
- Avg latency 63.10 ms, p50 61.91 ms, p95 190.86 ms, p99 284.04 ms
- Errors: 0 (0.00%)

Single-predict load test, 5,000 requests at 50 concurrent clients:

- Wall time 4.78 s, throughput **1,046.3 req/s**
- Avg latency 44.76 ms, p50 25.09 ms, p95 103.02 ms, p99 167.83 ms
- Errors: 0 (0.00%)

Server-side inference latency (from `GET /api/v1/inference/stats`,
after 11,502 total requests served): avg 0.018 ms, p50 0.005 ms,
p95 0.008 ms, p99 0.025 ms, rejected 0. The model itself runs in
microseconds; the client-measured latency is dominated by HTTP
handling and the shared-CPU sandbox, not by inference.

Batch endpoint: 500 transactions scored in **8.776 ms** server-side
(385.9 ms wall clock on the first hit, including connection setup
and JSON).

One measurement lesson, kept in the numbers: the first load-test
run used a fresh TCP connection per request (Python `urllib`) and
measured only 304.9 req/s with p95 501.07 ms at the same 3,000
requests / 100 clients. That run measured connection setup, not
inference, so the script was changed to keep-alive connections per
worker and the runs above are the ones that count.

No production throughput or latency is claimed anywhere. These
numbers describe this demo on this sandbox only.

## Tests

```bash
mvn -o -Dmaven.repo.local=/home/hatch/.m2/repository test
```

Tests cover the model (a fraud-like transaction scores far higher
than a normal one, scoring is deterministic, velocity raises the
score, contributions sum to the logit, risk bands map correctly)
and the endpoints (a fraud transaction returns CRITICAL with all
eight contributions, a normal one returns LOW, an invalid amount
returns 400, a batch scores all items, an empty batch returns 400,
stats and health respond).

## Data

All data is synthetic. The seed generator creates transactions from
a fixed random seed (42), with about 8% planted fraud-like patterns
(high-value foreign crypto/gambling transactions), so the dashboard
and load test have realistic variety. No real customer, merchant,
or transaction data is used.
