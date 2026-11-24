package com.saisuhas.inference.service;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-process latency tracking. Every scored request records its
 * latency in milliseconds; the last 10,000 samples are kept so p50
 * and p95 are real measured percentiles, not estimates. All state is
 * guarded for concurrent access from virtual threads.
 */
public class LatencyStats {

    private static final int MAX_SAMPLES = 10_000;

    private final AtomicLong totalRequests = new AtomicLong();
    private final AtomicLong singlePredictions = new AtomicLong();
    private final AtomicLong batchRequests = new AtomicLong();
    private final AtomicLong batchItems = new AtomicLong();
    private final AtomicLong rejected = new AtomicLong();
    private final ArrayDeque<Double> samples = new ArrayDeque<>();

    public void recordSingle(double latencyMs) {
        totalRequests.incrementAndGet();
        singlePredictions.incrementAndGet();
        addSample(latencyMs);
    }

    public void recordBatch(int items, double totalLatencyMs) {
        totalRequests.addAndGet(items);
        batchRequests.incrementAndGet();
        batchItems.addAndGet(items);
        addSample(totalLatencyMs / Math.max(items, 1));
    }

    public void recordRejected() {
        rejected.incrementAndGet();
    }

    private synchronized void addSample(double latencyMs) {
        if (samples.size() >= MAX_SAMPLES) {
            samples.pollFirst();
        }
        samples.addLast(latencyMs);
    }

    public synchronized Snapshot snapshot(long uptimeMs) {
        List<Double> sorted = new ArrayList<>(samples);
        Collections.sort(sorted);
        double avg = sorted.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        return new Snapshot(
                totalRequests.get(),
                singlePredictions.get(),
                batchRequests.get(),
                batchItems.get(),
                rejected.get(),
                round(avg),
                round(percentile(sorted, 50)),
                round(percentile(sorted, 95)),
                round(percentile(sorted, 99)),
                sorted.size(),
                uptimeMs);
    }

    private static double percentile(List<Double> sorted, int p) {
        if (sorted.isEmpty()) return 0.0;
        int index = (int) Math.ceil(p / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private static double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }

    public record Snapshot(
            long totalRequests,
            long singlePredictions,
            long batchRequests,
            long batchItems,
            long rejected,
            double avgLatencyMs,
            double p50LatencyMs,
            double p95LatencyMs,
            double p99LatencyMs,
            int samples,
            long uptimeMs) {}
}
