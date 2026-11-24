package com.saisuhas.inference.service;

import com.saisuhas.inference.model.InferenceModel;
import com.saisuhas.inference.model.InferenceResult;
import com.saisuhas.inference.model.LogisticRegressionFraudModel;
import com.saisuhas.inference.model.RemoteModelClient;
import com.saisuhas.inference.model.RiskBand;
import com.saisuhas.inference.model.TransactionFeatures;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Scores transactions with the configured model and tracks latency.
 *
 * <ul>
 *   <li>Single predictions run on the calling virtual thread, behind
 *       a semaphore that caps in-flight inference
 *       ({@code inference.max-inflight}, default 256).</li>
 *   <li>Batch predictions fan out to a virtual-thread-per-task
 *       executor, so a 500-item batch scores its items concurrently
 *       instead of one at a time.</li>
 *   <li>When a remote model endpoint is configured, it is tried
 *       first; any failure falls back to the local logistic model and
 *       the response says which model actually scored it.</li>
 * </ul>
 */
@Service
public class InferenceService {

    private final LogisticRegressionFraudModel localModel = new LogisticRegressionFraudModel();
    private final RemoteModelClient remoteModel;
    private final Semaphore inFlight;
    private final int maxBatchSize;
    private final ExecutorService batchExecutor = Executors.newVirtualThreadPerTaskExecutor();
    private final LatencyStats latencyStats = new LatencyStats();
    private final long startedAtMs = System.currentTimeMillis();

    public InferenceService(
            @Value("${inference.max-inflight:256}") int maxInFlight,
            @Value("${inference.max-batch-size:1000}") int maxBatchSize,
            @Value("${inference.model.endpoint:}") String endpoint,
            @Value("${inference.model.api-key:}") String apiKey) {
        this.inFlight = new Semaphore(maxInFlight);
        this.maxBatchSize = maxBatchSize;
        this.remoteModel = (endpoint == null || endpoint.isBlank()) ? null : new RemoteModelClient(endpoint, apiKey);
    }

    public int maxBatchSize() {
        return maxBatchSize;
    }

    public LatencyStats latencyStats() {
        return latencyStats;
    }

    public long uptimeMs() {
        return System.currentTimeMillis() - startedAtMs;
    }

    /** Score one transaction. Returns 429-style rejection via exception when saturated. */
    public InferenceResult predict(TransactionFeatures features) {
        if (!inFlight.tryAcquire()) {
            latencyStats.recordRejected();
            throw new InferenceOverloadedException("Inference engine is saturated, retry shortly");
        }
        long start = System.nanoTime();
        try {
            InferenceResult result = score(features, start);
            latencyStats.recordSingle(result.inferenceMs());
            return result;
        } finally {
            inFlight.release();
        }
    }

    /** Score a batch in parallel on virtual threads. */
    public BatchResult predictBatch(List<TransactionFeatures> features) {
        long start = System.nanoTime();
        List<Future<InferenceResult>> futures = new ArrayList<>();
        for (TransactionFeatures f : features) {
            futures.add(batchExecutor.submit(() -> score(f, System.nanoTime())));
        }
        List<InferenceResult> results = new ArrayList<>();
        for (Future<InferenceResult> future : futures) {
            try {
                results.add(future.get(30, TimeUnit.SECONDS));
            } catch (Exception e) {
                throw new IllegalStateException("Batch inference failed", e);
            }
        }
        double totalMs = (System.nanoTime() - start) / 1_000_000.0;
        latencyStats.recordBatch(results.size(), totalMs);
        return new BatchResult(results, results.size(), Math.round(totalMs * 1000.0) / 1000.0);
    }

    private InferenceResult score(TransactionFeatures f, long startNanos) {
        InferenceModel model = localModel;
        String modelName = localModel.name();
        if (remoteModel != null) {
            try {
                double p = remoteModel.predictProbability(f);
                return toResult(f, p, remoteModel.name(), startNanos, List.of());
            } catch (Exception ignored) {
                // Fall through to the local model; scoring never blocks on the remote model.
                model = localModel;
                modelName = localModel.name();
            }
        }
        double p = model.predictProbability(f);
        return toResult(f, p, modelName, startNanos, localModel.contributions(f));
    }

    private InferenceResult toResult(TransactionFeatures f, double probability, String modelName,
                                     long startNanos, java.util.List<com.saisuhas.inference.model.FeatureContribution> contributions) {
        int score = (int) Math.round(probability * 100.0);
        RiskBand band = RiskBand.fromScore(score);
        double inferenceMs = (System.nanoTime() - startNanos) / 1_000_000.0;
        return new InferenceResult(f.id(), f.customerId(), score, band, band.classification(),
                modelName, contributions, Math.round(inferenceMs * 1000.0) / 1000.0);
    }

    public record BatchResult(List<InferenceResult> results, int count, double totalMs) {}

    public static class InferenceOverloadedException extends RuntimeException {
        public InferenceOverloadedException(String message) {
            super(message);
        }
    }
}
