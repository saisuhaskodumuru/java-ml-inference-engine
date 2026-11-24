package com.saisuhas.inference.controller;

import com.saisuhas.inference.model.InferenceResult;
import com.saisuhas.inference.model.TransactionFeatures;
import com.saisuhas.inference.seed.SeedDataGenerator;
import com.saisuhas.inference.service.InferenceService;
import com.saisuhas.inference.service.LatencyStats;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

/** REST API for the inference engine. No database, no persistence. */
@RestController
@RequestMapping("/api/v1")
public class InferenceController {

    private final InferenceService service;

    public InferenceController(InferenceService service) {
        this.service = service;
    }

    @PostMapping("/inference/predict")
    public InferenceResult predict(@Valid @RequestBody TransactionFeatures features) {
        return service.predict(features);
    }

    @PostMapping("/inference/batch")
    public InferenceService.BatchResult predictBatch(@Valid @RequestBody List<@Valid TransactionFeatures> features) {
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("Batch must contain at least one transaction");
        }
        if (features.size() > service.maxBatchSize()) {
            throw new IllegalArgumentException("Batch too large, max " + service.maxBatchSize());
        }
        return service.predictBatch(features);
    }

    @GetMapping("/inference/stats")
    public Map<String, Object> stats() {
        LatencyStats.Snapshot s = service.latencyStats().snapshot(service.uptimeMs());
        Map<String, Object> out = new java.util.LinkedHashMap<>();
        out.put("totalRequests", s.totalRequests());
        out.put("singlePredictions", s.singlePredictions());
        out.put("batchRequests", s.batchRequests());
        out.put("batchItems", s.batchItems());
        out.put("rejected", s.rejected());
        out.put("avgLatencyMs", s.avgLatencyMs());
        out.put("p50LatencyMs", s.p50LatencyMs());
        out.put("p95LatencyMs", s.p95LatencyMs());
        out.put("p99LatencyMs", s.p99LatencyMs());
        out.put("samples", s.samples());
        out.put("uptimeMs", s.uptimeMs());
        out.put("model", "local-logistic-v1");
        out.put("virtualThreads", true);
        return out;
    }

    @GetMapping("/inference/samples")
    public List<TransactionFeatures> samples() {
        return SeedDataGenerator.samples();
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "UP", "model", "local-logistic-v1");
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, String>> badRequest(IllegalArgumentException e) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of("error", e.getMessage()));
    }

    @ExceptionHandler(InferenceService.InferenceOverloadedException.class)
    public ResponseEntity<Map<String, String>> overloaded(InferenceService.InferenceOverloadedException e) {
        return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS).body(Map.of("error", e.getMessage()));
    }
}
