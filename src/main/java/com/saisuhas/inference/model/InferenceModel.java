package com.saisuhas.inference.model;

/**
 * Pluggable ML model. Implementations take transaction features and
 * return a fraud probability in [0, 1]. The local model implements this
 * in-process; a remote model (ONNX server, Azure ML endpoint) would
 * implement it over HTTP and throw on failure so the service can fall
 * back to the local model.
 */
public interface InferenceModel {
    /** Unique model identifier reported in responses, e.g. "local-logistic-v1". */
    String name();

    /** Fraud probability in [0, 1] for one transaction. */
    double predictProbability(TransactionFeatures features);
}
