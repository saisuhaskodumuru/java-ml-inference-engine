package com.saisuhas.inference.model;

import java.util.List;

/** The result of scoring one transaction. */
public record InferenceResult(
        String transactionId,
        String customerId,
        int score,
        RiskBand riskBand,
        String classification,
        String model,
        List<FeatureContribution> contributions,
        double inferenceMs
) {}
