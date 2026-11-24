package com.saisuhas.inference.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LogisticRegressionFraudModelTest {

    private final LogisticRegressionFraudModel model = new LogisticRegressionFraudModel();

    private TransactionFeatures normal() {
        return new TransactionFeatures("TXN-1", "CUST-001", 42.50, "USD", "Starbucks",
                "GROCERIES", "US", "DEV-1042", 14, 1, 250.0, false, false);
    }

    private TransactionFeatures fraudLike() {
        return new TransactionFeatures("TXN-2", "CUST-002", 12400.00, "USD", "Crypto Exchange X",
                "CRYPTO", "NG", "DEV-UNKNOWN", 2, 0, 250.0, true, true);
    }

    @Test
    void fraudLikeScoresFarHigherThanNormal() {
        double normal = model.predictProbability(normal());
        double fraud = model.predictProbability(fraudLike());
        assertTrue(normal < 0.2, "normal transaction should be low probability, was " + normal);
        assertTrue(fraud > 0.9, "fraud-like transaction should be high probability, was " + fraud);
        assertTrue(fraud > normal);
    }

    @Test
    void scoreIsDeterministic() {
        assertEquals(model.predictProbability(fraudLike()), model.predictProbability(fraudLike()));
    }

    @Test
    void velocityRaisesTheScore() {
        TransactionFeatures base = new TransactionFeatures("TXN-3", "CUST-003", 100.00, "USD",
                "Walmart", "RETAIL", "US", "DEV-1", 14, 0, 250.0, false, false);
        TransactionFeatures burst = new TransactionFeatures("TXN-4", "CUST-003", 100.00, "USD",
                "Walmart", "RETAIL", "US", "DEV-1", 14, 8, 250.0, false, false);
        assertTrue(model.predictProbability(burst) > model.predictProbability(base));
    }

    @Test
    void contributionsSumToLogitMinusBias() {
        TransactionFeatures f = fraudLike();
        double sum = model.contributions(f).stream()
                .mapToDouble(FeatureContribution::contribution).sum();
        assertEquals(model.logit(f), sum - 3.2, 1e-9);
    }

    @Test
    void riskBandsMapCorrectly() {
        assertEquals(RiskBand.LOW, RiskBand.fromScore(15));
        assertEquals(RiskBand.MEDIUM, RiskBand.fromScore(40));
        assertEquals(RiskBand.HIGH, RiskBand.fromScore(65));
        assertEquals(RiskBand.CRITICAL, RiskBand.fromScore(90));
        assertEquals("FRAUD_SUSPECT", RiskBand.CRITICAL.classification());
        assertEquals("LEGITIMATE", RiskBand.LOW.classification());
    }
}
