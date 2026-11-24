package com.saisuhas.inference.model;

/** Risk band for a 0-100 inference score. */
public enum RiskBand {
    LOW, MEDIUM, HIGH, CRITICAL;

    public static RiskBand fromScore(int score) {
        if (score >= 75) return CRITICAL;
        if (score >= 55) return HIGH;
        if (score >= 30) return MEDIUM;
        return LOW;
    }

    /** Human-readable classification for the band. */
    public String classification() {
        return switch (this) {
            case LOW -> "LEGITIMATE";
            case MEDIUM -> "REVIEW";
            case HIGH, CRITICAL -> "FRAUD_SUSPECT";
        };
    }
}
