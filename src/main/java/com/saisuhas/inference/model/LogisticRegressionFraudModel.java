package com.saisuhas.inference.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A real, deterministic logistic-regression-style model that runs
 * in-process. It extracts eight features from a transaction, applies
 * fixed weights, and passes the weighted sum through a sigmoid to get
 * a fraud probability. The weights are hand-set for this demo (not
 * fitted on real data — there is no real data here), and every
 * contribution is exposed so a score is explainable.
 *
 * Features and weights:
 * <ul>
 *   <li>logAmount — log10(amount + 1) / 5, weight 1.4</li>
 *   <li>amountRatio — min(amount / customerAvgAmount, 10) / 10, weight 1.2</li>
 *   <li>velocity — min(transactionsLastHour, 10) / 10, weight 1.0</li>
 *   <li>night — 1 when hourOfDay is 00:00-04:59 UTC, weight 0.5</li>
 *   <li>foreignCountry — 1 when the transaction country is foreign to the customer, weight 0.7</li>
 *   <li>highRiskCountry — 1 for a small list of high-risk countries, weight 0.6</li>
 *   <li>newDevice — 1 when the device has not been seen for this customer, weight 0.8</li>
 *   <li>merchantRisk — 1.0 for CRYPTO/GAMBLING, 0.4 for ELECTRONICS/TRAVEL, else 0, weight 1.1</li>
 * </ul>
 * Bias is -3.2, so an ordinary transaction scores low and several
 * risk signals have to combine before a score turns high.
 */
public class LogisticRegressionFraudModel implements InferenceModel {

    public static final String MODEL_NAME = "local-logistic-v1";

    private static final double BIAS = -3.2;
    private static final Set<String> HIGH_RISK_COUNTRIES = Set.of("NG", "RU", "KP", "IR", "MM");
    private static final Set<String> HIGH_RISK_MERCHANTS = Set.of("CRYPTO", "GAMBLING");
    private static final Set<String> MEDIUM_RISK_MERCHANTS = Set.of("ELECTRONICS", "TRAVEL");

    @Override
    public String name() {
        return MODEL_NAME;
    }

    @Override
    public double predictProbability(TransactionFeatures f) {
        return sigmoid(logit(f));
    }

    /** The raw weighted sum before the sigmoid, exposed for tests. */
    public double logit(TransactionFeatures f) {
        double total = BIAS;
        for (FeatureContribution c : contributions(f)) {
            total += c.contribution();
        }
        return total;
    }

    /** Every feature's value, weight, and contribution to the logit. */
    public List<FeatureContribution> contributions(TransactionFeatures f) {
        List<FeatureContribution> out = new ArrayList<>();
        out.add(new FeatureContribution("logAmount",
                logAmount(f.amount()), 1.4, 1.4 * logAmount(f.amount())));
        double ratio = Math.min(f.amount() / f.customerAvgAmount(), 10.0) / 10.0;
        out.add(new FeatureContribution("amountRatio", ratio, 1.2, 1.2 * ratio));
        double velocity = Math.min(f.transactionsLastHour(), 10) / 10.0;
        out.add(new FeatureContribution("velocity", velocity, 1.0, 1.0 * velocity));
        double night = (f.hourOfDay() >= 0 && f.hourOfDay() <= 4) ? 1.0 : 0.0;
        out.add(new FeatureContribution("night", night, 0.5, 0.5 * night));
        double foreign = f.foreignCountry() ? 1.0 : 0.0;
        out.add(new FeatureContribution("foreignCountry", foreign, 0.7, 0.7 * foreign));
        double highRiskCountry = HIGH_RISK_COUNTRIES.contains(f.country().toUpperCase()) ? 1.0 : 0.0;
        out.add(new FeatureContribution("highRiskCountry", highRiskCountry, 0.6, 0.6 * highRiskCountry));
        double newDevice = f.newDevice() ? 1.0 : 0.0;
        out.add(new FeatureContribution("newDevice", newDevice, 0.8, 0.8 * newDevice));
        double merchantRisk = merchantRisk(f.merchantCategory());
        out.add(new FeatureContribution("merchantRisk", merchantRisk, 1.1, 1.1 * merchantRisk));
        return out;
    }

    private double logAmount(double amount) {
        return Math.log10(amount + 1.0) / 5.0;
    }

    private double merchantRisk(String category) {
        String c = category.toUpperCase();
        if (HIGH_RISK_MERCHANTS.contains(c)) return 1.0;
        if (MEDIUM_RISK_MERCHANTS.contains(c)) return 0.4;
        return 0.0;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
