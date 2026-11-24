package com.saisuhas.inference.seed;

import com.saisuhas.inference.model.TransactionFeatures;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Synthetic transactions for the dashboard samples and the load test.
 * Fixed seed, so the same transactions are generated on every run.
 * No real customer, merchant, or transaction data is used.
 */
public class SeedDataGenerator {

    private static final String[] MERCHANTS = {
            "Starbucks", "Amazon", "Walmart", "Uber", "Netflix",
            "Crypto Exchange X", "Lucky Slots", "Best Buy", "Delta Air Lines", "Shell"
    };
    private static final String[] CATEGORIES = {
            "GROCERIES", "RETAIL", "RETAIL", "TRAVEL", "ENTERTAINMENT",
            "CRYPTO", "GAMBLING", "ELECTRONICS", "TRAVEL", "FUEL"
    };
    private static final String[] COUNTRIES = {"US", "US", "US", "US", "GB", "CA", "NG", "IN"};

    /** Generate {@code count} synthetic transactions. */
    public static List<TransactionFeatures> generate(int count) {
        Random random = new Random(42);
        List<TransactionFeatures> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int m = random.nextInt(MERCHANTS.length);
            String country = COUNTRIES[random.nextInt(COUNTRIES.length)];
            double amount = Math.round((5 + random.nextDouble() * 2000) * 100.0) / 100.0;
            boolean fraudLike = random.nextDouble() < 0.08;
            if (fraudLike) {
                amount = Math.round((3000 + random.nextDouble() * 9000) * 100.0) / 100.0;
                m = 5 + random.nextInt(2); // crypto or gambling
                country = "NG";
            }
            out.add(new TransactionFeatures(
                    "TXN-SEED-" + (i + 1),
                    "CUST-" + String.format("%03d", random.nextInt(50) + 1),
                    amount,
                    "USD",
                    MERCHANTS[m],
                    CATEGORIES[m],
                    country,
                    "DEV-" + (1000 + random.nextInt(9000)),
                    random.nextInt(24),
                    random.nextInt(6),
                    250.0,
                    random.nextDouble() < 0.15,
                    !country.equals("US")));
        }
        return out;
    }

    /** A few hand-picked samples for the dashboard's "try it" buttons. */
    public static List<TransactionFeatures> samples() {
        return List.of(
                new TransactionFeatures("TXN-DEMO-FRAUD", "CUST-001", 12400.00, "USD",
                        "Crypto Exchange X", "CRYPTO", "NG", "DEV-UNKNOWN", 2, 0, 250.0, true, true),
                new TransactionFeatures("TXN-DEMO-NORMAL", "CUST-002", 42.50, "USD",
                        "Starbucks", "GROCERIES", "US", "DEV-1042", 14, 1, 250.0, false, false),
                new TransactionFeatures("TXN-DEMO-VELOCITY", "CUST-003", 899.00, "USD",
                        "Best Buy", "ELECTRONICS", "US", "DEV-7710", 22, 7, 180.0, true, false));
    }
}
