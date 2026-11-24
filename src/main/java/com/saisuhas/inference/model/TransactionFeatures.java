package com.saisuhas.inference.model;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;

/**
 * One transaction to score. All fields are synthetic demo data.
 * {@code hourOfDay} is UTC. {@code customerAvgAmount} is the customer's
 * own typical transaction amount, used for the amount-ratio feature.
 */
public record TransactionFeatures(
        @NotBlank String id,
        @NotBlank String customerId,
        @NotNull @DecimalMin(value = "0.0", inclusive = false) Double amount,
        @NotBlank String currency,
        @NotBlank String merchant,
        @NotBlank String merchantCategory,
        @NotBlank String country,
        @NotBlank String deviceId,
        @NotNull @Min(0) @Max(23) Integer hourOfDay,
        @NotNull @PositiveOrZero Integer transactionsLastHour,
        @NotNull @DecimalMin(value = "0.0", inclusive = false) Double customerAvgAmount,
        @NotNull Boolean newDevice,
        @NotNull Boolean foreignCountry
) {}
