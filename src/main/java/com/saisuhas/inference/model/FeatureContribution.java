package com.saisuhas.inference.model;

/** One feature's contribution to the model's logit, for explainability. */
public record FeatureContribution(String feature, double value, double weight, double contribution) {}
