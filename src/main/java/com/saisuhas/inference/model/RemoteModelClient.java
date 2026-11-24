package com.saisuhas.inference.model;

import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestClient;

import java.util.Map;

/**
 * Optional remote model client (ONNX server or Azure ML scoring
 * endpoint). It is only used when {@code inference.model.endpoint} is
 * set; otherwise the local model scores everything. Any failure here
 * throws, and the service falls back to the local model — scoring
 * never blocks on a remote model being down.
 *
 * The endpoint is expected to accept the transaction as JSON and
 * return JSON with a numeric {@code probability} field in [0, 1].
 * The API key, when set, is sent only as an Authorization header to
 * the configured endpoint and is never logged.
 */
public class RemoteModelClient implements InferenceModel {

    private final String endpoint;
    private final String apiKey;
    private final RestClient restClient;

    public RemoteModelClient(String endpoint, String apiKey) {
        this.endpoint = endpoint;
        this.apiKey = apiKey;
        this.restClient = RestClient.create();
    }

    @Override
    public String name() {
        return "remote-model";
    }

    @Override
    @SuppressWarnings("unchecked")
    public double predictProbability(TransactionFeatures f) {
        RestClient.RequestBodySpec spec = restClient.post()
                .uri(endpoint)
                .contentType(MediaType.APPLICATION_JSON);
        if (apiKey != null && !apiKey.isBlank()) {
            spec = spec.header(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey);
        }
        Map<String, Object> body = spec
                .body(Map.of(
                        "amount", f.amount(),
                        "merchantCategory", f.merchantCategory(),
                        "country", f.country(),
                        "hourOfDay", f.hourOfDay(),
                        "transactionsLastHour", f.transactionsLastHour(),
                        "customerAvgAmount", f.customerAvgAmount(),
                        "newDevice", f.newDevice(),
                        "foreignCountry", f.foreignCountry()))
                .retrieve()
                .body(Map.class);
        if (body == null || body.get("probability") == null) {
            throw new IllegalStateException("Remote model returned no probability");
        }
        double p = ((Number) body.get("probability")).doubleValue();
        if (p < 0.0 || p > 1.0) {
            throw new IllegalStateException("Remote model probability out of range: " + p);
        }
        return p;
    }
}
