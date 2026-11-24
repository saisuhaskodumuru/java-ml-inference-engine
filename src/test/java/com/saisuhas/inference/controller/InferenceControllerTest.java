package com.saisuhas.inference.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class InferenceControllerTest {

    @Autowired
    private MockMvc mockMvc;

    private static final String FRAUD_TXN = """
            {"id":"TXN-1","customerId":"CUST-001","amount":12400.00,"currency":"USD",
             "merchant":"Crypto Exchange X","merchantCategory":"CRYPTO","country":"NG",
             "deviceId":"DEV-UNKNOWN","hourOfDay":2,"transactionsLastHour":0,
             "customerAvgAmount":250.0,"newDevice":true,"foreignCountry":true}""";

    private static final String NORMAL_TXN = """
            {"id":"TXN-2","customerId":"CUST-002","amount":42.50,"currency":"USD",
             "merchant":"Starbucks","merchantCategory":"GROCERIES","country":"US",
             "deviceId":"DEV-1042","hourOfDay":14,"transactionsLastHour":1,
             "customerAvgAmount":250.0,"newDevice":false,"foreignCountry":false}""";

    @Test
    void predictFraudTransactionIsCritical() throws Exception {
        mockMvc.perform(post("/api/v1/inference/predict")
                        .contentType(MediaType.APPLICATION_JSON).content(FRAUD_TXN))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.score", greaterThanOrEqualTo(75)))
                .andExpect(jsonPath("$.riskBand").value("CRITICAL"))
                .andExpect(jsonPath("$.classification").value("FRAUD_SUSPECT"))
                .andExpect(jsonPath("$.model").value("local-logistic-v1"))
                .andExpect(jsonPath("$.contributions", hasSize(8)));
    }

    @Test
    void predictNormalTransactionIsLow() throws Exception {
        mockMvc.perform(post("/api/v1/inference/predict")
                        .contentType(MediaType.APPLICATION_JSON).content(NORMAL_TXN))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskBand").value("LOW"))
                .andExpect(jsonPath("$.classification").value("LEGITIMATE"));
    }

    @Test
    void invalidAmountReturns400() throws Exception {
        String bad = FRAUD_TXN.replace("12400.00", "-5.00");
        mockMvc.perform(post("/api/v1/inference/predict")
                        .contentType(MediaType.APPLICATION_JSON).content(bad))
                .andExpect(status().isBadRequest());
    }

    @Test
    void batchScoresAllItems() throws Exception {
        mockMvc.perform(post("/api/v1/inference/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("[" + FRAUD_TXN + "," + NORMAL_TXN + "]"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.count").value(2))
                .andExpect(jsonPath("$.results", hasSize(2)))
                .andExpect(jsonPath("$.results[0].riskBand").value("CRITICAL"))
                .andExpect(jsonPath("$.results[1].riskBand").value("LOW"));
    }

    @Test
    void emptyBatchReturns400() throws Exception {
        mockMvc.perform(post("/api/v1/inference/batch")
                        .contentType(MediaType.APPLICATION_JSON).content("[]"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void statsAndHealthRespond() throws Exception {
        mockMvc.perform(get("/api/v1/inference/stats"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("local-logistic-v1"))
                .andExpect(jsonPath("$.virtualThreads").value(true));
        mockMvc.perform(get("/api/v1/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("UP"));
    }
}
