package ai.knowly.langtorch.store.vectordb.integration.pinecone;

import static com.google.common.truth.Truth.assertThat;

import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.PineconeServiceConfig;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.Vector;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.UpsertRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.UpsertResponse;
import ai.knowly.langtorch.utils.ApiKeyUtils;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

@EnabledIf("ai.knowly.langtorch.TestingUtils#testWithHttpRequestEnabled")
class UpsertTest {
  @Test
  void upsertTest() {
    // Arrange.
    String token = ApiKeyUtils.getPineconeKeyFromEnv();
    PineconeService service =
        new PineconeService(
            PineconeServiceConfig.builder()
                .setApiKey(token)
                .setEndpoint("https://test1-c4943a1.svc.us-west4-gcp-free.pinecone.io")
                .build());

    UpsertRequest upsertRequest =
        UpsertRequest.builder()
            .setVectors(
                List.of(
                    Vector.builder()
                        .setId("test2")
                        .setValues(List.of(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
                        .setMetadata(Map.of("key", "val"))
                        .build()))
            .setNamespace("testr2")
            .build();

    // Act.
    UpsertResponse response = service.upsert(upsertRequest);
    // Assert.
    assertThat(response.getUpsertedCount()).isEqualTo(1);
  }
}
