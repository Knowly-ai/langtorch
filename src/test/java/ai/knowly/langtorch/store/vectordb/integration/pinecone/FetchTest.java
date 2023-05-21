package ai.knowly.langtorch.store.vectordb.integration.pinecone;

import static com.google.common.truth.Truth.assertThat;

import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.PineconeServiceConfig;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.Vector;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.fetch.FetchRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.fetch.PineconeFetchResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.PineconeUpsertRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.PineconeUpsertResponse;
import ai.knowly.langtorch.utils.ApiKeyUtils;
import com.google.common.collect.ImmutableMap;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

@EnabledIf("ai.knowly.langtorch.TestingUtils#testWithHttpRequestEnabled")
class FetchTest {
  @Test
  void test() {
    // Arrange.
    String token = ApiKeyUtils.getPineconeKeyFromEnv();
    PineconeService service =
        new PineconeService(
            PineconeServiceConfig.builder()
                .setApiKey(token)
                .setEndpoint("https://test1-c4943a1.svc.us-west4-gcp-free.pinecone.io")
                .build());

    Vector vector =
        Vector.builder()
            .setId("test2")
            .setValues(Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
            .setMetadata(ImmutableMap.of("key", "val"))
            .build();

    PineconeUpsertRequest pineconeUpsertRequest =
        PineconeUpsertRequest.builder().setVectors(Arrays.asList(vector)).setNamespace("namespace").build();

    FetchRequest fetchRequest =
        FetchRequest.builder().setIds(Arrays.asList("test2")).setNamespace("namespace").build();

    // Act.
    PineconeUpsertResponse response = service.upsert(pineconeUpsertRequest);
    PineconeFetchResponse pineconeFetchResponse = service.fetch(fetchRequest);

    // Assert.
    assertThat(response.getUpsertedCount()).isEqualTo(1);
    assertThat(pineconeFetchResponse.getVectors().get("test2")).isEqualTo(vector);
  }
}
