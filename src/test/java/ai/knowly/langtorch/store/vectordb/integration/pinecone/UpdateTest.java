package ai.knowly.langtorch.store.vectordb.integration.pinecone;

import static com.google.common.truth.Truth.assertThat;

import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.PineconeServiceConfig;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.Vector;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.fetch.FetchRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.fetch.PineconeFetchResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.update.PineconeUpdateRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.update.PineconeUpdateResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.PineconeUpsertRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.PineconeUpsertResponse;
import ai.knowly.langtorch.utils.ApiKeyUtils;
import com.google.common.collect.ImmutableMap;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

@EnabledIf("ai.knowly.langtorch.TestingUtils#testWithHttpRequestEnabled")
class UpdateTest {
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

    PineconeUpsertRequest pineconeUpsertRequest =
        PineconeUpsertRequest.builder()
            .setVectors(
                Arrays.asList(
                    Vector.builder()
                        .setId("test2")
                        .setValues(Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
                        .setMetadata(ImmutableMap.of("key", "val"))
                        .build()))
            .setNamespace("testr2")
            .build();

    PineconeUpdateRequest pineconeUpdateRequest =
        PineconeUpdateRequest.builder()
            .setId("test2")
            .setValues(Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9))
            .setNamespace("testr2")
            .build();

    FetchRequest fetchRequest =
        FetchRequest.builder().setIds(Arrays.asList("test2")).setNamespace("testr2").build();

    // Act.
    PineconeUpsertResponse response = service.upsert(pineconeUpsertRequest);
    PineconeUpdateResponse pineconeUpdateResponse = service.update(pineconeUpdateRequest);
    PineconeFetchResponse pineconeFetchResponse = service.fetch(fetchRequest);
    // Assert.
    assertThat(response.getUpsertedCount()).isEqualTo(1);
    assertThat(pineconeFetchResponse.getVectors().get("test2").getValues())
        .isEqualTo(Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9));
  }
}
