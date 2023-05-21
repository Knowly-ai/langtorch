package ai.knowly.langtorch.store.vectordb.integration.pinecone;

import static com.google.common.truth.Truth.assertThat;

import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.PineconeServiceConfig;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.Vector;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.delete.PineconeDeleteRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.delete.PineconeDeleteResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.PineconeQueryRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.PineconeQueryResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.PineconeUpsertRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.PineconeUpsertResponse;
import ai.knowly.langtorch.utils.ApiKeyUtils;
import com.google.common.collect.ImmutableMap;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

@EnabledIf("ai.knowly.langtorch.TestingUtils#testWithHttpRequestEnabled")
final class DeleteTest {
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
            .setNamespace("namespace")
            .build();

    PineconeDeleteRequest pineconeDeleteRequest =
        PineconeDeleteRequest.builder().setIds(Arrays.asList("test2")).setNamespace("namespace").build();

    PineconeQueryRequest pineconeQueryRequest =
        PineconeQueryRequest.builder().setNamespace("namespace").setId("test2").setTopK(1).build();

    // Act.
    PineconeUpsertResponse pineconeUpsertResponse = service.upsert(pineconeUpsertRequest);
    PineconeDeleteResponse pineconeDeleteResponse = service.delete(pineconeDeleteRequest);
    PineconeQueryResponse pineconeQueryResponse = service.query(pineconeQueryRequest);
    // Assert.
    assertThat(pineconeUpsertResponse.getUpsertedCount()).isEqualTo(1);
    assertThat(pineconeDeleteResponse).isNotNull();
    assertThat(pineconeQueryResponse.getMatches()).isEmpty();
  }
}
