package ai.knowly.langtorch.store.vectordb.integration.pinecone;

import static ai.knowly.langtorch.util.PineconeTestingUtils.PINECONE_TESTING_SERVICE;
import static com.google.common.truth.Truth.assertThat;

import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.Vector;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.delete.DeleteRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.delete.DeleteResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.QueryRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.QueryResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.UpsertRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.UpsertResponse;
import com.google.common.collect.ImmutableMap;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

@EnabledIf("ai.knowly.langtorch.util.TestingSettingUtils#enablePineconeVectorStoreLiveTrafficTest")
final class DeleteTest {
  @Test
  void test() {
    // Arrange.
    PineconeService service = PINECONE_TESTING_SERVICE;

    UpsertRequest upsertRequest =
        UpsertRequest.builder()
            .setVectors(
                Arrays.asList(
                    Vector.builder()
                        .setId("test2")
                        .setValues(Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
                        .setMetadata(ImmutableMap.of("key", "val"))
                        .build()))
            .setNamespace("namespace")
            .build();

    DeleteRequest deleteRequest =
        DeleteRequest.builder().setIds(Arrays.asList("test2")).setNamespace("namespace").build();

    QueryRequest queryRequest =
        QueryRequest.builder().setNamespace("namespace").setId("test2").setTopK(1).build();

    // Act.
    UpsertResponse upsertResponse = service.upsert(upsertRequest);
    DeleteResponse deleteResponse = service.delete(deleteRequest);
    QueryResponse queryResponse = service.query(queryRequest);
    // Assert.
    assertThat(upsertResponse.getUpsertedCount()).isEqualTo(1);
    assertThat(deleteResponse).isNotNull();
    assertThat(queryResponse.getMatches()).isEmpty();
  }
}
