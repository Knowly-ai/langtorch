package ai.knowly.langtorch.store.vectordb.integration.pinecone;

import static com.google.common.truth.Truth.assertThat;
import static java.util.Collections.emptyList;
import static org.mockito.Mockito.when;

import ai.knowly.langtorch.processor.EmbeddingProcessor;
import ai.knowly.langtorch.schema.embeddings.Embedding;
import ai.knowly.langtorch.schema.embeddings.EmbeddingOutput;
import ai.knowly.langtorch.schema.embeddings.EmbeddingType;
import ai.knowly.langtorch.schema.io.DomainDocument;
import ai.knowly.langtorch.schema.io.Metadata;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.PineconeVectorStoreSpec;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.SparseValues;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.delete.DeleteResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.Match;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.QueryResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.update.UpdateResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.UpsertResponse;
import ai.knowly.langtorch.store.vectordb.integration.schema.StringSimilaritySearchQuery;
import ai.knowly.langtorch.store.vectordb.integration.schema.VectorSimilaritySearchQuery;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentMatchers;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;


@ExtendWith(MockitoExtension.class)
final class PineconeVectorStoreTest {

  private static final int DOCUMENT_COUNT = 3;

  private String textKey;
  private PineconeVectorStore pineconeVectorStore;
  private PineconeService pineconeService;
  private EmbeddingProcessor embeddingProcessor;

  @BeforeEach
  void setUp() {
    textKey = "textKey";
    pineconeService = Mockito.mock(PineconeService.class);
    embeddingProcessor = Mockito.mock(EmbeddingProcessor.class);

    pineconeVectorStore =
        new PineconeVectorStore(
            embeddingProcessor,
            PineconeVectorStoreSpec.builder().setTextKey(textKey).build(),
            pineconeService);
  }

  @Test
  void testAddDocuments() {
    EmbeddingOutput embeddingOutput = EmbeddingOutput.of(EmbeddingType.OPEN_AI, getEmbeddings());
    Mockito.when(embeddingProcessor.run(ArgumentMatchers.any())).thenReturn(embeddingOutput);
    UpsertResponse upsertResponse = new UpsertResponse(DOCUMENT_COUNT);
    Mockito.when(pineconeService.upsert(ArgumentMatchers.any())).thenReturn(upsertResponse);

    // Act.
    boolean isSuccessful = pineconeVectorStore.addDocuments(getDocuments());
    // Assert.
    assertThat(isSuccessful).isEqualTo(true);
  }

  @Test
  void testSimilaritySearchStringWithScore() {
    double score = 0.5;
    String content = "Content";

    List<Match> matches = new ArrayList<>();
    Map<String, String> map = new HashMap<>();
    map.put(textKey, content);
    Match match =
            new Match(
                    UUID.randomUUID().toString(), score, emptyList(), SparseValues.builder().build(), map);
    matches.add(match);

    QueryResponse queryResponse = new QueryResponse(matches, "");
    Mockito.when(pineconeService.query(ArgumentMatchers.any())).thenReturn(queryResponse);
    List<Embedding> embeddings = ImmutableList.of(Embedding.of(List.of()));
    EmbeddingOutput embeddingOutput = EmbeddingOutput.of(EmbeddingType.OPEN_AI, embeddings);
    when(embeddingProcessor.run(ArgumentMatchers.any())).thenReturn(embeddingOutput);

    // Act.
    List<DomainDocument> result =
            pineconeVectorStore.similaritySearch(
                    StringSimilaritySearchQuery.builder().setQuery("").setTopK(0L).build());
    // Assert.
    assertThat(result).isNotEmpty();
    assertThat(result.get(0).getPageContent()).isEqualTo(content);
    assertThat(result.get(0).getSimilarityScore().orElse(-1.0)).isEqualTo(score);
  }

  @Test
  void testSimilaritySearchVectorWithScore() {
    double score = 0.5;
    String content = "Content";

    List<Match> matches = new ArrayList<>();
    Map<String, String> map = new HashMap<>();
    map.put(textKey, content);
    Match match =
        new Match(
            UUID.randomUUID().toString(), score, emptyList(), SparseValues.builder().build(), map);
    matches.add(match);

    QueryResponse queryResponse = new QueryResponse(matches, "");
    Mockito.when(pineconeService.query(ArgumentMatchers.any())).thenReturn(queryResponse);

    // Act.
    List<DomainDocument> result =
        pineconeVectorStore.similaritySearch(
            VectorSimilaritySearchQuery.builder().setQuery(emptyList()).setTopK(0L).build());
    // Assert.
    assertThat(result).isNotEmpty();
    assertThat(result.get(0).getPageContent()).isEqualTo(content);
    assertThat(result.get(0).getSimilarityScore().orElse(-1.0)).isEqualTo(score);
  }

  @Test
  void testSimilaritySearchVectorWithScoreEmpty() {
    QueryResponse queryResponse = new QueryResponse(null, null);
    Mockito.when(pineconeService.query(ArgumentMatchers.any())).thenReturn(queryResponse);
    // Act.
    List<DomainDocument> result =
        pineconeVectorStore.similaritySearch(
            VectorSimilaritySearchQuery.builder().setQuery(emptyList()).setTopK(0L).build());
    // Assert.
    assertThat(result).isEmpty();
  }

  @Test
  void testUpdateDocuments() {
    EmbeddingOutput embeddingOutput = EmbeddingOutput.of(EmbeddingType.OPEN_AI, getEmbeddings());
    Mockito.when(embeddingProcessor.run(ArgumentMatchers.any())).thenReturn(embeddingOutput);
    ListenableFuture<UpdateResponse> future = Futures.immediateFuture(new UpdateResponse());
    Mockito.when(pineconeService.updateAsync(ArgumentMatchers.any())).thenReturn(future);

    // Act.
    boolean isSuccess = pineconeVectorStore.updateDocuments(getDocuments());
    // Assert.
    assertThat(isSuccess).isTrue();
  }

  @Test
  void testDeleteDocuments() {
    Mockito.when(pineconeService.delete(ArgumentMatchers.any())).thenReturn(new DeleteResponse());

    // Act.
    boolean isSuccess = pineconeVectorStore.deleteDocuments(getDocuments());
    // Assert.
    assertThat(isSuccess).isTrue();
  }

  private List<DomainDocument> getDocuments() {
    ArrayList<DomainDocument> documents = new ArrayList<>();
    for (int i = 0; i < DOCUMENT_COUNT; i++) {
      DomainDocument document =
          DomainDocument.builder()
              .setId(UUID.randomUUID().toString())
              .setPageContent("content" + i)
              .setMetadata(Metadata.builder().setValue(ImmutableMap.of("key", "val")).build())
              .build();
      documents.add(document);
    }
    return documents;
  }

  private List<Embedding> getEmbeddings() {
    ArrayList<Embedding> embeddings = new ArrayList<>();
    embeddings.add(Embedding.of(Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)));
    return embeddings;
  }
}
