package ai.knowly.langtorch.store.vectordb.integration.pgvector;

import ai.knowly.langtorch.processor.EmbeddingProcessor;
import ai.knowly.langtorch.schema.embeddings.Embedding;
import ai.knowly.langtorch.schema.embeddings.EmbeddingOutput;
import ai.knowly.langtorch.schema.embeddings.EmbeddingType;
import ai.knowly.langtorch.schema.io.DomainDocument;
import ai.knowly.langtorch.schema.io.Metadata;
import ai.knowly.langtorch.store.vectordb.PGVectorStore;
import ai.knowly.langtorch.store.vectordb.integration.pgvector.schema.PGVectorStoreSpec;
import ai.knowly.langtorch.store.vectordb.integration.pgvector.schema.distance.DistanceStrategies;
import ai.knowly.langtorch.store.vectordb.integration.schema.StringSimilaritySearchQuery;
import ai.knowly.langtorch.store.vectordb.integration.schema.VectorSimilaritySearchQuery;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.pgvector.PGvector;
import kotlin.Triple;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentMatchers;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
final class PGVectorStoreTest {

  private static final int DOCUMENT_COUNT = 3;
  private static final float TOP_VECTOR_VALUE = 1;

  private EmbeddingProcessor embeddingProcessor;
  private PGVectorService pgVectorService;
  private PGVectorStoreSpec pgVectorStoreSpec;
  private PGVectorStore pgVectorStore;
  private String textKey;

  private PreparedStatement preparedStatement;

  @BeforeEach
  void setUp() throws SQLException {
    textKey = "text_key";
    embeddingProcessor = Mockito.mock(EmbeddingProcessor.class);
    pgVectorService = Mockito.mock(PGVectorService.class);
    preparedStatement = Mockito.mock(PreparedStatement.class);

    pgVectorStoreSpec =
        PGVectorStoreSpec.builder()
            .setTextKey(textKey)
            .setDatabaseName("test")
            .setVectorDimensions(3)
            .build();

    pgVectorStore =
        new PGVectorStore(
            embeddingProcessor, pgVectorStoreSpec, pgVectorService, DistanceStrategies.cosine());
  }

  @Test
  void testAddDocuments() throws SQLException {
    EmbeddingOutput embeddingOutput = EmbeddingOutput.of(EmbeddingType.OPEN_AI, getEmbeddings());
    when(embeddingProcessor.run(ArgumentMatchers.any())).thenReturn(embeddingOutput);
    when(pgVectorService.prepareStatement(ArgumentMatchers.any())).thenReturn(preparedStatement);
    when(preparedStatement.executeUpdate()).thenReturn(DOCUMENT_COUNT, DOCUMENT_COUNT);

    // Act.
    boolean isSuccessful = pgVectorStore.addDocuments(getDocuments());
    // Assert.
    assertThat(isSuccessful).isEqualTo(true);
  }


  @Test
  void testSimilaritySearchStringWithScoreEuclidean() throws SQLException {
    pgVectorStore =
            new PGVectorStore(
                    embeddingProcessor, pgVectorStoreSpec, pgVectorService, DistanceStrategies.euclidean());

    Triple<String, String, StringSimilaritySearchQuery> queryData = prepareSimilaritySearchStringQuery();
    StringSimilaritySearchQuery query = queryData.getThird();
    String firstPageContent = queryData.getFirst();
    String secondPageContent = queryData.getSecond();

    // Act.
    List<DomainDocument> documentsWithScores = pgVectorStore.similaritySearch(query);
    // Assert.
    double firstDocumentScore = documentsWithScores.get(0).getSimilarityScore().orElse(-1.0);
    double secondDocumentScore = documentsWithScores.get(1).getSimilarityScore().orElse(-1.0);
    String firstDocumentPageContent = documentsWithScores.get(0).getPageContent();
    String secondDocumentPageContent = documentsWithScores.get(1).getPageContent();
    assertThat(documentsWithScores.size()).isEqualTo(3);
    assertThat(firstDocumentScore).isEqualTo(0);
    assertThat(firstDocumentScore).isLessThan(secondDocumentScore);
    assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
    assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
  }

  @Test
  void testSimilaritySearchStringWithScoreInnerProduct() throws SQLException {
    pgVectorStore =
            new PGVectorStore(
                    embeddingProcessor,
                    pgVectorStoreSpec,
                    pgVectorService,
                    DistanceStrategies.innerProduct());

    Triple<String, String, StringSimilaritySearchQuery> queryData = prepareSimilaritySearchStringQuery();
    StringSimilaritySearchQuery query = queryData.getThird();
    String firstPageContent = queryData.getFirst();
    String secondPageContent = queryData.getSecond();

    // Act.
    List<DomainDocument> documentsWithScores = pgVectorStore.similaritySearch(query);
    // Assert.
    double firstDocumentScore = documentsWithScores.get(0).getSimilarityScore().orElse(-1.0);
    double secondDocumentScore = documentsWithScores.get(1).getSimilarityScore().orElse(-1.0);
    String firstDocumentPageContent = documentsWithScores.get(0).getPageContent();
    String secondDocumentPageContent = documentsWithScores.get(1).getPageContent();
    assertThat(documentsWithScores.size()).isEqualTo(3);
    assertThat(firstDocumentScore).isEqualTo(3);
    assertThat(firstDocumentScore).isLessThan(secondDocumentScore);
    assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
    assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
  }

  @Test
  void testSimilaritySearchStringWithScoreCosine() throws SQLException {
    pgVectorStore =
            new PGVectorStore(
                    embeddingProcessor, pgVectorStoreSpec, pgVectorService, DistanceStrategies.cosine());

    Triple<String, String, StringSimilaritySearchQuery> queryData = prepareSimilaritySearchStringQuery();
    StringSimilaritySearchQuery query = queryData.getThird();
    String firstPageContent = queryData.getFirst();
    String secondPageContent = queryData.getSecond();

    // Act.
    List<DomainDocument> documentsWithScores = pgVectorStore.similaritySearch(query);
    // Assert.
    double firstDocumentScore = documentsWithScores.get(0).getSimilarityScore().orElse(-1.0);
    double secondDocumentScore = documentsWithScores.get(1).getSimilarityScore().orElse(-1.0);
    String firstDocumentPageContent = documentsWithScores.get(0).getPageContent();
    String secondDocumentPageContent = documentsWithScores.get(1).getPageContent();
    assertThat(documentsWithScores.size()).isEqualTo(3);
    assertThat(Math.abs(firstDocumentScore - TOP_VECTOR_VALUE))
            .isLessThan(Math.abs(secondDocumentScore - TOP_VECTOR_VALUE));
    assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
    assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
  }

  @Test
  void testSimilaritySearchVectorWithScoreEuclidean() throws SQLException {
    pgVectorStore =
        new PGVectorStore(
            embeddingProcessor, pgVectorStoreSpec, pgVectorService, DistanceStrategies.euclidean());

    Triple<String, String, VectorSimilaritySearchQuery> queryData = prepareSimilaritySearchQuery();
    VectorSimilaritySearchQuery query = queryData.getThird();
    String firstPageContent = queryData.getFirst();
    String secondPageContent = queryData.getSecond();

    // Act.
    List<DomainDocument> documentsWithScores = pgVectorStore.similaritySearch(query);
    // Assert.
    double firstDocumentScore = documentsWithScores.get(0).getSimilarityScore().orElse(-1.0);
    double secondDocumentScore = documentsWithScores.get(1).getSimilarityScore().orElse(-1.0);
    String firstDocumentPageContent = documentsWithScores.get(0).getPageContent();
    String secondDocumentPageContent = documentsWithScores.get(1).getPageContent();
    assertThat(documentsWithScores.size()).isEqualTo(3);
    assertThat(firstDocumentScore).isEqualTo(0);
    assertThat(firstDocumentScore).isLessThan(secondDocumentScore);
    assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
    assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
  }

  @Test
  void testSimilaritySearchVectorWithScoreInnerProduct() throws SQLException {
    pgVectorStore =
        new PGVectorStore(
            embeddingProcessor,
            pgVectorStoreSpec,
            pgVectorService,
            DistanceStrategies.innerProduct());

    Triple<String, String, VectorSimilaritySearchQuery> queryData = prepareSimilaritySearchQuery();
    VectorSimilaritySearchQuery query = queryData.getThird();
    String firstPageContent = queryData.getFirst();
    String secondPageContent = queryData.getSecond();

    // Act.
    List<DomainDocument> documentsWithScores = pgVectorStore.similaritySearch(query);
    // Assert.
    double firstDocumentScore = documentsWithScores.get(0).getSimilarityScore().orElse(-1.0);
    double secondDocumentScore = documentsWithScores.get(1).getSimilarityScore().orElse(-1.0);
    String firstDocumentPageContent = documentsWithScores.get(0).getPageContent();
    String secondDocumentPageContent = documentsWithScores.get(1).getPageContent();
    assertThat(documentsWithScores.size()).isEqualTo(3);
    assertThat(firstDocumentScore).isEqualTo(3);
    assertThat(firstDocumentScore).isLessThan(secondDocumentScore);
    assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
    assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
  }

  @Test
  void testSimilaritySearchVectorWithScoreCosine() throws SQLException {
    pgVectorStore =
        new PGVectorStore(
            embeddingProcessor, pgVectorStoreSpec, pgVectorService, DistanceStrategies.cosine());

    Triple<String, String, VectorSimilaritySearchQuery> queryData = prepareSimilaritySearchQuery();
    VectorSimilaritySearchQuery query = queryData.getThird();
    String firstPageContent = queryData.getFirst();
    String secondPageContent = queryData.getSecond();

    // Act.
    List<DomainDocument> documentsWithScores = pgVectorStore.similaritySearch(query);
    // Assert.
    double firstDocumentScore = documentsWithScores.get(0).getSimilarityScore().orElse(-1.0);
    double secondDocumentScore = documentsWithScores.get(1).getSimilarityScore().orElse(-1.0);
    String firstDocumentPageContent = documentsWithScores.get(0).getPageContent();
    String secondDocumentPageContent = documentsWithScores.get(1).getPageContent();
    assertThat(documentsWithScores.size()).isEqualTo(3);
    assertThat(Math.abs(firstDocumentScore - TOP_VECTOR_VALUE))
        .isLessThan(Math.abs(secondDocumentScore - TOP_VECTOR_VALUE));
    assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
    assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
  }

  @Test
  void testUpdateDocuments() throws SQLException {
    EmbeddingOutput embeddingOutput = EmbeddingOutput.of(EmbeddingType.OPEN_AI, getEmbeddings());
    when(embeddingProcessor.run(ArgumentMatchers.any())).thenReturn(embeddingOutput);
    when(pgVectorService.prepareStatement(ArgumentMatchers.any())).thenReturn(preparedStatement);
    when(preparedStatement.executeUpdate()).thenReturn(DOCUMENT_COUNT, DOCUMENT_COUNT);

    // Act.
    boolean isSuccess = pgVectorStore.updateDocuments(getDocuments());
    // Assert.
    assertThat(isSuccess).isTrue();
  }

  @Test
  void testDeleteDocuments() throws SQLException {
    List<DomainDocument> documents = getDocuments();
    Mockito.when(pgVectorService.executeUpdate(ArgumentMatchers.any()))
        .thenReturn(documents.size());

    // Act.
    boolean isSuccess = pgVectorStore.deleteDocuments(documents);
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

  private Triple<String, String, VectorSimilaritySearchQuery> prepareSimilaritySearchQuery()
      throws SQLException {
    String firstPageContent = "content 0";
    String secondPageContent = "content 1";
    ResultSet resultSet = Mockito.mock(ResultSet.class);
    when(pgVectorService.prepareStatement(ArgumentMatchers.any())).thenReturn(preparedStatement);
    when(preparedStatement.executeQuery()).thenReturn(resultSet);
    when(resultSet.next()).thenReturn(true, true, true, false);
    when(resultSet.getObject(1))
        .thenReturn(
            UUID.randomUUID().toString(),
            UUID.randomUUID().toString(),
            UUID.randomUUID().toString());
    when(resultSet.getObject(2))
        .thenReturn(
            new PGvector(new float[] {TOP_VECTOR_VALUE, TOP_VECTOR_VALUE, TOP_VECTOR_VALUE}),
            new PGvector(new float[] {2.1f, 2.2f, 2.3f}),
            new PGvector(new float[] {-3, -3, -3}));
    when(resultSet.getObject(3)).thenReturn(textKey);
    when(resultSet.getObject(4)).thenReturn(firstPageContent, secondPageContent);
    double v = TOP_VECTOR_VALUE;
    VectorSimilaritySearchQuery query =
        VectorSimilaritySearchQuery.builder().setTopK(0L).setQuery(Arrays.asList(v, v, v)).build();
    return new Triple<>(firstPageContent, secondPageContent, query);
  }

  private Triple<String, String, StringSimilaritySearchQuery> prepareSimilaritySearchStringQuery()
          throws SQLException {
    String firstPageContent = "content 0";
    String secondPageContent = "content 1";
    ResultSet resultSet = Mockito.mock(ResultSet.class);
    double v = TOP_VECTOR_VALUE;
    List<Embedding> embeddings = ImmutableList.of(Embedding.of(Arrays.asList(v, v, v)));
    EmbeddingOutput embeddingOutput = EmbeddingOutput.of(EmbeddingType.OPEN_AI, embeddings);
    when(embeddingProcessor.run(ArgumentMatchers.any())).thenReturn(embeddingOutput);
    when(pgVectorService.prepareStatement(ArgumentMatchers.any())).thenReturn(preparedStatement);
    when(preparedStatement.executeQuery()).thenReturn(resultSet);
    when(resultSet.next()).thenReturn(true, true, true, false);
    when(resultSet.getObject(1))
            .thenReturn(
                    UUID.randomUUID().toString(),
                    UUID.randomUUID().toString(),
                    UUID.randomUUID().toString());
    when(resultSet.getObject(2))
            .thenReturn(
                    new PGvector(new float[] {TOP_VECTOR_VALUE, TOP_VECTOR_VALUE, TOP_VECTOR_VALUE}),
                    new PGvector(new float[] {2.1f, 2.2f, 2.3f}),
                    new PGvector(new float[] {-3, -3, -3}));
    when(resultSet.getObject(3)).thenReturn(textKey);
    when(resultSet.getObject(4)).thenReturn(firstPageContent, secondPageContent);

    StringSimilaritySearchQuery query =
            StringSimilaritySearchQuery.builder().setTopK(0L).setQuery("").build();
    return new Triple<>(firstPageContent, secondPageContent, query);
  }
}
