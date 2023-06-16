package ai.knowly.langtorch.store.vectordb.integration.pgvector;

import ai.knowly.langtorch.processor.EmbeddingProcessor;
import ai.knowly.langtorch.schema.embeddings.Embedding;
import ai.knowly.langtorch.schema.embeddings.EmbeddingOutput;
import ai.knowly.langtorch.schema.embeddings.EmbeddingType;
import ai.knowly.langtorch.schema.io.DomainDocument;
import ai.knowly.langtorch.schema.io.Metadata;
import ai.knowly.langtorch.store.vectordb.integration.pgvector.schema.PGVectorStoreSpec;
import ai.knowly.langtorch.store.vectordb.integration.pgvector.schema.distance.DistanceStrategy;
import ai.knowly.langtorch.store.vectordb.integration.schema.SimilaritySearchQuery;
import com.google.common.collect.ImmutableMap;
import com.pgvector.PGvector;
import kotlin.Triple;
import org.apache.commons.lang3.tuple.Pair;
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
import static java.util.Collections.emptyList;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
    void setUp() {
        textKey = "text_key";
        embeddingProcessor = Mockito.mock(EmbeddingProcessor.class);
        pgVectorService = Mockito.mock(PGVectorService.class);
        preparedStatement = Mockito.mock(PreparedStatement.class);

        pgVectorStoreSpec = PGVectorStoreSpec.builder()
                .setTextKey(textKey)
                .setPgVectorService(pgVectorService)
                .setDistanceStrategy(DistanceStrategy.euclidean())
                .setDatabaseName("test")
                .setVectorDimensions(3)
                .build();

        pgVectorStore = new PGVectorStore(embeddingProcessor, pgVectorStoreSpec);
    }

    @Test
    public void testAddDocuments() throws SQLException {
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
    public void testAddDocumentsEmpty() {
        // Act.
        Exception exception = assertThrows(IllegalStateException.class, () -> pgVectorStore.addDocuments(emptyList()));
        String expectedMessage = "Attempted to add an empty list";
        //Assert.
        assertThat(exception.getMessage()).isEqualTo(expectedMessage);
    }

    @Test
    public void testSimilaritySearchVectorWithScoreEuclidean() throws SQLException {
        pgVectorStoreSpec = pgVectorStoreSpec.toBuilder()
                .setDistanceStrategy(DistanceStrategy.euclidean())
                .build();
        pgVectorStore = new PGVectorStore(embeddingProcessor, pgVectorStoreSpec);

        Triple<String, String, SimilaritySearchQuery> queryData = prepareSimilaritySearchQuery();
        SimilaritySearchQuery query = queryData.getThird();
        String firstPageContent = queryData.getFirst();
        String secondPageContent = queryData.getSecond();

        // Act.
        List<Pair<DomainDocument, Double>> documentsWithScores = pgVectorStore.similaritySearchVectorWithScore(query);
        // Assert.
        double firstDocumentScore = documentsWithScores.get(0).getRight();
        double secondDocumentScore = documentsWithScores.get(1).getRight();
        String firstDocumentPageContent = documentsWithScores.get(0).getLeft().getPageContent();
        String secondDocumentPageContent = documentsWithScores.get(1).getLeft().getPageContent();
        assertThat(documentsWithScores.size()).isEqualTo(3);
        assertThat(firstDocumentScore).isEqualTo(0);
        assertThat(firstDocumentScore).isLessThan(secondDocumentScore);
        assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
        assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
    }

    @Test
    public void testSimilaritySearchVectorWithScoreInnerProduct() throws SQLException {
        pgVectorStoreSpec = pgVectorStoreSpec.toBuilder()
                .setDistanceStrategy(DistanceStrategy.innerProduct())
                .build();
        pgVectorStore = new PGVectorStore(embeddingProcessor, pgVectorStoreSpec);

        Triple<String, String, SimilaritySearchQuery> queryData = prepareSimilaritySearchQuery();
        SimilaritySearchQuery query = queryData.getThird();
        String firstPageContent = queryData.getFirst();
        String secondPageContent = queryData.getSecond();

        // Act.
        List<Pair<DomainDocument, Double>> documentsWithScores = pgVectorStore.similaritySearchVectorWithScore(query);
        // Assert.
        double firstDocumentScore = documentsWithScores.get(0).getRight();
        double secondDocumentScore = documentsWithScores.get(1).getRight();
        String firstDocumentPageContent = documentsWithScores.get(0).getLeft().getPageContent();
        String secondDocumentPageContent = documentsWithScores.get(1).getLeft().getPageContent();
        assertThat(documentsWithScores.size()).isEqualTo(3);
        assertThat(firstDocumentScore).isEqualTo(3);
        assertThat(firstDocumentScore).isLessThan(secondDocumentScore);
        assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
        assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
    }

    @Test
    public void testSimilaritySearchVectorWithScoreCosine() throws SQLException {
        pgVectorStoreSpec = pgVectorStoreSpec.toBuilder()
                .setDistanceStrategy(DistanceStrategy.cosine())
                .build();
        pgVectorStore = new PGVectorStore(embeddingProcessor, pgVectorStoreSpec);

        Triple<String, String, SimilaritySearchQuery> queryData = prepareSimilaritySearchQuery();
        SimilaritySearchQuery query = queryData.getThird();
        String firstPageContent = queryData.getFirst();
        String secondPageContent = queryData.getSecond();

        // Act.
        List<Pair<DomainDocument, Double>> documentsWithScores = pgVectorStore.similaritySearchVectorWithScore(query);
        // Assert.
        double firstDocumentScore = documentsWithScores.get(0).getRight();
        double secondDocumentScore = documentsWithScores.get(1).getRight();
        String firstDocumentPageContent = documentsWithScores.get(0).getLeft().getPageContent();
        String secondDocumentPageContent = documentsWithScores.get(1).getLeft().getPageContent();
        assertThat(documentsWithScores.size()).isEqualTo(3);
        assertThat(Math.abs(firstDocumentScore - TOP_VECTOR_VALUE)).isLessThan(Math.abs(secondDocumentScore - TOP_VECTOR_VALUE));
        assertThat(firstDocumentPageContent).isEqualTo(firstPageContent);
        assertThat(secondDocumentPageContent).isEqualTo(secondPageContent);
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

    private Triple<String, String, SimilaritySearchQuery> prepareSimilaritySearchQuery() throws SQLException {
        String firstPageContent = "content 0";
        String secondPageContent = "content 1";
        ResultSet resultSet = Mockito.mock(ResultSet.class);
        when(pgVectorService.prepareStatement(ArgumentMatchers.any())).thenReturn(preparedStatement);
        when(preparedStatement.executeQuery()).thenReturn(resultSet);
        when(resultSet.next()).thenReturn(true, true, true, false);
        when(resultSet.getObject(1)).thenReturn(
                UUID.randomUUID().toString(),
                UUID.randomUUID().toString(),
                UUID.randomUUID().toString()
        );
        when(resultSet.getObject(2)).thenReturn(
                new PGvector(new float[]{TOP_VECTOR_VALUE, TOP_VECTOR_VALUE, TOP_VECTOR_VALUE}),
                new PGvector(new float[]{2.1f, 2.2f, 2.3f}),
                new PGvector(new float[]{-3, -3, -3})
        );
        when(resultSet.getObject(3)).thenReturn(textKey);
        when(resultSet.getObject(4)).thenReturn(firstPageContent, secondPageContent);
        double v = TOP_VECTOR_VALUE;
        SimilaritySearchQuery query = SimilaritySearchQuery.builder()
                .setTopK(0L)
                .setQuery(Arrays.asList(v, v, v))
                .build();
        return new Triple<>(firstPageContent, secondPageContent, query);
    }

}
