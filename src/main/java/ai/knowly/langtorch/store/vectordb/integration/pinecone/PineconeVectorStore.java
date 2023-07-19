package ai.knowly.langtorch.store.vectordb.integration.pinecone;

import ai.knowly.langtorch.processor.EmbeddingProcessor;
import ai.knowly.langtorch.schema.embeddings.EmbeddingInput;
import ai.knowly.langtorch.schema.embeddings.EmbeddingOutput;
import ai.knowly.langtorch.schema.io.DomainDocument;
import ai.knowly.langtorch.schema.io.Metadata;
import ai.knowly.langtorch.store.vectordb.integration.VectorStore;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.PineconeVectorStoreSpec;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.Vector;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.delete.DeleteRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.Match;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.QueryRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.query.QueryResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.update.UpdateRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.update.UpdateResponse;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.UpsertRequest;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.schema.dto.upsert.UpsertResponse;
import ai.knowly.langtorch.store.vectordb.integration.schema.StringSimilaritySearchQuery;
import com.google.common.collect.ImmutableList;

import ai.knowly.langtorch.store.vectordb.integration.schema.VectorSimilaritySearchQuery;
import com.google.common.flogger.FluentLogger;
import lombok.NonNull;

import javax.inject.Inject;
import java.util.*;
import java.util.concurrent.*;

/**
 * The PineconeVectorStore class is an implementation of the VectorStore interface, which provides
 * integration with the Pinecone service for storing and querying vectors.
 */
public class PineconeVectorStore implements VectorStore {

  private static final long UPDATE_TIMEOUT_SECONDS = 20;
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  private final EmbeddingProcessor embeddingProcessor;
  private final PineconeVectorStoreSpec pineconeVectorStoreSpec;
  @NonNull private final PineconeService pineconeService;
  private Optional<ExecutorService> executorService = Optional.empty();

  @Inject
  public PineconeVectorStore(
      EmbeddingProcessor embeddingProcessor,
      PineconeVectorStoreSpec pineconeVectorStoreSpec,
      @NonNull PineconeService pineconeService) {
    this.embeddingProcessor = embeddingProcessor;
    this.pineconeVectorStoreSpec = pineconeVectorStoreSpec;
    this.pineconeService = pineconeService;
  }

  /**
   * Adds the specified documents to the Pinecone vector store database.
   *
   * @return true if documents added successfully, otherwise false
   */
  @Override
  public boolean addDocuments(List<DomainDocument> documents) {
    if (documents.isEmpty()) {
      return true;
    }

    return addVectors(
        documents.stream().map(this::createVector).collect(ImmutableList.toImmutableList()));
  }

  /**
   * Adds a list of vectors to the Pinecone vector store database.
   *
   * @return true if vectors added successfully, otherwise false
   */
  private boolean addVectors(List<Vector> vectors) {
    UpsertRequest.UpsertRequestBuilder upsertRequestBuilder =
        UpsertRequest.builder().setVectors(vectors);
    this.pineconeVectorStoreSpec.getNamespace().ifPresent(upsertRequestBuilder::setNamespace);
    UpsertResponse response = this.pineconeService.upsert(upsertRequestBuilder.build());
    return response.getUpsertedCount() == vectors.size();
  }

  /**
   * Creates an instance of Vector from given DomainDocument
   *
   * @param document the document from which a Vector will be created
   * @return an instance of {@link Vector}
   */
  private Vector createVector(DomainDocument document) {
    EmbeddingOutput embeddingOutput = createEmbeddingOutput(document.getPageContent());
    return Vector.builder()
        .setId(document.getId().orElse(UUID.randomUUID().toString()))
        .setMetadata(document.getMetadata().orElse(Metadata.getDefaultInstance()).getValue())
        .setValues(embeddingOutput.getValue().get(0).getVector())
        .build();
  }

  /**
   * Performs a similarity search using a vector represenation of string query and returns a list of documents
   * containing their corresponding similarity scores.
   */
  @Override
  public List<DomainDocument> similaritySearch(StringSimilaritySearchQuery similaritySearchQuery) {
    List<Double> vector = createEmbeddingOutput(similaritySearchQuery.getQuery())
            .getValue().get(0).getVector();
    return similaritySearch(similaritySearchQuery.toVectorSimilaritySearchQuery(vector));
  }

  /**
   * Performs a similarity search using a vector query and returns a list of documents
   * containing their corresponding similarity scores.
   */
  @Override
  public List<DomainDocument> similaritySearch(VectorSimilaritySearchQuery similaritySearchQuery) {

    QueryRequest.QueryRequestBuilder requestBuilder =
        QueryRequest.builder()
            .setIncludeMetadata(true)
            .setTopK(similaritySearchQuery.getTopK())
            .setVector(similaritySearchQuery.getQuery())
            .setFilter(similaritySearchQuery.getFilter());

    pineconeVectorStoreSpec.getNamespace().ifPresent(requestBuilder::setNamespace);
    QueryResponse response = pineconeService.query(requestBuilder.build());

    List<DomainDocument> result = new ArrayList<>();

    // create mapping of PineCone metadata to schema meta data
    if (response.getMatches() != null) {
      for (Match match : response.getMatches()) {
        if (pineconeVectorStoreSpec.getTextKey().isEmpty()) {
          continue;
        }
        Metadata metadata =
            match.getMetadata() == null
                ? Metadata.getDefaultInstance()
                : Metadata.builder().setValue(match.getMetadata()).build();

        if (match.getScore() != null) {
          result.add(
              DomainDocument.builder()
                  .setPageContent(
                      metadata.getValue().get(this.pineconeVectorStoreSpec.getTextKey().get()))
                  .setMetadata(metadata)
                  .setSimilarityScore(Optional.of(match.getScore()))
                  .build());
        }
      }
    }

    return result;
  }

  @Override
  public boolean updateDocuments(List<DomainDocument> documents) {
    if (this.executorService.isEmpty()) {
      this.executorService = Optional.of(Executors.newFixedThreadPool(16));
    }
    ExecutorService localExecutorService = this.executorService.get();
    int responseCount = 0;

    CompletionService<UpdateResponse> updateCompletionService =
        new ExecutorCompletionService<>(localExecutorService);
    submitDocumentUpdateRequests(updateCompletionService, documents);

    while (responseCount < documents.size()) {
      try {
        updateCompletionService.take().get(UPDATE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        responseCount++;
      } catch (Exception e) {
        logger.atSevere().withCause(e).log(
            "Failed to update documents at document: " + documents.get(responseCount));
        Thread.currentThread().interrupt();
        return false;
      }
    }

    return true;
  }

  private void submitDocumentUpdateRequests(
      CompletionService<UpdateResponse> updateCompletionService, List<DomainDocument> documents) {
    for (DomainDocument document : documents) {
      updateCompletionService.submit(
          () -> {
            UpdateRequest.UpdateRequestBuilder requestBuilder = UpdateRequest.builder();
            pineconeVectorStoreSpec.getNamespace().ifPresent(requestBuilder::setNamespace);
            Vector vector = createVector(document);
            requestBuilder
                .setValues(vector.getValues())
                .setId(vector.getId())
                .setSetMetadata(vector.getMetadata())
                .build();
            return pineconeService.updateAsync(requestBuilder.build()).get();
          });
    }
  }

  @Override
  public boolean deleteDocuments(List<DomainDocument> documents) {
    List<String> documentIds = new ArrayList<>();
    for (DomainDocument document : documents) {
      document.getId().ifPresent(documentIds::add);
    }
    return deleteDocumentsByIds(documentIds);
  }

  @Override
  public boolean deleteDocumentsByIds(List<String> documentsIds) {
    if (documentsIds.isEmpty()) return false;

    DeleteRequest.DeleteRequestBuilder requestBuilder =
        DeleteRequest.builder().setIds(documentsIds);
    pineconeVectorStoreSpec.getNamespace().ifPresent(requestBuilder::setNamespace);
    try {
      pineconeService.delete(requestBuilder.build());
    } catch (Exception e) {
      logger.atSevere().withCause(e).log("Failed to delete documents");
      return false;
    }
    return true;
  }

  private EmbeddingOutput createEmbeddingOutput(String input) {
    return embeddingProcessor.run(
            EmbeddingInput.builder()
                    .setModel(pineconeVectorStoreSpec.getModel())
                    .setInput(Collections.singletonList(input))
                    .build());
  }
}
