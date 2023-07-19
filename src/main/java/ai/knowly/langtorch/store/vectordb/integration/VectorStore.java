package ai.knowly.langtorch.store.vectordb.integration;

import ai.knowly.langtorch.schema.io.DomainDocument;
import ai.knowly.langtorch.store.vectordb.integration.schema.StringSimilaritySearchQuery;
import ai.knowly.langtorch.store.vectordb.integration.schema.VectorSimilaritySearchQuery;
import java.util.List;

/** A shared interface for all Vector Store Databases */
public interface VectorStore {

  boolean addDocuments(List<DomainDocument> documents);

  List<DomainDocument> similaritySearch(VectorSimilaritySearchQuery similaritySearchQuery);

  List<DomainDocument> similaritySearch(StringSimilaritySearchQuery similaritySearchQuery);

  boolean updateDocuments(List<DomainDocument> documents);

  boolean deleteDocuments(List<DomainDocument> documents);

  boolean deleteDocumentsByIds(List<String> documentsIds);
}
