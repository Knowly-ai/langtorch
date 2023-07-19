package ai.knowly.langtorch.store.vectordb.integration.schema;

import lombok.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
@Builder(toBuilder = true, setterPrefix = "set")
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class StringSimilaritySearchQuery {
  @Builder.Default Map<String, String> filter = new HashMap<>();
  @NonNull private String query;
  @NonNull private Long topK;

  public VectorSimilaritySearchQuery toVectorSimilaritySearchQuery(List<Double> vector) {
    return VectorSimilaritySearchQuery.builder()
            .setFilter(getFilter())
            .setTopK(getTopK())
            .setQuery(vector)
            .build();
  }
}
