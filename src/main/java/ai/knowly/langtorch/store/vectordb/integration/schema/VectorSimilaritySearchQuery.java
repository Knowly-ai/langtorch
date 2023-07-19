package ai.knowly.langtorch.store.vectordb.integration.schema;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NonNull;

@Data
@Builder(toBuilder = true, setterPrefix = "set")
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class VectorSimilaritySearchQuery {
  @Builder.Default Map<String, String> filter = new HashMap<>();
  @NonNull private List<Double> query;
  @NonNull private Long topK;
}
