package ai.knowly.langtorch.llm.openai.schema.dto;

import java.util.List;
import lombok.Data;

/** A wrapper class to fit the OpenAI engine and search endpoints */
@Data
public class OpenAIResponse<T> {
  /** A list containing the actual results */
  private List<T> data;

  /** The type of object returned, should be "list" */
  private String object;
}
