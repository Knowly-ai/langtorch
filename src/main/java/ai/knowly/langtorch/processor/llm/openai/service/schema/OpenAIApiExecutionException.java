package ai.knowly.langtorch.processor.llm.openai.service.schema;

public class OpenAIApiExecutionException extends RuntimeException {
  public OpenAIApiExecutionException(Exception e) {
    super(e);
  }
}
