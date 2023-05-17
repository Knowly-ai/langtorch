package ai.knowly.langtoch.llm.processor.openai.embeddings;

import static com.google.common.collect.ImmutableList.toImmutableList;

import ai.knowly.langtoch.llm.integration.openai.service.OpenAIService;
import ai.knowly.langtoch.llm.integration.openai.service.schema.embedding.EmbeddingResult;
import ai.knowly.langtoch.llm.processor.Processor;
import ai.knowly.langtoch.llm.processor.openai.OpenAIServiceProvider;
import ai.knowly.langtoch.schema.embeddings.Embedding;
import ai.knowly.langtoch.schema.embeddings.EmbeddingType;
import ai.knowly.langtoch.schema.embeddings.Embeddings;
import ai.knowly.langtoch.schema.io.EmbeddingInput;
import com.google.common.flogger.FluentLogger;
import com.google.common.util.concurrent.ListenableFuture;

public class OpenAIEmbeddingsProcessor implements Processor<EmbeddingInput, Embeddings> {

  private static final FluentLogger logger = FluentLogger.forEnclosingClass();
  private final OpenAIService openAIService;

  private OpenAIEmbeddingsProcessorConfig openAIEmbeddingsProcessorConfig =
      OpenAIEmbeddingsProcessorConfig.builder().build();

  public OpenAIEmbeddingsProcessor(OpenAIService openAiApi) {
    this.openAIService = openAiApi;
  }

  private OpenAIEmbeddingsProcessor() {
    this.openAIService = OpenAIServiceProvider.createOpenAIService();
  }

  public static OpenAIEmbeddingsProcessor create(OpenAIService openAIService) {
    return new OpenAIEmbeddingsProcessor(openAIService);
  }

  public static OpenAIEmbeddingsProcessor create(String openAIKey) {
    return new OpenAIEmbeddingsProcessor(OpenAIServiceProvider.createOpenAIService(openAIKey));
  }

  public static OpenAIEmbeddingsProcessor create() {
    return new OpenAIEmbeddingsProcessor();
  }

  public OpenAIEmbeddingsProcessor withConfig(
      OpenAIEmbeddingsProcessorConfig openAIEmbeddingsProcessorConfig) {
    this.openAIEmbeddingsProcessorConfig = openAIEmbeddingsProcessorConfig;
    return this;
  }

  @Override
  public Embeddings run(EmbeddingInput inputData) {
    EmbeddingResult embeddingResult =
        openAIService.createEmbeddings(
            OpenAIEmbeddingsProcessorRequestConverter.convert(
                openAIEmbeddingsProcessorConfig, inputData.getModel(), inputData.getInput()));
    return Embeddings.of(
        EmbeddingType.OPEN_AI,
        embeddingResult.getData().stream()
            .map(embedding -> Embedding.of(embedding.getEmbedding()))
            .collect(toImmutableList()));
  }

  @Override
  public ListenableFuture<Embeddings> runAsync(EmbeddingInput inputData) {
    return null;
  }
}
