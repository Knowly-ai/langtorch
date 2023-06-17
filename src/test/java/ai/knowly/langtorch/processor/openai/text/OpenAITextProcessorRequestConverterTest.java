package ai.knowly.langtorch.processor.openai.text;

import static com.google.common.truth.Truth.assertThat;

import ai.knowly.langtorch.llm.openai.schema.dto.completion.CompletionRequest;
import com.google.common.collect.ImmutableMap;
import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
final class OpenAITextProcessorRequestConverterTest {

  private OpenAITextProcessorConfig openAITextProcessorConfig;

  @BeforeEach
  public void setUp() {
    openAITextProcessorConfig =
        OpenAITextProcessorConfig.builder()
            .setModel("text-davinci-003")
            .setSuffix("Test suffix")
            .setMaxTokens(100)
            .setTemperature(0.8)
            .setTopP(0.9)
            .setN(5)
            .setStream(true)
            .setLogprobs(10)
            .setEcho(true)
            .setStop(Arrays.asList("stop1", "stop2"))
            .setPresencePenalty(0.5)
            .setFrequencyPenalty(0.6)
            .setBestOf(3)
            .setLogitBias(ImmutableMap.of("key1", 50, "key2", -20))
            .setUser("user123")
            .build();
  }

  @Test
  void testConvert() {
    CompletionRequest completionRequest =
        OpenAITextProcessorRequestConverter.convert(openAITextProcessorConfig, "Test prompt");

    assertThat(completionRequest.getModel()).isEqualTo(openAITextProcessorConfig.getModel());
    assertThat(completionRequest.getPrompt()).isEqualTo("Test prompt");
    assertThat(completionRequest.getSuffix())
        .isEqualTo(openAITextProcessorConfig.getSuffix().get());
    assertThat(completionRequest.getMaxTokens())
        .isEqualTo(openAITextProcessorConfig.getMaxTokens().get());
    assertThat(completionRequest.getTemperature())
        .isEqualTo(openAITextProcessorConfig.getTemperature().get());
    assertThat(completionRequest.getTopP()).isEqualTo(openAITextProcessorConfig.getTopP().get());
    assertThat(completionRequest.getN()).isEqualTo(openAITextProcessorConfig.getN().get());
    assertThat(completionRequest.getStream())
        .isEqualTo(openAITextProcessorConfig.getStream().get());
    assertThat(completionRequest.getLogprobs())
        .isEqualTo(openAITextProcessorConfig.getLogprobs().get());
    assertThat(completionRequest.getEcho()).isEqualTo(openAITextProcessorConfig.getEcho().get());
    assertThat(completionRequest.getStop()).isEqualTo(openAITextProcessorConfig.getStop());
    assertThat(completionRequest.getPresencePenalty())
        .isEqualTo(openAITextProcessorConfig.getPresencePenalty().get());
    assertThat(completionRequest.getFrequencyPenalty())
        .isEqualTo(openAITextProcessorConfig.getFrequencyPenalty().get());
    assertThat(completionRequest.getBestOf())
        .isEqualTo(openAITextProcessorConfig.getBestOf().get());
    assertThat(completionRequest.getLogitBias())
        .isEqualTo(openAITextProcessorConfig.getLogitBias());
    assertThat(completionRequest.getUser()).isEqualTo(openAITextProcessorConfig.getUser().get());
  }
}
