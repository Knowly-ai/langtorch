package ai.knowly.langtorch.capability.module.openai;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import ai.knowly.langtorch.processor.llm.openai.service.OpenAIService;
import ai.knowly.langtorch.processor.module.openai.text.OpenAITextProcessor;
import ai.knowly.langtorch.util.OpenAIServiceTestingUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
final class SimpleTextCapabilityTest {
  @Mock private OpenAIService openAIService;

  @Test
  void simpleTest() {
    // Arrange.
    when(openAIService.createCompletion(any()))
        .thenReturn(
            OpenAIServiceTestingUtils.TextCompletion.createCompletionResult(
                "Changsha is a city in Hunan province, China."));

    // Act.
    String output =
        SimpleTextCapability.create(OpenAITextProcessor.create(openAIService))
            .run("Where is Changsha?");

    // Assert.
    assertThat(output).isEqualTo("Changsha is a city in Hunan province, China.");
  }
}
