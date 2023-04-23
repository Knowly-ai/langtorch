package ai.knowly.langtoch.llm.provider.openai;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import ai.knowly.langtoch.llm.processor.ProcessorType;
import ai.knowly.langtoch.llm.processor.openai.chat.OpenAIChatProcessor;
import ai.knowly.langtoch.llm.processor.openai.chat.OpenAIChatProcessorConfig;
import ai.knowly.langtoch.llm.processor.openai.chat.OpenAIChatProcessorRequestConverter;
import ai.knowly.langtoch.llm.processor.openai.text.OpenAITextProcessor;
import ai.knowly.langtoch.llm.processor.openai.text.OpenAITextProcessorConfig;
import ai.knowly.langtoch.llm.processor.openai.text.OpenAITextProcessorRequestConverter;
import ai.knowly.langtoch.llm.schema.chat.Role;
import ai.knowly.langtoch.llm.schema.chat.UserMessage;
import com.theokanning.openai.completion.CompletionChoice;
import com.theokanning.openai.completion.CompletionRequest;
import com.theokanning.openai.completion.CompletionResult;
import com.theokanning.openai.completion.chat.ChatCompletionChoice;
import com.theokanning.openai.completion.chat.ChatCompletionRequest;
import com.theokanning.openai.completion.chat.ChatCompletionResult;
import com.theokanning.openai.completion.chat.ChatMessage;
import com.theokanning.openai.service.OpenAiService;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;

@RunWith(MockitoJUnitRunner.class)
public class OpenAITest {
  private static final OpenAIChatProcessorConfig openAIChatProcessorConfig =
      OpenAIChatProcessorConfig.builder().setModel("gpt-3.5-turbo").setMaxTokens(2048).build();
  private static final OpenAITextProcessorConfig openAITextProcessorConfig =
      OpenAITextProcessorConfig.builder().setModel("text-davinci-003").setMaxTokens(2048).build();
  @Mock private OpenAiService openAiService;

  @Test
  public void runWithTextProcessorTest() {
    // Arrange.
    OpenAITextProcessor openAITextProcessor = OpenAITextProcessor.create(openAiService);

    OpenAI openAI =
        OpenAI.create().withProcessor(ProcessorType.TEXT_PROCESSOR, openAITextProcessor);
    CompletionRequest completionRequest =
        OpenAITextProcessorRequestConverter.convert(openAITextProcessorConfig, "Hi!");

    CompletionResult completionResult = new CompletionResult();
    CompletionChoice completionChoice = new CompletionChoice();
    completionChoice.setText("What can i do for you?");
    completionResult.setChoices(Arrays.asList(completionChoice));

    when(openAiService.createCompletion(completionRequest)).thenReturn(completionResult);

    // Act.
    String response = openAI.runTextProcessor("Hi!");
    // Assert.
    assertThat(response).isEqualTo("What can i do for you?");
  }

  @Test
  public void runWithChatProcessorTest() {
    // Arrange.
    OpenAIChatProcessor openAIChatProcessor = OpenAIChatProcessor.create(openAiService);
    OpenAI openAI =
        OpenAI.create().withProcessor(ProcessorType.CHAT_PROCESSOR, openAIChatProcessor);
    ChatCompletionRequest completionRequest =
        OpenAIChatProcessorRequestConverter.convert(
            openAIChatProcessorConfig,
            List.of(ai.knowly.langtoch.llm.schema.chat.ChatMessage.of(Role.USER, "Hi!")));

    ChatCompletionResult completionResult = new ChatCompletionResult();

    ChatMessage chatMessage = new ChatMessage();
    chatMessage.setContent("What can i do for you?");
    chatMessage.setRole("assistant");

    ChatCompletionChoice completionChoice = new ChatCompletionChoice();
    completionChoice.setMessage(chatMessage);
    completionResult.setChoices(Arrays.asList(completionChoice));

    when(openAiService.createChatCompletion(completionRequest)).thenReturn(completionResult);

    // Act.
    ai.knowly.langtoch.llm.schema.chat.ChatMessage message =
        openAI.runChatProcessor(UserMessage.of("Hi!"));
    // Assert.
    assertThat(message.getMessage()).isEqualTo("What can i do for you?");
  }

  @Test
  public void runWithMultipleProcessorTest() {
    // Arrange.
    OpenAIChatProcessor openAIChatProcessor = OpenAIChatProcessor.create(openAiService);
    OpenAITextProcessor openAITextProcessor = OpenAITextProcessor.create(openAiService);
    OpenAI openAI =
        OpenAI.create()
            .withProcessor(ProcessorType.CHAT_PROCESSOR, openAIChatProcessor)
            .withProcessor(ProcessorType.TEXT_PROCESSOR, openAITextProcessor);
    ChatCompletionRequest chatCompletionRequest =
        OpenAIChatProcessorRequestConverter.convert(
            openAIChatProcessorConfig,
            List.of(
                ai.knowly.langtoch.llm.schema.chat.ChatMessage.of(
                    Role.USER, "Where is Changsha?")));

    ChatCompletionResult chatCompletionResult = new ChatCompletionResult();
    ChatMessage chatMessage = new ChatMessage();
    chatMessage.setContent("It's in hunan province, China.");
    chatMessage.setRole("assistant");
    ChatCompletionChoice chatCompletionChoice = new ChatCompletionChoice();
    chatCompletionChoice.setMessage(chatMessage);
    chatCompletionResult.setChoices(Arrays.asList(chatCompletionChoice));
    when(openAiService.createChatCompletion(chatCompletionRequest))
        .thenReturn(chatCompletionResult);

    CompletionRequest completionRequest =
        OpenAITextProcessorRequestConverter.convert(openAITextProcessorConfig, "Hi!");
    CompletionResult completionResult = new CompletionResult();
    CompletionChoice completionChoice = new CompletionChoice();
    completionChoice.setText("What can i do for you?");
    completionResult.setChoices(Arrays.asList(completionChoice));
    when(openAiService.createCompletion(completionRequest)).thenReturn(completionResult);

    // Act.
    ai.knowly.langtoch.llm.schema.chat.ChatMessage message =
        openAI.runChatProcessor(UserMessage.of("Where is Changsha?"));
    String response = openAI.runTextProcessor("Hi!");
    // Assert.
    assertThat(message.getMessage()).isEqualTo("It's in hunan province, China.");
    assertThat(response).isEqualTo("What can i do for you?");
  }
}
