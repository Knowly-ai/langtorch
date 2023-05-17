package ai.knowly.langtoch.llm.processor.openai.chat;

import static com.google.common.collect.ImmutableList.toImmutableList;

import ai.knowly.langtoch.llm.integration.openai.service.schema.completion.chat.ChatCompletionRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.completion.chat.ChatMessage;
import java.util.List;

// Converter class to convert OpenAIChatProcessorConfig and a list of chat messages
// to a ChatCompletionRequest
public final class OpenAIChatProcessorRequestConverter {
  // Helper method to convert a chat message to the corresponding OpenAI chat message type
  public static ChatMessage convertChatMessage(
      ai.knowly.langtoch.schema.chat.ChatMessage chatMessage) {
    return new ChatMessage(chatMessage.getRole().name().toLowerCase(), chatMessage.getMessage());
  }

  // Method to convert OpenAIChatProcessorConfig and a list of chat messages
  // to a ChatCompletionRequest
  public static ChatCompletionRequest convert(
      OpenAIChatProcessorConfig openAIChatProcessorConfig,
      List<ai.knowly.langtoch.schema.chat.ChatMessage> messages) {
    ChatCompletionRequest.ChatCompletionRequestBuilder completionRequestBuilder =
        ChatCompletionRequest.builder()
            .model(openAIChatProcessorConfig.getModel())
            .messages(
                messages.stream()
                    .map(OpenAIChatProcessorRequestConverter::convertChatMessage)
                    .collect(toImmutableList()));

    // Set optional configuration properties
    openAIChatProcessorConfig.getTemperature().ifPresent(completionRequestBuilder::temperature);
    openAIChatProcessorConfig.getTopP().ifPresent(completionRequestBuilder::topP);
    openAIChatProcessorConfig.getN().ifPresent(completionRequestBuilder::n);
    openAIChatProcessorConfig.getStream().ifPresent(completionRequestBuilder::stream);
    if (!openAIChatProcessorConfig.getStop().isEmpty()) {
      completionRequestBuilder.stop(openAIChatProcessorConfig.getStop());
    }
    openAIChatProcessorConfig.getMaxTokens().ifPresent(completionRequestBuilder::maxTokens);
    openAIChatProcessorConfig
        .getPresencePenalty()
        .ifPresent(completionRequestBuilder::presencePenalty);
    openAIChatProcessorConfig
        .getFrequencyPenalty()
        .ifPresent(completionRequestBuilder::frequencyPenalty);
    completionRequestBuilder.logitBias(openAIChatProcessorConfig.getLogitBias());
    openAIChatProcessorConfig.getUser().ifPresent(completionRequestBuilder::user);
    // Build and return the ChatCompletionRequest
    return completionRequestBuilder.build();
  }
}
