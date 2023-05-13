package ai.knowly.langtoch.util;

import com.google.common.collect.ImmutableList;
import com.theokanning.openai.completion.CompletionChoice;
import com.theokanning.openai.completion.CompletionResult;
import com.theokanning.openai.completion.chat.ChatCompletionChoice;
import com.theokanning.openai.completion.chat.ChatCompletionResult;
import com.theokanning.openai.completion.chat.ChatMessage;
import com.theokanning.openai.embedding.EmbeddingResult;

import java.util.List;

public class OpenAIServiceTestingUtils {

  // ChatCompletion related utils
  public static class ChatCompletion {
    public static ChatCompletionResult createChatCompletionResult(
        ai.knowly.langtoch.schema.chat.ChatMessage message) {
      return createChatCompletionResult(ImmutableList.of(message));
    }

    public static ChatCompletionResult createChatCompletionResult(
        ai.knowly.langtoch.schema.chat.ChatMessage... messages) {
      return createChatCompletionResult(ImmutableList.copyOf(messages));
    }

    public static ChatCompletionResult createChatCompletionResult(
        Iterable<ai.knowly.langtoch.schema.chat.ChatMessage> chatMessages) {
      ChatCompletionResult completionResult = new ChatCompletionResult();
      completionResult.setChoices(createChatCompletionChoices(chatMessages));
      return completionResult;
    }

    public static ImmutableList<ChatCompletionChoice> createChatCompletionChoices(
        ai.knowly.langtoch.schema.chat.ChatMessage... chatMessages) {
      return createChatCompletionChoices(ImmutableList.copyOf(chatMessages));
    }

    public static ImmutableList<ChatCompletionChoice> createChatCompletionChoices(
        Iterable<ai.knowly.langtoch.schema.chat.ChatMessage> chatMessages) {
      ImmutableList.Builder<ChatCompletionChoice> builder = ImmutableList.builder();
      for (ai.knowly.langtoch.schema.chat.ChatMessage chatMessage : chatMessages) {
        ChatCompletionChoice completionChoice = new ChatCompletionChoice();
        completionChoice.setMessage(createChatMessage(chatMessage));
        builder.add(completionChoice);
      }
      return builder.build();
    }

    public static ChatMessage createChatMessage(ai.knowly.langtoch.schema.chat.ChatMessage msg) {
      ChatMessage chatMessage = new ChatMessage();
      chatMessage.setContent(msg.getMessage());
      chatMessage.setRole(msg.getRole().toString().toLowerCase());
      return chatMessage;
    }
  }

  // TextCompletion related utils
  public static class TextCompletion {
    public static CompletionResult createCompletionResult(String text) {
      return createCompletionResult(ImmutableList.of(text));
    }

    public static CompletionResult createCompletionResult(String... texts) {
      return createCompletionResult(ImmutableList.copyOf(texts));
    }

    public static CompletionResult createCompletionResult(Iterable<String> texts) {
      CompletionResult completionResult = new CompletionResult();
      completionResult.setChoices(createCompletionChoices(texts));
      return completionResult;
    }

    public static ImmutableList<CompletionChoice> createCompletionChoices(String... texts) {
      return createCompletionChoices(ImmutableList.copyOf(texts));
    }

    public static ImmutableList<CompletionChoice> createCompletionChoices(Iterable<String> texts) {
      ImmutableList.Builder<CompletionChoice> builder = ImmutableList.builder();
      for (String text : texts) {
        CompletionChoice completionChoice = new CompletionChoice();
        completionChoice.setText(text);
        builder.add(completionChoice);
      }
      return builder.build();
    }
  }

  public static class Embeddings {
    public static EmbeddingResult createQueryEmbeddingResult(String inputData) {
      EmbeddingResult embeddingResult = new EmbeddingResult();
      embeddingResult.setObject("list");
      return embeddingResult;
    }
  }
}
