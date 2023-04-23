package ai.knowly.langtoch.llm.processor.openai.chat;

import static com.google.common.truth.Truth.assertThat;

import ai.knowly.langtoch.llm.schema.chat.ChatMessage;
import ai.knowly.langtoch.llm.schema.chat.Role;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ChatProcessorConfigAdapterTest {
  @Test
  public void testConvertChatMessage() {
    // Arrange.
    ai.knowly.langtoch.llm.schema.chat.ChatMessage chatMessage =
        ChatMessage.of(Role.ASSISTANT, "Hello, how can I help you?");

    // Act.
    com.theokanning.openai.completion.chat.ChatMessage convertedMessage =
        ChatProcessorRequestConverter.convertChatMessage(chatMessage);

    // Assert.
    assertThat(convertedMessage.getRole()).isEqualTo(Role.ASSISTANT.toString().toLowerCase());
    assertThat(convertedMessage.getContent()).isEqualTo("Hello, how can I help you?");
  }
}
