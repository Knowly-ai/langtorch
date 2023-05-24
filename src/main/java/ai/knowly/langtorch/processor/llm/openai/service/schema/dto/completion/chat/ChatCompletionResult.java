package ai.knowly.langtorch.processor.llm.openai.service.schema.dto.completion.chat;

import ai.knowly.langtorch.processor.llm.openai.service.schema.dto.Usage;
import java.util.List;
import lombok.Data;

/** Object containing a response from the chat completions api. */
@Data
public class ChatCompletionResult {

  /** Unique id assigned to this chat completion. */
  String id;

  /** The type of object returned, should be "chat.completion" */
  String object;

  /** The creation time in epoch seconds. */
  long created;

  /** The GPT model used. */
  String model;

  /** A list of all generated completions. */
  List<ChatCompletionChoice> choices;

  /** The API usage for this request. */
  Usage usage;
}
