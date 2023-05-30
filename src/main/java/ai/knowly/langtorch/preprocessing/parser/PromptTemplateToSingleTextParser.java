package ai.knowly.langtorch.preprocessing.parser;

import ai.knowly.langtorch.schema.text.SingleText;
import ai.knowly.langtorch.prompt.template.PromptTemplate;

public class PromptTemplateToSingleTextParser implements Parser<PromptTemplate, SingleText> {

  private PromptTemplateToSingleTextParser() {
    super();
  }

  public static PromptTemplateToSingleTextParser create() {
    return new PromptTemplateToSingleTextParser();
  }

  @Override
  public SingleText parse(PromptTemplate input) {
    return SingleText.of(input.format());
  }
}
