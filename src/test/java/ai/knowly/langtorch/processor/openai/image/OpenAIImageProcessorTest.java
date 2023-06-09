package ai.knowly.langtorch.processor.openai.image;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import ai.knowly.langtorch.llm.openai.OpenAIService;
import ai.knowly.langtorch.llm.openai.schema.dto.image.Image;
import ai.knowly.langtorch.llm.openai.schema.dto.image.ImageResult;
import ai.knowly.langtorch.schema.image.Images;
import ai.knowly.langtorch.schema.text.SingleText;
import com.google.inject.testing.fieldbinder.Bind;
import java.util.Arrays;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
final class OpenAIImageProcessorTest {
  @Mock @Bind private OpenAIService openAIService;

  @Test
  void testRun() {
    // Arrange.
    SingleText inputData = SingleText.of("image description");
    OpenAIImageProcessorConfig config = OpenAIImageProcessorConfig.builder().setN(2).build();
    OpenAIImageProcessor openAIImageProcessor = new OpenAIImageProcessor(openAIService, config);

    ImageResult expectedResult = new ImageResult();
    expectedResult.setCreated(123L);
    Image image1 = new Image();
    image1.setUrl("image1-url");
    Image image2 = new Image();
    image1.setUrl("image2-url");

    expectedResult.setData(Arrays.asList(image1, image2));

    when(openAIService.createImage(
            OpenAIImageProcessorRequestConverter.convert(config, "image description")))
        .thenReturn(expectedResult);

    // Act.
    Images output = openAIImageProcessor.run(inputData);

    // Assert.
    assertThat(output.getCreated()).isEqualTo(expectedResult.getCreated());
    assertThat(
            output.getImageData().stream()
                .map(ai.knowly.langtorch.schema.image.Image::getUrl)
                .collect(Collectors.toList()))
        .containsExactlyElementsIn(
            expectedResult.getData().stream().map(Image::getUrl).collect(Collectors.toList()));
  }
}
