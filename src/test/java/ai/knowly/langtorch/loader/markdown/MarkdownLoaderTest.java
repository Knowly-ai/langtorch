package ai.knowly.langtorch.loader.markdown;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import org.junit.jupiter.api.Test;

class MarkdownLoaderTest {

  @Test
  void testMdNotExist() throws IOException {
    // Arrange.
    String testFilePath = "src/test/resources/test1.md";

    // Act.
    IOException e =
        assertThrows(IOException.class, () -> MarkdownLoader.create().read(testFilePath));

    // Assert.
    assertThat(e).hasMessageThat().contains(testFilePath);
  }

  @Test
  void testReadMd() throws IOException {
    // Arrange.
    String testFilePath = "src/test/resources/test.md";

    // Act.
    String result = MarkdownLoader.create().read(testFilePath);

    // Assert.
    String expectedContent =
        "# Example Markdown File\n"
            + "\n"
            + "This is a sample Markdown file for testing purposes. It contains various Markdown"
            + " elements such as headings, lists, and emphasis.\n"
            + "\n"
            + "## Lists\n"
            + "\n"
            + "Here's an example of an unordered list:\n"
            + "\n"
            + "- Item 1\n"
            + "- Item 2\n"
            + "- Item 3\n"
            + "\n"
            + "```java\n"
            + "public class HelloWorld {\n"
            + "    public static void main(String[] args) {\n"
            + "        System.out.println(\"Hello, world!\");\n"
            + "    }\n"
            + "}";
    assertThat(result).isEqualTo(expectedContent);
  }
}
