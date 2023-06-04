package ai.knowly.langtorch.loader.vertical.spreadsheet;

import ai.knowly.langtorch.loader.Loader;
import ai.knowly.langtorch.loader.vertical.spreadsheet.SpreadSheetLoadOption.SpreadSheetFormat;
import com.google.common.flogger.FluentLogger;
import java.io.FileReader;
import java.io.IOException;
import java.util.Optional;
import lombok.AllArgsConstructor;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

/** Implementation of DocumentConnector for CSV files. */
@AllArgsConstructor(access = lombok.AccessLevel.PRIVATE)
public class SpreadSheetLoader implements Loader<String> {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  SpreadSheetLoadOption readOption;

  public static SpreadSheetLoader create(SpreadSheetLoadOption readOption) {
    return new SpreadSheetLoader(readOption);
  }

  private CSVFormat convertCSVFormat(Optional<SpreadSheetFormat> spreadSheetFormat) {
    if (!spreadSheetFormat.isPresent()) {
      return CSVFormat.DEFAULT;
    }
    if (SpreadSheetFormat.CSV == spreadSheetFormat.get()) {
      return CSVFormat.DEFAULT;
    }
    if (SpreadSheetFormat.EXCEL == spreadSheetFormat.get()) {
      return CSVFormat.EXCEL;
    }
    return CSVFormat.DEFAULT;
  }

  @Override
  public Optional<String> read() {
    StringBuilder sb = new StringBuilder();
    try (FileReader fileReader = new FileReader(readOption.getFilePath())) {
      CSVParser csvParser =
          new CSVParser(fileReader, convertCSVFormat(readOption.getSpreadSheetFormat()));
      for (CSVRecord csvRecord : csvParser) {
        sb.append(csvRecord.toString()).append(readOption.getSeparatorForEachLine().orElse("\n"));
      }
    } catch (IOException e) {
      logger.atSevere().withCause(e).log("Error reading CSV file.");
      throw new SpreadsheetReadException(e);
    }

    return Optional.of(sb.toString());
  }
}
