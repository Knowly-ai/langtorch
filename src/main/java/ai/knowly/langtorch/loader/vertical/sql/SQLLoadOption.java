package ai.knowly.langtorch.loader.vertical.sql;

import ai.knowly.langtorch.loader.LoadOption;
import java.util.Optional;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NonNull;

@Data
@Builder(toBuilder = true, setterPrefix = "set")
@AllArgsConstructor(access = lombok.AccessLevel.PRIVATE)
public class SQLLoadOption<S extends StorageObject> implements LoadOption {
  @NonNull private String query;
  @NonNull private StorageObjectTransformFunction<S> storageObjectTransformFunction;

  private String url;
  private String user;
  private String password;

  public Optional<String> getUrl() {
    return Optional.ofNullable(url);
  }

  public Optional<String> getUser() {
    return Optional.ofNullable(user);
  }

  public Optional<String> getPassword() {
    return Optional.ofNullable(password);
  }
}
