package ai.knowly.langtoch.llm.integration.openai.service;

import ai.knowly.langtoch.llm.integration.openai.service.schema.OpenAIError;
import ai.knowly.langtoch.llm.integration.openai.service.schema.OpenAIHttpException;
import ai.knowly.langtoch.llm.integration.openai.service.schema.completion.CompletionRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.completion.CompletionResult;
import ai.knowly.langtoch.llm.integration.openai.service.schema.completion.chat.ChatCompletionRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.completion.chat.ChatCompletionResult;
import ai.knowly.langtoch.llm.integration.openai.service.schema.edit.EditRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.edit.EditResult;
import ai.knowly.langtoch.llm.integration.openai.service.schema.embedding.EmbeddingRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.embedding.EmbeddingResult;
import ai.knowly.langtoch.llm.integration.openai.service.schema.image.CreateImageEditRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.image.CreateImageRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.image.CreateImageVariationRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.image.ImageResult;
import ai.knowly.langtoch.llm.integration.openai.service.schema.moderation.ModerationRequest;
import ai.knowly.langtoch.llm.integration.openai.service.schema.moderation.ModerationResult;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import io.reactivex.BackpressureStrategy;
import io.reactivex.Flowable;
import io.reactivex.Single;
import java.io.IOException;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import okhttp3.*;
import retrofit2.Call;
import retrofit2.HttpException;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory;
import retrofit2.converter.jackson.JacksonConverterFactory;

public class OpenAIService {

  private static final String BASE_URL = "https://api.openai.com/";
  private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(10);
  private static final ObjectMapper mapper = defaultObjectMapper();

  private final OpenAIApi api;
  private final ExecutorService executorService;

  /**
   * Creates a new OpenAiService that wraps OpenAiApi
   *
   * @param token OpenAi token string "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
   */
  public OpenAIService(final String token) {
    this(token, DEFAULT_TIMEOUT);
  }

  /**
   * Creates a new OpenAiService that wraps OpenAiApi
   *
   * @param token OpenAi token string "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
   * @param timeout http read timeout, Duration.ZERO means no timeout
   */
  public OpenAIService(final String token, final Duration timeout) {
    ObjectMapper mapper = defaultObjectMapper();
    OkHttpClient client = defaultClient(token, timeout);
    Retrofit retrofit = defaultRetrofit(client, mapper);

    this.api = retrofit.create(OpenAIApi.class);
    this.executorService = client.dispatcher().executorService();
  }

  /**
   * Creates a new OpenAiService that wraps OpenAiApi. Use this if you need more customization, but
   * use OpenAiService(api, executorService) if you use streaming and want to shut down instantly
   *
   * @param api OpenAiApi instance to use for all methods
   */
  public OpenAIService(final OpenAIApi api) {
    this.api = api;
    this.executorService = null;
  }

  /**
   * Creates a new OpenAiService that wraps OpenAiApi. The ExecutorService must be the one you get
   * from the client you created the api with otherwise shutdownExecutor() won't work.
   *
   * <p>Use this if you need more customization.
   *
   * @param api OpenAiApi instance to use for all methods
   * @param executorService the ExecutorService from client.dispatcher().executorService()
   */
  public OpenAIService(final OpenAIApi api, final ExecutorService executorService) {
    this.api = api;
    this.executorService = executorService;
  }

  /** Calls the Open AI api, returns the response, and parses error messages if the request fails */
  public static <T> T execute(Single<T> apiCall) {
    try {
      return apiCall.blockingGet();
    } catch (HttpException e) {
      try {
        if (e.response() == null || e.response().errorBody() == null) {
          throw e;
        }
        String errorBody = e.response().errorBody().string();

        OpenAIError error = mapper.readValue(errorBody, OpenAIError.class);
        throw new OpenAIHttpException(error, e, e.code());
      } catch (IOException ex) {
        // couldn't parse OpenAI error
        throw e;
      }
    }
  }

  /**
   * Calls the Open AI api and returns a Flowable of SSE for streaming omitting the last message.
   *
   * @param apiCall The api call
   */
  public static Flowable<SSE> stream(Call<ResponseBody> apiCall) {
    return stream(apiCall, false);
  }

  /**
   * Calls the Open AI api and returns a Flowable of SSE for streaming.
   *
   * @param apiCall The api call
   * @param emitDone If true the last message ([DONE]) is emitted
   */
  public static Flowable<SSE> stream(Call<ResponseBody> apiCall, boolean emitDone) {
    return Flowable.create(
        emitter -> apiCall.enqueue(new ResponseBodyCallback(emitter, emitDone)),
        BackpressureStrategy.BUFFER);
  }

  /**
   * Calls the Open AI api and returns a Flowable of type T for streaming omitting the last message.
   *
   * @param apiCall The api call
   * @param cl Class of type T to return
   */
  public static <T> Flowable<T> stream(Call<ResponseBody> apiCall, Class<T> cl) {
    return stream(apiCall).map(sse -> mapper.readValue(sse.getData(), cl));
  }

  public static OpenAIApi buildApi(String token, Duration timeout) {
    ObjectMapper mapper = defaultObjectMapper();
    OkHttpClient client = defaultClient(token, timeout);
    Retrofit retrofit = defaultRetrofit(client, mapper);

    return retrofit.create(OpenAIApi.class);
  }

  public static ObjectMapper defaultObjectMapper() {
    ObjectMapper mapper = new ObjectMapper();
    mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
    mapper.setPropertyNamingStrategy(PropertyNamingStrategy.SNAKE_CASE);
    return mapper;
  }

  public static OkHttpClient defaultClient(String token, Duration timeout) {
    return new OkHttpClient.Builder()
        .addInterceptor(new OpenAIAuthenticationInterceptor(token))
        .connectionPool(new ConnectionPool(5, 1, TimeUnit.SECONDS))
        .readTimeout(timeout.toMillis(), TimeUnit.MILLISECONDS)
        .build();
  }

  public static Retrofit defaultRetrofit(OkHttpClient client, ObjectMapper mapper) {
    return new Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(client)
        .addConverterFactory(JacksonConverterFactory.create(mapper))
        .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
        .build();
  }

  public CompletionResult createCompletion(CompletionRequest request) {
    return execute(api.createCompletion(request));
  }

  public ChatCompletionResult createChatCompletion(ChatCompletionRequest request) {
    return execute(api.createChatCompletion(request));
  }

  public EditResult createEdit(EditRequest request) {
    return execute(api.createEdit(request));
  }

  public EmbeddingResult createEmbeddings(EmbeddingRequest request) {
    return execute(api.createEmbeddings(request));
  }

  public ImageResult createImage(CreateImageRequest request) {
    return execute(api.createImage(request));
  }

  public ImageResult createImageEdit(
      CreateImageEditRequest request, String imagePath, String maskPath) {
    java.io.File image = new java.io.File(imagePath);
    java.io.File mask = null;
    if (maskPath != null) {
      mask = new java.io.File(maskPath);
    }
    return createImageEdit(request, image, mask);
  }

  public ImageResult createImageEdit(
      CreateImageEditRequest request, java.io.File image, java.io.File mask) {
    RequestBody imageBody = RequestBody.create(MediaType.parse("image"), image);

    MultipartBody.Builder builder =
        new MultipartBody.Builder()
            .setType(MediaType.get("multipart/form-data"))
            .addFormDataPart("prompt", request.getPrompt())
            .addFormDataPart("size", request.getSize())
            .addFormDataPart("response_format", request.getResponseFormat())
            .addFormDataPart("image", "image", imageBody);

    if (request.getN() != null) {
      builder.addFormDataPart("n", request.getN().toString());
    }

    if (mask != null) {
      RequestBody maskBody = RequestBody.create(MediaType.parse("image"), mask);
      builder.addFormDataPart("mask", "mask", maskBody);
    }

    return execute(api.createImageEdit(builder.build()));
  }

  public ImageResult createImageVariation(CreateImageVariationRequest request, String imagePath) {
    java.io.File image = new java.io.File(imagePath);
    return createImageVariation(request, image);
  }

  public ImageResult createImageVariation(CreateImageVariationRequest request, java.io.File image) {
    RequestBody imageBody = RequestBody.create(MediaType.parse("image"), image);

    MultipartBody.Builder builder =
        new MultipartBody.Builder()
            .setType(MediaType.get("multipart/form-data"))
            .addFormDataPart("size", request.getSize())
            .addFormDataPart("response_format", request.getResponseFormat())
            .addFormDataPart("image", "image", imageBody);

    if (request.getN() != null) {
      builder.addFormDataPart("n", request.getN().toString());
    }

    return execute(api.createImageVariation(builder.build()));
  }

  public ModerationResult createModeration(ModerationRequest request) {
    return execute(api.createModeration(request));
  }

  /**
   * Shuts down the OkHttp ExecutorService. The default behaviour of OkHttp's ExecutorService
   * (ConnectionPool) is to shut down after an idle timeout of 60s. Call this method to shut down
   * the ExecutorService immediately.
   */
  public void shutdownExecutor() {
    Objects.requireNonNull(
        this.executorService, "executorService must be set in order to shut down");
    this.executorService.shutdown();
  }
}