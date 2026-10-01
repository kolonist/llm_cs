/*
 * Пример параллельного ведения нескольких диалогов с сохранением сессий
 */

#:package LLamaSharp@0.27.0
#:package LLamaSharp.Backend.Cpu@0.27.0
#:property JsonSerializerIsReflectionEnabledByDefault=true

using System.Threading.Channels;
using LLama;
using LLama.Batched;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

NativeLibraryConfig.All.WithLogCallback((_, _) => { });

var parameters = new ModelParams("../model/gemma-4-E4B-it-Q4_0.gguf")
{
    GpuLayerCount = 999,

    // максимальное количество последовательностей (диалогов) в общем контексте
    SeqMax = 16
};

using LLamaWeights model = LLamaWeights.LoadFromFile(parameters);

// пакетный вычислитель объединяет токены разных диалогов в общий инференс,
// при этом у каждого диалога остается своя последовательность в KV-кэше
using var executor = new BatchedExecutor(model, parameters);

var requests = Channel.CreateUnbounded<Request>();
var responses = Channel.CreateUnbounded<Response>();

var processing = ProcessRequests(requests.Reader, responses.Writer, executor);
var printing = PrintResponses(responses.Reader);

while (true)
{
    Console.Write("$ ");

    var input = Console.ReadLine()!;
    if (input == "")
    {
        break;
    }

    var parts = input.Split(' ', 2);
    await requests.Writer.WriteAsync(new Request(parts[0], parts[1]));
}

requests.Writer.Complete();

await processing;
await printing;


static async Task ProcessRequests(
    ChannelReader<Request> reader,
    ChannelWriter<Response> responses,
    BatchedExecutor executor)
{
    var sessions = new Dictionary<string, Session>();

    // семплер выбирает следующий токен из логитов - оценок, рассчитанных моделью
    // Greedy-семплер всегда выбирает токен с наибольшей оценкой
    using var sampler = new GreedySamplingPipeline();

    try
    {
        while (true)
        {
            reader.TryRead(out var request);

            // ждем новый запрос только когда нет токенов для обработки,
            // иначе продолжаем генерировать ответы уже запущенных сессий
            if (request is null && executor.BatchedTokenCount == 0)
            {
                try
                {
                    request = await reader.ReadAsync();
                }
                catch (ChannelClosedException)
                {
                    break;
                }
            }

            if (request is not null)
            {
                if (!sessions.TryGetValue(request.SessionId, out var session))
                {
                    var filename = $"session-{request.SessionId}.bin";

                    // Conversation хранит отдельный диалог в общем контексте
                    // его KV-кэш и позиция сохраняются вместе в один файл
                    var conversation = File.Exists(filename)
                        ? executor.Load(filename)
                        : executor.Create();

                    session = new Session(request.SessionId, filename, conversation);
                    sessions.Add(request.SessionId, session);
                }

                // разные сессии могут генерировать ответы одновременно,
                // но новый запрос в занятую сессию не принимаем
                if (session.Status != SessionStatus.Idle)
                {
                    await responses.WriteAsync(new Response(
                        session.Id,
                        "Ответ на предыдущий вопрос еще не готов, ждите"));
                }
                else
                {
                    var prompt = $"""
                        <|turn>user:
                        {request.Prompt}
                        <turn|>
                        <|turn>model:

                        """;

                    // добавляем промпт в пакет для последующих вычислений
                    // BOS (Begin Of Sequence) нужен только в начале диалога
                    // special разрешает распознавать спецтокены
                    session.Conversation.Prompt(
                        prompt,
                        addBos: session.Conversation.TokenCount == 0,
                        special: true);

                    session.Status = SessionStatus.Generating;
                }
            }

            // обрабатываем очередной пакет токенов всех запущенных диалогов
            await executor.Infer();

            foreach (var session in sessions.Values)
            {
                // семплировать можно только когда вычислены логиты для этого диалога
                // длинный промпт может потребовать нескольких вызовов Infer
                if (!session.Conversation.RequiresSampling)
                {
                    continue;
                }

                // перед сохранением нужно обязательно обработать спецтокен конца ответа отдельным инференсом,
                // иначе он останется в пакете и Conversation не позволит сохранить сессию
                if (session.Status == SessionStatus.Finishing)
                {
                    await responses.WriteAsync(new Response(session.Id, session.Answer));

                    // сохранение KV-кэша и состояния контекста конкретного диалога
                    session.Conversation.Save(session.Filename);

                    session.Answer = "";
                    session.Status = SessionStatus.Idle;

                    continue;
                }

                // сэиплировение токена
                var token = session.Conversation.Sample(sampler);

                // выбранный токен отправляем на следующий инференс,
                // чтобы добавить его в KV-кэш и получить логиты для следующего токена
                session.Conversation.Prompt(token);

                // конец генерации определяем по словарю модели
                if (token.IsEndOfGeneration(executor.Model.Vocab))
                {
                    session.Status = SessionStatus.Finishing;
                }
                else
                {
                    // добавление нового токена в декодер
                    session.Decoder.Add(token);

                    // непосредственно декодирование накопленной в декодере информации
                    // добавление и декодирование разделены на 2 операции и проводятся через внутреннее
                    // состояние декодера потому, что не каждый токен может быть декодирован в текст (символ
                    // или последовательность символов) и для некоторых последоательностей симвоов
                    // необходимо набрать более одного токена. В случае, если внутри декодера лежит недостаточно
                    // токенов для их декодлирования в текст, данный метод вернет пустую строку
                    session.Answer += session.Decoder.Read();
                }
            }
        }
    }
    finally
    {
        foreach (var session in sessions.Values)
        {
            session.Dispose();
        }

        responses.Complete();
    }
}

static async Task PrintResponses(ChannelReader<Response> responses)
{
    await foreach (var response in responses.ReadAllAsync())
    {
        Console.WriteLine($"\r[{response.SessionId}] {response.Text}");
        Console.Write("$ ");
    }
}

record Request(string SessionId, string Prompt);
record Response(string SessionId, string Text);

enum SessionStatus
{
    Idle,
    Generating,
    Finishing
}

sealed class Session(string id, string filename, Conversation conversation) : IDisposable
{
    public string Id { get; } = id;
    public string Filename { get; } = filename;

    // абстракция "разговора" (диалога), ведущегося в одном контексте
    public Conversation Conversation { get; } = conversation;

    // декодер собирает текст из токенов, один символ может занимать несколько токенов
    // каждому диалогу нужен свой декодер, чтобы не смешивать незавершенные символы
    public StreamingTokenDecoder Decoder { get; } = new(conversation.Executor.Context);

    public string Answer { get; set; } = "";
    public SessionStatus Status { get; set; } = SessionStatus.Idle;

    public void Dispose()
    {
        Conversation.Dispose();
    }
}
