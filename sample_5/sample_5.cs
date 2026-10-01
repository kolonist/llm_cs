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
    SeqMax = 16
};

using LLamaWeights model = LLamaWeights.LoadFromFile(parameters);
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
    using var sampler = new GreedySamplingPipeline();

    try
    {
        while (true)
        {
            reader.TryRead(out var request);

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
                    
                    var conversation = File.Exists(filename)
                        ? executor.Load(filename)
                        : executor.Create();

                    session = new Session(request.SessionId, filename, conversation);
                    sessions.Add(request.SessionId, session);
                }

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

                    session.Conversation.Prompt(
                        prompt,
                        addBos: session.Conversation.TokenCount == 0,
                        special: true);

                    session.Status = SessionStatus.Generating;                    
                }
            }

            await executor.Infer();

            foreach (var session in sessions.Values)
            {
                if (!session.Conversation.RequiresSampling)
                {
                    continue;
                }

                if (session.Status == SessionStatus.Finishing)
                {
                    await responses.WriteAsync(new Response(session.Id, session.Answer));

                    session.Conversation.Save(session.Filename);
                    session.Answer = "";
                    session.Status = SessionStatus.Idle;

                    continue;
                }

                var token = session.Conversation.Sample(sampler);
                session.Conversation.Prompt(token);

                if (token.IsEndOfGeneration(executor.Model.Vocab))
                {
                    session.Status = SessionStatus.Finishing;
                }
                else
                {
                    session.Decoder.Add(token);
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
    public Conversation Conversation { get; } = conversation;
    public StreamingTokenDecoder Decoder { get; } = new(conversation.Executor.Context);
    public string Answer { get; set; } = "";
    public SessionStatus Status { get; set; } = SessionStatus.Idle;

    public void Dispose()
    {
        Conversation.Dispose();
    }
}
