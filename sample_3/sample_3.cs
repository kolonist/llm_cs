#:package LLamaSharp@0.27.0
#:package LLamaSharp.Backend.Cpu@0.27.0
#:property JsonSerializerIsReflectionEnabledByDefault=true

using LLama;
using LLama.Common;
using LLama.Native;

NativeLibraryConfig.All.WithLogCallback((_, _) => { });

var parameters = new ModelParams("../model/gemma-4-E4B-it-Q4_0.gguf")
{
    GpuLayerCount = 999
};

using LLamaWeights model = LLamaWeights.LoadFromFile(parameters);
using LLamaContext context = model.CreateContext(parameters);

var executor = new InteractiveExecutor(context);

const string KvCache = "kv-cache.bin";
const string ExecutorState = "executor-state.json";

if (File.Exists(KvCache) && File.Exists(ExecutorState))
{
    await executor.LoadState(ExecutorState);
    context.LoadState(KvCache);
}

var input = string.Join(' ', args);

var prompt = $"""
    <|turn>user:
    {input}
    <turn|>
    <|turn>model:

    """;

await foreach (var text in executor.InferAsync(prompt))
{
    Console.Write(text);
}

context.SaveState(KvCache);
await executor.SaveState(ExecutorState);
