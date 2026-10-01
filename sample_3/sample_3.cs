/*
 * Пример сохранения диалога и продолжения после перезапуска программы
 */

#:package LLamaSharp@0.27.0
#:package LLamaSharp.Backend.Cpu@0.27.0

// разрешаем сериализацию через рефлексию - она используется внутри вычислителя
// для сохранения и загрузки его состояния
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

// имя файла, в который будет сохранено содержимое KV-кэша
const string KvCache = "kv-cache.bin";

// имя файла, в который будут сохранены параменты вычислителя (например, текущее смешение в KV-кэше,
// необходимое для продолжение с места остановки после восстановления KV-кэша из файла)
const string ExecutorState = "executor-state.json";

if (File.Exists(KvCache) && File.Exists(ExecutorState))
{
    // для продолжения диалога нужно загрузить состояние вычислителя и KV-кэш контекста
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

// контекст сохраняет KV-кэш - результаты вычислений для уже обработанных токенов,
// а вычислитель - свою позицию в диалоге и токены, ожидающие обработки
context.SaveState(KvCache);
await executor.SaveState(ExecutorState);
