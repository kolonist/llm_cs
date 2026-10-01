/*
 * Пример ведения диалога в едином контексте
 */

#:package LLamaSharp@0.27.0
#:package LLamaSharp.Backend.Cpu@0.27.0

using LLama;
using LLama.Common;
using LLama.Native;

NativeLibraryConfig.All.WithLogCallback((_, _) => { });

var parameters = new ModelParams("../model/gemma-4-E4B-it-Q4_0.gguf")
{
    GpuLayerCount = 999
};

using LLamaWeights model = LLamaWeights.LoadFromFile(parameters);

// объект для управления контекстом
using LLamaContext context = model.CreateContext(parameters);

// интерактивный вычислитель позволяет добавлять новые промпты в общий контекст
var executor = new InteractiveExecutor(context);

while (true)
{
    Console.Write("$ ");

    var input = Console.ReadLine()!;
    if (input == "")
    {
        break;
    }

    // разметка промпта специальными токенами
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

    Console.WriteLine();
}
