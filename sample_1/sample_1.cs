/*
 * Минимальный пример использования LLamaSharp
 */

#:package LLamaSharp@0.27.0

// вычислительное устройство, на котором будут производиться рассчеты
// для Mac на базе ARM-процессоров необходимо выбирать CPU - рассчеты все равно будут
// производиться на GPU
#:package LLamaSharp.Backend.Cpu@0.27.0

using LLama;
using LLama.Common;
using LLama.Native;

// подавляем логирование llama.cpp, для отладки рекомендуется включать,
// но в проде в нем нет большого смысла
NativeLibraryConfig.All.WithLogCallback((_, _) => { });

// имя модели и опции инференса
// скачать данную модель можно здесь: https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/tree/main
var parameters = new ModelParams("../model/gemma-4-E4B-it-Q4_0.gguf")
{
    // количество слоев модели, размещаемых в видеопамяти
    // в некоторых архитектурах можно загрузить в видеопамять только часть слоев, оставив
    // остальные в оперативной памяти для вычислений на ЦПУ. Для Mac на базе ARM-процессоров
    // нужно помещать в видеопамять все слои модели, иначе инференс не будет работать
    GpuLayerCount = 999
};

// загрузка модели (ее "весов") в память
using LLamaWeights model = LLamaWeights.LoadFromFile(parameters);

// создание "вычислителя" - моделя, который будет производить вычисления (инференс)
// данный вычислитель является простейшим вычислителем "без состояния" - каждый новый инференс
// запускает все вычисления заново, не хранит кэши и не поддерживает продолжение диалога
var executor = new StatelessExecutor(model, parameters);

var prompt = string.Join(' ', args);

// запускаем инференс
IAsyncEnumerable<string> result = executor.InferAsync(prompt);

// читаем декодированный текст по мере генерирования токенов
await foreach (var text in result)
{
    Console.Write(text);
}
