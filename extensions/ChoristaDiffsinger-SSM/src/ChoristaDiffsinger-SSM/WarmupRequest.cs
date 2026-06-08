using System.Collections.Generic;
using System.Linq;

namespace DiffsingerSSM;

internal enum WarmupTensorDType
{
    Int64,
    Float32,
}

internal sealed class WarmupTensor
{
    public required string Name { get; init; }
    public required WarmupTensorDType DType { get; init; }
    public required int[] Shape { get; init; }
}

/// <summary>
/// Builder result describing a small dummy ORT input set we feed to the acoustic session
/// once at voicebank load.  Goal: pre-warm the cumsum/exp graph so the user's first real
/// synthesis doesn't take 5–10 seconds of cold-start.
/// </summary>
internal sealed class WarmupRequest
{
    public int FrameLength { get; init; }
    public IReadOnlyList<WarmupTensor> Tensors { get; init; } = System.Array.Empty<WarmupTensor>();

    /// <summary>
    /// Constants chosen so the dummy run is *cheap* but exercises every layer.
    /// 16 frames at 44.1k/hop=512 ≈ 186 ms of "audio" — short enough to keep warmup under
    /// a second, long enough for batch/seq dims to be non-trivial in the SSM scan.
    /// </summary>
    private const int DummyFrameLength = 16;

    public static WarmupRequest Build(
        IReadOnlyList<string> inputNames,
        int phonemeCount,
        int sampleRate,
        int hopSize)
    {
        var names = inputNames.ToHashSet();
        var tensors = new List<WarmupTensor>();

        if (names.Contains("tokens"))
            tensors.Add(new WarmupTensor { Name = "tokens", DType = WarmupTensorDType.Int64, Shape = new[] { 1, phonemeCount } });
        if (names.Contains("durations"))
            tensors.Add(new WarmupTensor { Name = "durations", DType = WarmupTensorDType.Int64, Shape = new[] { 1, phonemeCount } });
        if (names.Contains("f0"))
            tensors.Add(new WarmupTensor { Name = "f0", DType = WarmupTensorDType.Float32, Shape = new[] { 1, DummyFrameLength } });
        if (names.Contains("languages"))
            tensors.Add(new WarmupTensor { Name = "languages", DType = WarmupTensorDType.Int64, Shape = new[] { 1, phonemeCount } });
        if (names.Contains("spk_embed"))
            tensors.Add(new WarmupTensor { Name = "spk_embed", DType = WarmupTensorDType.Float32, Shape = new[] { 1, DummyFrameLength, 256 } });
        if (names.Contains("gender"))
            tensors.Add(new WarmupTensor { Name = "gender", DType = WarmupTensorDType.Float32, Shape = new[] { 1, DummyFrameLength } });
        if (names.Contains("velocity"))
            tensors.Add(new WarmupTensor { Name = "velocity", DType = WarmupTensorDType.Float32, Shape = new[] { 1, DummyFrameLength } });
        if (names.Contains("energy"))
            tensors.Add(new WarmupTensor { Name = "energy", DType = WarmupTensorDType.Float32, Shape = new[] { 1, DummyFrameLength } });
        if (names.Contains("breathiness"))
            tensors.Add(new WarmupTensor { Name = "breathiness", DType = WarmupTensorDType.Float32, Shape = new[] { 1, DummyFrameLength } });
        if (names.Contains("voicing"))
            tensors.Add(new WarmupTensor { Name = "voicing", DType = WarmupTensorDType.Float32, Shape = new[] { 1, DummyFrameLength } });
        if (names.Contains("tension"))
            tensors.Add(new WarmupTensor { Name = "tension", DType = WarmupTensorDType.Float32, Shape = new[] { 1, DummyFrameLength } });
        if (names.Contains("depth"))
            tensors.Add(new WarmupTensor { Name = "depth", DType = WarmupTensorDType.Float32, Shape = new[] { 1 } });
        // Modern (continuous-acceleration) models declare "steps"; legacy models declare "speedup".
        // Always emit exactly one based on which one the model asks for.
        if (names.Contains("steps"))
            tensors.Add(new WarmupTensor { Name = "steps", DType = WarmupTensorDType.Int64, Shape = new[] { 1 } });
        else if (names.Contains("speedup"))
            tensors.Add(new WarmupTensor { Name = "speedup", DType = WarmupTensorDType.Int64, Shape = new[] { 1 } });

        return new WarmupRequest
        {
            FrameLength = DummyFrameLength,
            Tensors = tensors,
        };
    }
}