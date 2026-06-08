using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace DiffsingerSSM.Tests;

/// <summary>
/// SSM warmup spec.  After loading a voicebank we want to dispatch a tiny dummy run so the
/// first user-triggered synthesis doesn't pay the full cumsum/exp ORT graph cold-start cost.
/// We can't unit-test against a real onnx, so we test the request-builder that decides
/// the tensor shapes / dtypes.
/// </summary>
public class WarmupRequestBuilderTests
{
    [Fact]
    public void Builds_Single_Frame_Length_By_Default()
    {
        var req = WarmupRequest.Build(
            inputNames: new[] { "tokens", "durations", "f0", "speedup" },
            phonemeCount: 8,
            sampleRate: 44100,
            hopSize: 512);
        Assert.Equal(16, req.FrameLength);
    }

    [Fact]
    public void Includes_Lang_Tensor_Only_When_Requested()
    {
        var withLang = WarmupRequest.Build(
            inputNames: new[] { "tokens", "durations", "f0", "languages", "speedup" },
            phonemeCount: 4, sampleRate: 44100, hopSize: 512);
        var withoutLang = WarmupRequest.Build(
            inputNames: new[] { "tokens", "durations", "f0", "speedup" },
            phonemeCount: 4, sampleRate: 44100, hopSize: 512);

        Assert.Contains("languages", withLang.Tensors.Select(t => t.Name));
        Assert.DoesNotContain("languages", withoutLang.Tensors.Select(t => t.Name));
    }

    [Fact]
    public void Token_Tensor_Has_Shape_1xN_And_Type_Long()
    {
        var req = WarmupRequest.Build(
            inputNames: new[] { "tokens", "durations", "f0" },
            phonemeCount: 5, sampleRate: 44100, hopSize: 512);
        var tokens = req.Tensors.Single(t => t.Name == "tokens");
        Assert.Equal(WarmupTensorDType.Int64, tokens.DType);
        Assert.Equal(new[] { 1, 5 }, tokens.Shape);
    }

    [Fact]
    public void F0_Tensor_Length_Matches_Total_Frame_Duration()
    {
        var req = WarmupRequest.Build(
            inputNames: new[] { "tokens", "durations", "f0" },
            phonemeCount: 5, sampleRate: 44100, hopSize: 512);
        var f0 = req.Tensors.Single(t => t.Name == "f0");
        Assert.Equal(WarmupTensorDType.Float32, f0.DType);
        Assert.Equal(new[] { 1, req.FrameLength }, f0.Shape);
    }

    [Fact]
    public void Steps_And_Speedup_Are_Mutually_Exclusive()
    {
        // Acoustic models with useContinuousAcceleration use "steps", legacy uses "speedup".
        // The builder must include exactly one based on input names — never both, never neither.
        var modern = WarmupRequest.Build(
            inputNames: new[] { "tokens", "durations", "f0", "depth", "steps" },
            phonemeCount: 4, sampleRate: 44100, hopSize: 512);
        var legacy = WarmupRequest.Build(
            inputNames: new[] { "tokens", "durations", "f0", "speedup" },
            phonemeCount: 4, sampleRate: 44100, hopSize: 512);

        Assert.Contains("steps", modern.Tensors.Select(t => t.Name));
        Assert.DoesNotContain("speedup", modern.Tensors.Select(t => t.Name));
        Assert.Contains("speedup", legacy.Tensors.Select(t => t.Name));
        Assert.DoesNotContain("steps", legacy.Tensors.Select(t => t.Name));
    }

    [Fact]
    public void Skips_Unrecognized_Input_Names_With_No_Throw()
    {
        // The builder must be tolerant: if the model declares an input name we don't know about,
        // the warmup result is still emitted (without that input) so the user's first synth
        // attempt isn't blocked by warmup throwing.
        var req = WarmupRequest.Build(
            inputNames: new[] { "tokens", "durations", "f0", "WeirdNewInput" },
            phonemeCount: 4, sampleRate: 44100, hopSize: 512);
        Assert.NotNull(req);
        Assert.DoesNotContain("WeirdNewInput", req.Tensors.Select(t => t.Name));
    }
}