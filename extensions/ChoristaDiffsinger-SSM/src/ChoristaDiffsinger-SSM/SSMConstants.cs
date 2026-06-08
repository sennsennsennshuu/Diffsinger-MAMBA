using System.Collections.Generic;
using System.Linq;
using Diffsinger3rdApi.IOSys;
using TuneLab.Base.Properties;

namespace DiffsingerSSM;

/// <summary>
/// Engine-private property keys & default configs.  All keys are deliberately prefixed
/// with "DiffSinger-SSM:" so they cannot collide with Choristad's keys when both engines
/// happen to be loaded in the same TuneLab session.
/// </summary>
internal static class SSMConstants
{
    public const string PitchTransitionTimeID = "DiffSinger-SSM:PitchTransitionTime";
    public static readonly NumberConfig PitchTransitionTimeConfig =
        new(defaultValue: 0.12, minValue: 0.0, maxValue: 0.2, isInterger: false);

    public const string MinSegmentSpacingID = "DiffSinger-SSM:MinSegmentSpacingMs";
    public static readonly NumberConfig MinSegmentSpacingConfig =
        new(defaultValue: 0.0, minValue: 0.0, maxValue: 64.0, isInterger: false);

    public const string PhonemizerID = "DiffSinger-SSM:Phonemizer";

    /// <summary>
    /// Diffusion sampler step count.  reflow-style ckpts can run shorter (5-10) but the
    /// useful range depends on training; we expose 1-100 with default 20.  Internally
    /// this overwrites <see cref="Diffsinger3rdApi.Util.DiffsingerPreferences.Default.DiffSingerSteps"/>
    /// for the duration of one synth task — that field is a plain public static int, not
    /// a thread-locked resource.
    /// </summary>
    public const string RenderStepsID = "DiffSinger-SSM:RenderSteps";
    public static readonly NumberConfig RenderStepsConfig =
        new(defaultValue: 20.0, minValue: 1.0, maxValue: 100.0, isInterger: true);

    /// <summary>
    /// Shallow-diffusion source mix.  Only meaningful when the acoustic model was trained
    /// with use_shallow_diffusion=true and use_variable_depth=true.  Range mirrors the
    /// model's max_depth (typically 0.6 for SSM voicebanks).
    /// </summary>
    public const string RenderDepthID = "DiffSinger-SSM:RenderDepth";
    public static readonly NumberConfig RenderDepthConfig =
        new(defaultValue: 1.0, minValue: 0.0, maxValue: 1.0, isInterger: false);

    /// <summary>
    /// On-disk render cache toggle.  Choristad caches per-phrase ORT outputs by xxHash64
    /// to avoid recomputing identical phrases.  Disabling it forces every synth to re-run
    /// the full graph — useful when iterating on parameters and the cache hit defeats the
    /// purpose.
    /// </summary>
    public const string TensorCacheID = "DiffSinger-SSM:TensorCache";
    public static readonly BooleanConfig TensorCacheConfig =
        new(defaultValue: true);

    /// <summary>
    /// 渲染设备：Auto / CPU / GPU.
    ///
    /// 直接映射到 <see cref="Diffsinger3rdApi.Util.DiffsingerPreferences.Default.OnnxRunner"/>:
    ///   Auto → "autogpu"   (Onnx.cs 在 Windows 上自动选 CUDA / DirectML / CPU)
    ///   CPU  → "cpu"
    ///   GPU  → "autogpu"   (语义同 Auto，但配合 RenderDevice=GPU 时会强制走 GPU 路径；
    ///                       autogpu 已经会优先 CUDA/DML，所以这里复用)
    ///
    /// 切换设备后，必须把已经缓存的 <c>InferenceSession</c> 释放掉，否则旧 session 仍跑在
    /// 上次的 EP 上 — 见 <see cref="DiffsingerSSMTask.ApplyRenderPreferences"/>。
    /// </summary>
    public const string RenderDeviceID = "DiffSinger-SSM:RenderDevice";
    public static readonly EnumConfig RenderDeviceConfig =
        new(new List<string> { "Auto", "CPU", "GPU" }, 0);

    /// <summary>
    /// Choristad declares a fixed enum of phonemizer keys at engine load (registry lives in
    /// <see cref="PhonemizerProcesser.DiffsingerPhonemizer"/>).  We mirror the same set so
    /// the dropdown a user sees in TuneLab is the union of "Default" and the registry keys.
    /// "Default" defers to the singer's character.yaml.
    /// </summary>
    public static EnumConfig PhonemizerConfig
    {
        get
        {
            var keys = new List<string> { "Default" };
            keys.AddRange(PhonemizerProcesser.DiffsingerPhonemizer.Keys);
            return new EnumConfig(keys, 0);
        }
    }

    /// <summary>
    /// Number of samples the warmup dummy run targets.  Kept tiny (16 frames ≈ 186 ms at
    /// 44.1k/hop=512) so the user-visible startup cost is sub-second on a modern CPU.
    /// </summary>
    public const int WarmupFrameLength = 16;
}