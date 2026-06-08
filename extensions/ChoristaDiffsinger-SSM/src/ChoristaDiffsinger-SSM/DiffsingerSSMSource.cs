using System;
using System.Collections.Generic;
using Diffsinger3rdApi.DiffSinger;
using Diffsinger3rdApi.IOSys;
using TuneLab.Base.Properties;
using TuneLab.Base.Structures;
using TuneLab.Extensions.Voices;

namespace DiffsingerSSM;

/// <summary>
/// Wraps a <see cref="DiffSingerSinger"/> as a TuneLab <see cref="IVoiceSource"/>.
///
/// We intentionally lean on <see cref="RenderCurveDefine"/> from Diffsinger3rdApi to derive
/// the automation/note/part property maps so the SSM engine offers exactly the same set of
/// curves as the reference Choristad engine — users moving between them shouldn't lose UI.
///
/// Per-engine extras (kept distinct from Choristad to avoid PartProperties key clashes):
///   * "PitchTransitionTime"   — cross-fades pitch between notes (s)
///   * "MinSegmentSpacingMs"   — gap below which two notes are merged into one synth segment
///   * "Phonemizer"            — overrides character.yaml's default_phonemizer
/// </summary>
internal sealed class DiffsingerSSMSource : IVoiceSource
{
    private readonly DiffSingerSinger _singer;
    private readonly OrderedMap<string, AutomationConfig> _automations;
    private readonly OrderedMap<string, IPropertyConfig> _partProps;
    private readonly OrderedMap<string, IPropertyConfig> _noteProps;

    public string Name => "DiffSinger-SSM : " + _singer.Name;
    public string DefaultLyric => "a";

    public IReadOnlyOrderedMap<string, AutomationConfig> AutomationConfigs => _automations;
    public IReadOnlyOrderedMap<string, IPropertyConfig> PartProperties => _partProps;
    public IReadOnlyOrderedMap<string, IPropertyConfig> NoteProperties => _noteProps;

    public DiffsingerSSMSource(DiffSingerSinger singer)
    {
        _singer = singer ?? throw new ArgumentNullException(nameof(singer));

        _automations = RenderCurveDefine.ParameterSwitcher(singer);
        _noteProps   = RenderCurveDefine.NotePropsSwitcher(singer);
        _partProps   = RenderCurveDefine.PartPropsSwitcher(singer);

        // Engine-level controls.  These IDs intentionally use a "DiffSinger-SSM:" prefix so
        // they never collide with anything Choristad puts in PartProperties.
        _partProps.Add(SSMConstants.PitchTransitionTimeID, SSMConstants.PitchTransitionTimeConfig);
        _partProps.Add(SSMConstants.MinSegmentSpacingID,    SSMConstants.MinSegmentSpacingConfig);
        _partProps.Add(SSMConstants.PhonemizerID,           SSMConstants.PhonemizerConfig);
        _partProps.Add(SSMConstants.RenderStepsID,          SSMConstants.RenderStepsConfig);
        _partProps.Add(SSMConstants.RenderDepthID,          SSMConstants.RenderDepthConfig);
        _partProps.Add(SSMConstants.TensorCacheID,          SSMConstants.TensorCacheConfig);
        _partProps.Add(SSMConstants.RenderDeviceID,         SSMConstants.RenderDeviceConfig);

        // Best-effort warmup.  If anything goes wrong (model schema changed, no GPU, etc.)
        // we proceed silently — warmup is an optimisation, not a correctness contract.
        TryWarmup();
    }

    public IReadOnlyList<SynthesisSegment<T>> Segment<T>(SynthesisSegment<T> segment) where T : ISynthesisNote
    {
        var minSpacingMs = segment.PartProperties.GetDouble(
            SSMConstants.MinSegmentSpacingID,
            SSMConstants.MinSegmentSpacingConfig.DefaultValue);
        return this.SimpleSegment(segment, minSpacingMs / 1000.0);
    }

    public ISynthesisTask CreateSynthesisTask(ISynthesisData data)
        => new DiffsingerSSMTask(_singer, data);

    private void TryWarmup()
    {
        // Cold-start cost on Diffsinger3rdApi splits into:
        //   (a) Onnx.getInferenceSession ctor       — model deserialisation + EP init
        //   (b) first session.Run                  — kernel selection / memory tuning
        // (a) accounts for ~70-90% on the SSM acoustic onnx (302 MB) and is what users
        // perceive as "TuneLab hangs for 5-10s on the first note".  We trigger every
        // lazily-constructed session here so it lands during voicebank load instead.
        //
        // (b) we deliberately do NOT pre-run.  Diffsinger3rdApi.Util.Onnx.VerifyInputNames
        // throws on any extra/missing input name — and the SSM acoustic graph's input set
        // varies based on dsconfig flags (useVariableDepth, useContinuousAcceleration,
        // use_lang_id, useSpeedEmbed, etc.).  Trying to enumerate exactly the right set
        // for a dummy Run is a maintenance hazard relative to ~1s of avoided cost.
        //
        // Each call is independently try/catch'd: a missing dspitch/ dsvariance/ dsvocoder
        // is fine (some models don't ship them) and must not block voicebank load.
        SafeRun(() => _ = _singer.getAcousticSession());
        SafeRun(() => _ = _singer.getVocoder());
        SafeRun(() => _ = _singer.getPitchPredictor());
        SafeRun(() => _ = _singer.getVariancePredictor());
    }

    private static void SafeRun(System.Action a)
    {
        try { a(); }
        catch
        {
            // intentionally swallowed
        }
    }
}