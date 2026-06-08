using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Diffsinger3rdApi.DiffSinger;
using Diffsinger3rdApi.DiffSinger.Phonemizers;
using Diffsinger3rdApi.DiffSinger.Phonemizers.BasePhonemizers;
using Diffsinger3rdApi.IOSys;
using Diffsinger3rdApi.Util;
using TuneLab.Base.Structures;
using TuneLab.Extensions.Voices;

namespace DiffsingerSSM;

/// <summary>
/// One synthesis task for a single SSM voicebank phrase.
///
/// Lifecycle:
///   1. <see cref="Start"/> kicks off a Task.Run pipeline:
///        SetUp phonemizer → run G2P → assemble phoneme table → build RenderPhrase →
///        invoke Diffsinger3rdApi's RenderDiffsinger → decode resulting wav → fire Complete.
///   2. <see cref="Stop"/> cancels via the shared <see cref="CancellationTokenSource"/>;
///      RenderDiffsinger checks the token before each ORT.Run call.
///   3. <see cref="Suspend"/>/<see cref="Resume"/>/<see cref="SetDirty"/> are intentionally
///      no-ops, mirroring Choristad's behavior — the renderer doesn't have a
///      pause-resume primitive and TuneLab tolerates this.
///
/// Differences from Choristad's DiffsingerTask:
///   * No ToneRangeBoost (we don't ship the ChoristaEffectBoost feature in the SSM tlx —
///     it's a Choristad "advanced" feature and not relevant to the SSM speed work).
///   * No ChoristaVocoder post-process either; raw renderer output is what the user gets.
///   * Reads PartProperties via the SSM-prefixed keys to avoid clashing with Choristad's
///     PropertyObject if both engines coexist in one project.
/// </summary>
internal sealed class DiffsingerSSMTask : ISynthesisTask
{
    private readonly DiffSingerSinger _singer;
    private readonly ISynthesisData _data;
    private readonly CancellationTokenSource _cancel = new();

    public event Action<SynthesisResult>? Complete;
#pragma warning disable CS0067 // Event 'Progress' is required by ISynthesisTask but the underlying renderer doesn't surface progress.
    public event Action<double>? Progress;
#pragma warning restore CS0067
    public event Action<string>? Error;

    public DiffsingerSSMTask(DiffSingerSinger singer, ISynthesisData data)
    {
        _singer = singer ?? throw new ArgumentNullException(nameof(singer));
        _data   = data   ?? throw new ArgumentNullException(nameof(data));
    }

    public void Suspend() { }
    public void Resume()  { }
    public void SetDirty(string dirtyType) { }
    public void Stop() => _cancel.Cancel();

    public void Start()
    {
        Task.Run(RunPipeline);
    }

    private void RunPipeline()
    {
        try
        {
            string phonemizerKey = _data.PartProperties.GetString(SSMConstants.PhonemizerID, "Default");
            if (phonemizerKey == "Default") phonemizerKey = _singer.PhonemizerName;

            var phonemizerType = PhonemizerProcesser.DiffsingerPhonemizer.ContainsKey(phonemizerKey)
                ? PhonemizerProcesser.DiffsingerPhonemizer[phonemizerKey]
                : typeof(DiffSingerPhonemizer);

            var phonemizer = PhonemizerLoader.Load(_singer, phonemizerType);

            // DiffSingerBasePhonemizer carries a SingerLoaded flag; when false, the
            // phonemizer simply cannot operate on this singer (e.g., language mismatch).
            // We surface this as a soft Error rather than throwing.
            var sl = phonemizer.GetType().GetField("SingerLoaded",
                BindingFlags.Instance | BindingFlags.Public);
            if (sl != null && phonemizer is DiffSingerBasePhonemizer basic && !basic.SingerLoaded)
            {
                Error?.Invoke($"Phonemizer {phonemizerKey} cannot be used on this singer.");
                return;
            }

            List<PhonemizerProcesser.PhonemizerProcessResult> g2p;
            try
            {
                g2p = PhonemizerProcesser.Process(_data, phonemizer);
            }
            catch (Exception ex)
            {
                Error?.Invoke("Phonemizer process error: " + ex.Message);
                return;
            }

            var phonemes = PhonemizerProcesser.GeneratePhonemeTable(
                g2p, _data.StartTime(), out var leadingTime);
            if (phonemes == null || phonemes.Count == 0)
            {
                Error?.Invoke("Empty phonemes.");
                return;
            }

            double startTime = _data.StartTime() - leadingTime;
            double pitchTransitionTime = _data.PartProperties.GetDouble(
                SSMConstants.PitchTransitionTimeID,
                SSMConstants.PitchTransitionTimeConfig.DefaultValue);

            var phrase = new RenderPhrase(_singer, _data, phonemes, pitchTransitionTime);
            var pitchSnapshot = new SortedDictionary<double, double>(phrase.pitches);

            ApplyRenderPreferences(_data, _singer);

            string wavPath;
            try
            {
                wavPath = new RenderDiffsinger().Render(phrase, _cancel);
            }
            catch (Exception ex)
            {
                Error?.Invoke(ex.Message);
                return;
            }

            if (string.IsNullOrEmpty(wavPath))
            {
                // Renderer returned "" — treated as cancel/abort, no need to report.
                return;
            }

            var wav = WavReader.Read(wavPath);

            // The renderer caches .wav files relative to its phrase hash; the file's
            // start-of-audio aligns with (phrase.startTime - leadingTime - 0.1s) per
            // Choristad's convention, so we report startTime - 0.1 here.
            Complete?.Invoke(new SynthesisResult(
                startTime: startTime - 0.1,
                samplingRate: wav.SampleRate,
                audioData: wav.Samples,
                synthesizedPitch: PitchLineFormatter.Format(pitchSnapshot, startTime),
                synthesizedPhoneme: PhonemeRenderExporter.Export(phonemes, _data, startTime)));
        }
        catch (Exception ex)
        {
            Error?.Invoke("SSM task error: " + ex.Message);
        }
    }

    /// <summary>
    /// Bridge SSM-prefixed PartProperties into Diffsinger3rdApi's global preference singleton
    /// (<see cref="DiffsingerPreferences.Default"/>).  RenderDiffsinger and DsPitch/DsVariance
    /// read these statics directly with no override hook, so this is the only place we can
    /// inject per-task settings without forking the third-party renderer.
    ///
    /// Thread safety: TuneLab serialises Render calls per voicebank via <c>lock(acousticSession)</c>
    /// inside RenderDiffsinger, so a stale write here cannot leak across two concurrent tasks
    /// of the same engine instance.  The static field is overwritten on the next task anyway.
    /// </summary>
    private static void ApplyRenderPreferences(ISynthesisData data, DiffSingerSinger singer)
    {
        var steps = (int)data.PartProperties.GetDouble(
            SSMConstants.RenderStepsID,
            SSMConstants.RenderStepsConfig.DefaultValue);
        var depth = data.PartProperties.GetDouble(
            SSMConstants.RenderDepthID,
            SSMConstants.RenderDepthConfig.DefaultValue);
        var cache = data.PartProperties.GetBool(
            SSMConstants.TensorCacheID,
            SSMConstants.TensorCacheConfig.DefaultValue);
        var device = data.PartProperties.GetString(
            SSMConstants.RenderDeviceID,
            SSMConstants.RenderDeviceConfig.DefaultValue);

        DiffsingerPreferences.Default.DiffSingerSteps = Math.Clamp(steps, 1, 100);
        DiffsingerPreferences.Default.DiffSingerDepth = Math.Clamp(depth, 0.0, 1.0);
        DiffsingerPreferences.Default.DiffSingerTensorCache = cache;

        // RenderDevice → OnnxRunner.  See SSMConstants.RenderDeviceID.
        // Any unrecognised label falls back to autogpu.
        var newRunner = device?.ToLowerInvariant() switch
        {
            "cpu" => "cpu",
            _      => "autogpu",
        };
        var oldRunner = DiffsingerPreferences.Default.OnnxRunner ?? string.Empty;
        if (!string.Equals(oldRunner, newRunner, StringComparison.OrdinalIgnoreCase))
        {
            // Swap the EP first, then dispose the cached sessions so the next
            // get*Session() call rebuilds them with the new provider.  Doing it
            // in this order means a concurrent reader, if any, never sees a
            // null session paired with a stale runner string.
            DiffsingerPreferences.Default.OnnxRunner = newRunner;
            try { singer.FreeMemory(); } catch { /* ignored: singer may not yet have any session */ }
        }
    }
}