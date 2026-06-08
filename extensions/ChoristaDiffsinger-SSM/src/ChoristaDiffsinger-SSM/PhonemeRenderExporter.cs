using System.Collections.Generic;
using System.Linq;
using Diffsinger3rdApi.IOSys;
using TuneLab.Extensions.Voices;

namespace DiffsingerSSM;

/// <summary>
/// Builds the <c>ISynthesisNote → SynthesizedPhoneme[]</c> map TuneLab needs in the
/// <see cref="SynthesisResult"/>.
///
/// Why we don't reuse Choristad's <c>DiffsingerPhonemizerLoader.ExportRendered</c>:
///   That extension method is declared <c>internal</c>, so referencing it from a separate
///   tlx triggers CS0122.  Re-implementing here is cheaper than IVT/reflection tricks and
///   gives us a place to add SSM-specific behavior later if needed.
///
/// Behavior matches Choristad's contract:
///   * each rendered phoneme's <c>StartTime/EndTime</c> are converted to absolute project
///     seconds (caller passes the phrase start time),
///   * notes that received zero phonemes (rare; happens when the phonemizer fails on a
///     specific note) get a single empty placeholder so TuneLab still has something to
///     render in its grid.
/// </summary>
internal static class PhonemeRenderExporter
{
    public static Dictionary<ISynthesisNote, SynthesizedPhoneme[]> Export(
        IReadOnlyList<ISynthesisPhoneme> phonemes,
        ISynthesisData data,
        double startTime)
    {
        var bucket = new Dictionary<ISynthesisNote, List<SynthesizedPhoneme>>();
        foreach (var ph in phonemes)
        {
            if (ph.Parent == null) continue;
            if (!bucket.TryGetValue(ph.Parent, out var list))
            {
                list = new List<SynthesizedPhoneme>();
                bucket.Add(ph.Parent, list);
            }
            list.Add(new SynthesizedPhoneme
            {
                StartTime = startTime + ph.StartTime,
                EndTime   = startTime + ph.EndTime,
                Symbol    = ph.Symbol,
            });
        }

        var result = new Dictionary<ISynthesisNote, SynthesizedPhoneme[]>();
        foreach (var kv in bucket) result[kv.Key] = kv.Value.ToArray();
        foreach (var note in data.Notes.Where(n => !bucket.ContainsKey(n)))
        {
            result[note] = new[]
            {
                new SynthesizedPhoneme { StartTime = 0.0, EndTime = 0.0, Symbol = "" },
            };
        }
        return result;
    }
}