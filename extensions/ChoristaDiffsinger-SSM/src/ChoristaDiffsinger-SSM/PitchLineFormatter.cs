using System.Collections.Generic;
using TuneLab.Base.Structures;

namespace DiffsingerSSM;

/// <summary>
/// Converts the synthesized pitch curve from RenderPhrase (a SortedDictionary keyed by
/// "ms relative to phrase start") into the per-track point-list format TuneLab consumes
/// in <see cref="TuneLab.Extensions.Voices.SynthesisResult.SynthesizedPitch"/>.
///
/// The shape we return is always exactly one inner list (single pitch line); the outer
/// list-of-lists exists in TuneLab's API to support multi-line rendering, which DiffSinger
/// does not produce.
///
/// Time stamps are converted ms → seconds and offset by <paramref name="startTime"/> so the
/// pitch line aligns with absolute project time, matching what
/// <c>DiffsingerUtils.FormatPitchLines</c> does in ChoristaDsForTuneLab.
/// </summary>
internal static class PitchLineFormatter
{
    public static List<List<Point>> Format(
        SortedDictionary<double, double> pitchLines,
        double startTime = 0.0)
    {
        var inner = new List<Point>(pitchLines.Count);
        foreach (var kv in pitchLines)
        {
            inner.Add(new Point
            {
                X = startTime + kv.Key / 1000.0,
                Y = kv.Value,
            });
        }
        return new List<List<Point>> { inner };
    }
}