using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace DiffsingerSSM;

/// <summary>
/// Pure function that computes the ordered list of directories the SSM engine should scan
/// for voicebanks.
///
/// Why isolate this from the engine class:
///   * pure functions are TDD-friendly (no Environment access, no I/O),
///   * the same plan should be reproducible across runs and easy to inspect,
///   * if we ever expose the plan in TuneLab UI, we already have the data shape.
///
/// Order is deliberate (high-priority first):
///   1. {enginePath}/voicedb               — banks shipped inside the .tlx
///   2. {profile}/.TuneLab/ChoristaDS-SSM/voicedb
///                                         — user's SSM-only stash (no clash with Choristad)
///   3. {profile}/diffsingervbs            — historical convention
///   4. {profile}/Documents/OpenUtau/Singers
///                                         — every existing DiffSinger user dumps here
///   5. user-supplied entries from voicedirs.txt (preserved order)
///
/// Duplicate paths are deduplicated by ordinal string equality.  We don't normalize case or
/// resolve symlinks — TuneLab itself runs win-x64 only and the resolver downstream tolerates
/// missing dirs, so light-touch dedup is enough.
/// </summary>
internal static class EngineDirectories
{
    public static IReadOnlyList<string> Plan(
        string enginePath,
        string profilePath,
        IEnumerable<string> voiceDirsExtras)
    {
        var seen = new HashSet<string>(System.StringComparer.Ordinal);
        var result = new List<string>();

        void Add(string p)
        {
            if (seen.Add(p)) result.Add(p);
        }

        Add(Path.Combine(enginePath, "voicedb"));
        Add(Path.Combine(profilePath, ".TuneLab", "ChoristaDS-SSM", "voicedb"));
        Add(Path.Combine(profilePath, "diffsingervbs"));
        Add(Path.Combine(profilePath, "Documents", "OpenUtau", "Singers"));

        foreach (var extra in voiceDirsExtras)
            Add(extra);

        return result;
    }
}