using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace DiffsingerSSM;

/// <summary>
/// Locates DiffSinger voicebanks under a given root directory.
///
/// Rules:
///   1. A directory containing dsconfig.yaml IS a voicebank — return it as-is, do NOT descend
///      into its subtree.  (Some banks ship sub-banks for testing; we treat the outer one as
///      authoritative.)
///   2. Otherwise descend up to <paramref name="maxDepth"/> levels.
///   3. Output is sorted with ordinal string comparison so the user-visible voice list is
///      stable across launches.
/// </summary>
internal static class VoiceBankResolver
{
    public static IEnumerable<string> Find(string root, int maxDepth)
    {
        var results = new List<string>();
        Walk(root, maxDepth, results);
        results.Sort(System.StringComparer.Ordinal);
        return results;
    }

    private static void Walk(string dir, int depthRemaining, List<string> sink)
    {
        if (!Directory.Exists(dir)) return;
        if (File.Exists(Path.Combine(dir, "dsconfig.yaml")))
        {
            sink.Add(dir);
            return; // rule 1: don't descend into a voicebank
        }
        if (depthRemaining <= 0) return;

        string[] children;
        try
        {
            children = Directory.GetDirectories(dir);
        }
        catch
        {
            // permission errors, broken symlinks, etc. — skip silently rather than crash the engine
            return;
        }

        foreach (var child in children)
        {
            Walk(child, depthRemaining - 1, sink);
        }
    }
}