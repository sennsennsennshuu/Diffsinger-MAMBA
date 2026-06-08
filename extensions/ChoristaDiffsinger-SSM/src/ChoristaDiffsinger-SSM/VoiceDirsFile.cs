using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace DiffsingerSSM;

/// <summary>
/// Reads the user-editable list of extra voicebank directories.
///
/// File format:
///   - one absolute directory per line
///   - leading/trailing whitespace is trimmed
///   - blank lines are ignored
///   - lines starting with '#' are comments
///   - non-existent directories are silently dropped (the user may have unplugged a USB drive)
/// </summary>
internal static class VoiceDirsFile
{
    public static IEnumerable<string> Parse(string path)
    {
        if (!File.Exists(path))
            yield break;

        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw.Trim();
            if (line.Length == 0) continue;
            if (line.StartsWith("#")) continue;
            if (!Directory.Exists(line)) continue;
            yield return line;
        }
    }
}