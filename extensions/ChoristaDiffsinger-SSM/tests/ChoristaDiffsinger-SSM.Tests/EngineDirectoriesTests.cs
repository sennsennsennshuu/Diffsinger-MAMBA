using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace DiffsingerSSM.Tests;

/// <summary>
/// EngineDirectories computes the prioritized list of directories the engine should scan
/// for SSM voicebanks.  Rules:
///   1. Engine install dir's "voicedb" subfolder always wins (allows shipping samples in tlx).
///   2. Profile path %USERPROFILE%/.TuneLab/ChoristaDS-SSM/voicedb is next — this is the
///      SSM-specific equivalent of Choristad's voicedb (separate so we don't fight over IDs).
///   3. %USERPROFILE%/diffsingervbs is included for compatibility with old habits.
///   4. ~/Documents/OpenUtau/Singers — most users dump their SSM tests there first.
///   5. Whatever the user added to voicedirs.txt.
/// We test only the ordering & dedup rules; actual filesystem walks live in VoiceBankResolver.
/// </summary>
public class EngineDirectoriesTests : System.IDisposable
{
    private readonly string _root;
    private readonly string _engineDir;
    private readonly string _profile;

    public EngineDirectoriesTests()
    {
        _root = Path.Combine(Path.GetTempPath(),
            "Diffsinger-SSM-engine-" + System.Guid.NewGuid().ToString("N"));
        _engineDir = Path.Combine(_root, "engine");
        _profile = Path.Combine(_root, "userprofile");
        Directory.CreateDirectory(_engineDir);
        Directory.CreateDirectory(_profile);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    [Fact]
    public void Plan_Includes_Engine_Voicedb_First()
    {
        var plan = EngineDirectories.Plan(_engineDir, _profile, voiceDirsExtras: System.Array.Empty<string>());
        Assert.Equal(Path.Combine(_engineDir, "voicedb"), plan[0]);
    }

    [Fact]
    public void Plan_Includes_Profile_Voicedb()
    {
        var plan = EngineDirectories.Plan(_engineDir, _profile, voiceDirsExtras: System.Array.Empty<string>());
        Assert.Contains(Path.Combine(_profile, ".TuneLab", "ChoristaDS-SSM", "voicedb"), plan);
    }

    [Fact]
    public void Plan_Includes_OpenUtau_Singers_Last_Of_Defaults()
    {
        var plan = EngineDirectories.Plan(_engineDir, _profile, voiceDirsExtras: System.Array.Empty<string>());
        var expected = Path.Combine(_profile, "Documents", "OpenUtau", "Singers");
        Assert.Contains(expected, plan);
    }

    [Fact]
    public void Plan_Appends_VoicedirsExtras_After_Defaults()
    {
        var extra = Path.Combine(_root, "my-banks");
        Directory.CreateDirectory(extra);
        var plan = EngineDirectories.Plan(_engineDir, _profile, voiceDirsExtras: new[] { extra });
        Assert.Equal(extra, plan.Last());
    }

    [Fact]
    public void Plan_Deduplicates_Identical_Paths()
    {
        var extra = Path.Combine(_engineDir, "voicedb"); // collide with default[0]
        var plan = EngineDirectories.Plan(_engineDir, _profile, voiceDirsExtras: new[] { extra });
        Assert.Equal(plan.Distinct().Count(), plan.Count);
    }
}