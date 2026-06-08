using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace DiffsingerSSM.Tests;

/// <summary>
/// voicedirs.txt is the user-editable list of "extra voicebank directories" the engine should
/// scan, located at %USERPROFILE%/.TuneLab/ChoristaDS-SSM/voicedirs.txt. We isolate parsing
/// so it stays predictable across line endings, blank lines, comments, and missing dirs.
/// </summary>
public class VoiceDirsFileTests : System.IDisposable
{
    private readonly string _root;

    public VoiceDirsFileTests()
    {
        _root = Path.Combine(Path.GetTempPath(),
            "ChoristaDS-SSM-vdirs-" + System.Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Parse_Empty_File_Yields_No_Dirs()
    {
        var file = Path.Combine(_root, "voicedirs.txt");
        File.WriteAllText(file, "");
        var dirs = VoiceDirsFile.Parse(file).ToList();
        Assert.Empty(dirs);
    }

    [Fact]
    public void Parse_Skips_Blank_Lines_And_Comments()
    {
        var dirA = Path.Combine(_root, "A");
        var dirB = Path.Combine(_root, "B");
        Directory.CreateDirectory(dirA);
        Directory.CreateDirectory(dirB);

        var file = Path.Combine(_root, "voicedirs.txt");
        File.WriteAllText(file,
            $"\n# a comment line\n{dirA}\n  \n# another comment\n{dirB}\n");
        var dirs = VoiceDirsFile.Parse(file).ToList();
        Assert.Equal(new[] { dirA, dirB }, dirs);
    }

    [Fact]
    public void Parse_Drops_NonExistent_Dirs()
    {
        var file = Path.Combine(_root, "voicedirs.txt");
        File.WriteAllText(file,
            $"{Path.Combine(_root, "A")}\n{Path.Combine(_root, "ghost")}\n");
        Directory.CreateDirectory(Path.Combine(_root, "A"));
        var dirs = VoiceDirsFile.Parse(file).ToList();
        Assert.Single(dirs);
    }

    [Fact]
    public void Parse_Trims_Whitespace_Around_Path()
    {
        var dir = Path.Combine(_root, "spaced");
        Directory.CreateDirectory(dir);
        var file = Path.Combine(_root, "voicedirs.txt");
        File.WriteAllText(file, $"   {dir}   \r\n");
        var dirs = VoiceDirsFile.Parse(file).ToList();
        Assert.Single(dirs);
        Assert.Equal(dir, dirs[0]);
    }
}