using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace DiffsingerSSM.Tests;

/// <summary>
/// Voice bank discovery rules:
///   1. A directory containing dsconfig.yaml IS a voice bank (don't descend into it).
///   2. Otherwise descend up to MaxDepth levels looking for one.
///   3. Search must be deterministic (sorted) so user-visible voice list doesn't reshuffle on each launch.
///   4. Symlinks/non-existent roots must not crash the resolver.
/// These rules are taken from ChoristaDiffsinger.DiffsingerEngine.FindVB but isolated so we can
/// unit-test them without touching Diffsinger3rdApi.
/// </summary>
public class VoiceBankResolverTests : System.IDisposable
{
    private readonly string _root;

    public VoiceBankResolverTests()
    {
        _root = Path.Combine(Path.GetTempPath(),
            "ChoristaDS-SSM-tests-" + System.Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best-effort cleanup */ }
    }

    private string Bank(string relative)
    {
        var dir = Path.Combine(_root, relative);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "dsconfig.yaml"), "phonemes: x.json\n");
        return dir;
    }

    [Fact]
    public void Returns_Empty_When_Root_Missing()
    {
        var missing = Path.Combine(_root, "does-not-exist");
        var found = VoiceBankResolver.Find(missing, maxDepth: 10);
        Assert.Empty(found);
    }

    [Fact]
    public void Returns_Self_When_Root_Is_VoiceBank()
    {
        var bank = Bank("self");
        var found = VoiceBankResolver.Find(bank, maxDepth: 10).ToList();
        Assert.Single(found);
        Assert.Equal(bank, found[0]);
    }

    [Fact]
    public void Does_Not_Descend_Into_Voice_Bank()
    {
        // Edge case: a singer ships a sub-bank inside its own directory. We must treat the
        // outer dsconfig.yaml as the bank and not duplicate-list the inner one.
        var outer = Bank("outer");
        Bank("outer/inner");
        var found = VoiceBankResolver.Find(outer, maxDepth: 10).ToList();
        Assert.Single(found);
        Assert.Equal(outer, found[0]);
    }

    [Fact]
    public void Discovers_Multiple_Banks_Below_Root()
    {
        var a = Bank("alpha");
        var b = Bank("beta");
        Directory.CreateDirectory(Path.Combine(_root, "empty"));
        var found = VoiceBankResolver.Find(_root, maxDepth: 10).ToList();
        Assert.Equal(2, found.Count);
        Assert.Contains(a, found);
        Assert.Contains(b, found);
    }

    [Fact]
    public void Respects_MaxDepth()
    {
        Bank("a/b/c/deep");
        var found0 = VoiceBankResolver.Find(_root, maxDepth: 0).ToList();
        var found2 = VoiceBankResolver.Find(_root, maxDepth: 2).ToList();
        var found4 = VoiceBankResolver.Find(_root, maxDepth: 4).ToList();
        Assert.Empty(found0);
        Assert.Empty(found2);
        Assert.Single(found4);
    }

    [Fact]
    public void Result_Is_Sorted_Deterministically()
    {
        Bank("zeta");
        Bank("alpha");
        Bank("mu");
        var found = VoiceBankResolver.Find(_root, maxDepth: 5).ToList();
        var expected = found.OrderBy(p => p, System.StringComparer.Ordinal).ToList();
        Assert.Equal(expected, found);
    }
}