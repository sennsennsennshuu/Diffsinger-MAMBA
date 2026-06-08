using System.Collections.Generic;
using Xunit;

namespace DiffsingerSSM.Tests;

/// <summary>
/// Decision matrix for selecting a phonemizer at synthesis time.
/// Inputs:  user-set string (PartProperties), singer's preferred phonemizer name,
///          and the available phonemizer registry.
/// Outputs: the chosen phonemizer key.
/// We codify it once so the runtime path can't drift.
/// </summary>
public class PhonemizerSelectionTests
{
    private static readonly IReadOnlyDictionary<string, System.Type> Registry =
        new Dictionary<string, System.Type>
        {
            { "DiffSingerPhonemizer", typeof(object) },
            { "DiffSingerChinesePhonemizer", typeof(object) },
            { "DiffSingerJapanesePhonemizer", typeof(object) },
        };

    [Fact]
    public void User_Default_Falls_Back_To_Singer_Default()
    {
        var picked = PhonemizerSelector.Select(
            userChoice: "Default",
            singerPreferred: "DiffSingerJapanesePhonemizer",
            registry: Registry);
        Assert.Equal("DiffSingerJapanesePhonemizer", picked);
    }

    [Fact]
    public void Explicit_User_Choice_Wins_Over_Singer_Default()
    {
        var picked = PhonemizerSelector.Select(
            userChoice: "DiffSingerChinesePhonemizer",
            singerPreferred: "DiffSingerJapanesePhonemizer",
            registry: Registry);
        Assert.Equal("DiffSingerChinesePhonemizer", picked);
    }

    [Fact]
    public void Unknown_User_Choice_Falls_Back_To_Generic_DiffSingerPhonemizer()
    {
        var picked = PhonemizerSelector.Select(
            userChoice: "DoesNotExistPhonemizer",
            singerPreferred: "DiffSingerJapanesePhonemizer",
            registry: Registry);
        Assert.Equal("DiffSingerPhonemizer", picked);
    }

    [Fact]
    public void Unknown_Singer_Default_Falls_Back_To_Generic_DiffSingerPhonemizer()
    {
        var picked = PhonemizerSelector.Select(
            userChoice: "Default",
            singerPreferred: "SomethingExotic",
            registry: Registry);
        Assert.Equal("DiffSingerPhonemizer", picked);
    }
}