using System;
using System.Collections.Generic;

namespace DiffsingerSSM;

/// <summary>
/// Decision logic for picking the phonemizer at synthesis time.
///
/// Priority (high to low):
///   1. The user explicitly set PartProperties[PhonemizerID] to a non-"Default" value
///      that the registry knows about — use it.
///   2. The singer's default_phonemizer (from character.yaml) is in the registry — use it.
///   3. Fall back to the generic <c>DiffSingerPhonemizer</c>, which the registry is required
///      to contain.  If somehow it doesn't, this method throws — that's a packaging bug,
///      not a runtime decision.
/// </summary>
internal static class PhonemizerSelector
{
    private const string Generic = "DiffSingerPhonemizer";
    private const string DefaultMarker = "Default";

    public static string Select(
        string userChoice,
        string singerPreferred,
        IReadOnlyDictionary<string, Type> registry)
    {
        if (registry == null) throw new ArgumentNullException(nameof(registry));

        // "Default" is the convention TuneLab puts in the dropdown to mean "use whatever the
        // singer asked for".  Anything else is a deliberate user choice — even if the choice
        // turns out to be unknown to us, we don't sneak the singer's preference back in:
        // we fall straight through to Generic.  This matches ChoristaDiffsinger's behavior
        // and keeps the rule predictable from the user's POV.
        if (string.Equals(userChoice, DefaultMarker, StringComparison.Ordinal))
        {
            if (!string.IsNullOrEmpty(singerPreferred) && registry.ContainsKey(singerPreferred))
                return singerPreferred;
            return RequireGeneric(registry);
        }

        if (registry.ContainsKey(userChoice))
            return userChoice;

        return RequireGeneric(registry);
    }

    private static string RequireGeneric(IReadOnlyDictionary<string, Type> registry)
    {
        if (!registry.ContainsKey(Generic))
        {
            throw new InvalidOperationException(
                $"Phonemizer registry is missing required key '{Generic}'.");
        }
        return Generic;
    }
}