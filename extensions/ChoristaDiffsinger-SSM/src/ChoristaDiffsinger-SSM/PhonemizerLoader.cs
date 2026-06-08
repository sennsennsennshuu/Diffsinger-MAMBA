using System;
using System.Collections.Generic;
using Diffsinger3rdApi.DiffSinger;
using Diffsinger3rdApi.DiffSinger.Phonemizers.BasePhonemizers;

namespace DiffsingerSSM;

/// <summary>
/// Per-singer cache of phonemizer instances.
///
/// Why a cache: a phonemizer's <c>SetSinger</c> can be expensive (loads ONNX-backed G2P
/// models, dictionaries, etc.).  Re-creating one for every synthesis pass would charge that
/// cost on every Start() call.
///
/// Why we don't share Choristad's <c>DiffsingerPhonemizerLoader.PhonemizerCache</c>:
///   Choristad's class is <c>internal</c> and its cache is not addressable from outside that
///   assembly.  Keeping our own cache also avoids accidentally crossing engine boundaries —
///   if the user updates a phonemizer dll specifically for the SSM engine, our cache sees
///   the change without affecting the Choristad engine, and vice versa.
/// </summary>
internal static class PhonemizerLoader
{
    private static readonly Dictionary<DiffSingerSinger, Dictionary<Type, MachineLearningPhonemizer>>
        _cache = new();
    private static readonly object _gate = new();

    public static MachineLearningPhonemizer Load(DiffSingerSinger singer, Type phonemizerType)
    {
        Dictionary<Type, MachineLearningPhonemizer> perSinger;
        lock (_gate)
        {
            if (!_cache.TryGetValue(singer, out perSinger!))
            {
                perSinger = new Dictionary<Type, MachineLearningPhonemizer>();
                _cache[singer] = perSinger;
            }
        }

        lock (perSinger)
        {
            if (perSinger.TryGetValue(phonemizerType, out var cached) && cached != null)
                return cached;

            // Activator path mirrors Diffsinger3rdApi's expectation: phonemizer types have
            // a public parameterless ctor and override SetSinger.
            var instance = (MachineLearningPhonemizer?)Activator.CreateInstance(phonemizerType)
                ?? throw new InvalidOperationException(
                    $"Activator.CreateInstance returned null for {phonemizerType.FullName}.");
            instance.SetSinger(singer);
            perSinger[phonemizerType] = instance;
            return instance;
        }
    }
}