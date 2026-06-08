using System;
using System.Collections.Generic;
using System.IO;
using Diffsinger3rdApi.DiffSinger;
using Diffsinger3rdApi.IOSys;
using TuneLab.Base.Structures;
using TuneLab.Extensions.Voices;

namespace DiffsingerSSM;

/// <summary>
/// SSM-flavored DiffSinger engine for TuneLab v1.6.x.
///
/// Why a separate engine type:
///   The reference ChoristaDiffsinger.tlx is a great Transformer-era engine, but for
///   Mamba3/SSM voicebanks we want differentiated behavior:
///     * dedicated voicedb search path under %USERPROFILE%/.TuneLab/ChoristaDS-SSM (no race
///       with Choristad over the same `voicedirs.txt`),
///     * cold-start warmup at voice creation time (cumsum/exp graphs are slow on first run),
///     * future hook for a custom CUDA/CPU "SSMSelectiveScan" op.
///   None of these are achievable by patching Choristad in-place because Choristad is a
///   compiled Beta dll — we own this one.
///
/// What this class is NOT:
///   It is NOT a re-implementation of <c>Diffsinger3rdApi</c>'s C# DiffSinger pipeline —
///   that pipeline (G2P, Phonemizers, vocoder bridging, dsconfig parsing) is reused via
///   the existing dlls.  This file contains only the TuneLab integration surface.
/// </summary>
[VoiceEngine("Diffsinger-SSM")]
public class DiffsingerSSMEngine : IVoiceEngine
{
    private readonly OrderedMap<string, VoiceSourceInfo> _voiceInfos = new();
    private readonly Dictionary<string, string> _voicePaths = new();

    public IReadOnlyOrderedMap<string, VoiceSourceInfo> VoiceInfos => _voiceInfos;

    public DiffsingerSSMEngine() { }

    public bool Init(string enginePath, out string? error)
    {
        error = null;
        var trace = new List<string>();
        try
        {
            trace.Add($"enginePath = {enginePath}");

            // Diffsinger3rdApi reads its own EnginePath through this static singleton.
            // It controls where it looks for vocoders/cache; we point it at our tlx folder
            // so a vocoder shipped inside the SSM .tlx (Vocoders/<name>) can be found.
            RenderPathManager.Inst.EnginePath = enginePath;

            string profile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string ssmRoot = Path.Combine(profile, ".TuneLab", "ChoristaDS-SSM");
            Directory.CreateDirectory(ssmRoot);
            trace.Add($"profile  = {profile}");
            trace.Add($"ssmRoot  = {ssmRoot}");

            string voiceDirsTxt = Path.Combine(ssmRoot, "voicedirs.txt");
            if (!File.Exists(voiceDirsTxt))
                File.WriteAllText(voiceDirsTxt, "");

            // Auto-import from Choristad's voicedirs.txt as well, so users who already set
            // up their voicebank parents in the original ChoristaDiffsinger.tlx don't need
            // to duplicate the configuration.  We never write to Choristad's file; we only
            // read it as an additional source.  Order: SSM-specific entries first
            // (highest user intent), then Choristad's entries.
            var ssmExtras       = VoiceDirsFile.Parse(voiceDirsTxt);
            var choristadExtras = VoiceDirsFile.Parse(
                Path.Combine(profile, ".TuneLab", "ChoristaDS", "voicedirs.txt"));
            var extras = new List<string>(ssmExtras);
            extras.AddRange(choristadExtras);
            trace.Add($"extras   = [{string.Join(" | ", extras)}]");

            var roots = EngineDirectories.Plan(enginePath, profile, extras);
            trace.Add("roots:");
            foreach (var r in roots) trace.Add($"  - {r} (exists={Directory.Exists(r)})");

            foreach (var root in roots)
            {
                foreach (var bank in VoiceBankResolver.Find(root, maxDepth: 10))
                {
                    trace.Add($"found bank: {bank}");
                    var singer = DiffSingerSinger.CreateSinger(bank);
                    if (singer == null)
                    {
                        trace.Add($"  -> CreateSinger returned null (dsconfig invalid?)");
                        continue;
                    }
                    if (_voicePaths.ContainsKey(singer.Id))
                    {
                        trace.Add($"  -> duplicate id, skipped");
                        continue;
                    }

                    _voicePaths[singer.Id] = singer.Location;
                    _voiceInfos.Add(singer.Id, new VoiceSourceInfo
                    {
                        Name = singer.Name,
                    });
                    trace.Add($"  -> added '{singer.Name}' (id={singer.Id})");
                }
            }
            trace.Add($"total registered voices = {_voiceInfos.Count}");

            // Pick up phonemizer extension dlls from both engine-shipped and user-managed
            // locations.  We share the file format with Choristad so users can drop the
            // same DsdurExt/RhythmizerExt into either engine.
            foreach (var dir in new[]
                     {
                         Path.Combine(enginePath, "phonemizers"),
                         Path.Combine(ssmRoot, "phonemizers"),
                     })
            {
                LoadPhonemizersFrom(dir);
            }

            WriteTrace(ssmRoot, trace, errorMessage: null);
            return true;
        }
        catch (Exception ex)
        {
            error = "Diffsinger-SSM init failed: " + ex.Message;
            try
            {
                string profile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string ssmRoot = Path.Combine(profile, ".TuneLab", "ChoristaDS-SSM");
                WriteTrace(ssmRoot, trace, errorMessage: ex.ToString());
            }
            catch { /* best effort */ }
            return false;
        }
    }

    private static void WriteTrace(string ssmRoot, IReadOnlyList<string> trace, string? errorMessage)
    {
        try
        {
            Directory.CreateDirectory(ssmRoot);
            string log = Path.Combine(ssmRoot, "init.log");
            using var sw = new StreamWriter(log, append: false);
            sw.WriteLine($"# Diffsinger-SSM init {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            foreach (var line in trace) sw.WriteLine(line);
            if (errorMessage != null)
            {
                sw.WriteLine();
                sw.WriteLine("ERROR:");
                sw.WriteLine(errorMessage);
            }
        }
        catch
        {
            // We can't surface the failure to the user (no logger available); silently
            // give up.  The engine will still report 'Init failed' via the out param.
        }
    }

    public void Destroy()
    {
        _voiceInfos.Clear();
        _voicePaths.Clear();
    }

    public IVoiceSource CreateVoiceSource(string id)
    {
        if (!_voicePaths.TryGetValue(id, out var path))
        {
            throw new InvalidOperationException(
                $"Voice id '{id}' is not registered. Did Init() complete successfully?");
        }

        var singer = DiffSingerSinger.CreateSinger(path)
            ?? throw new InvalidOperationException(
                $"Failed to create singer for voice id '{id}' at '{path}'.");

        return new DiffsingerSSMSource(singer);
    }

    private static void LoadPhonemizersFrom(string directory)
    {
        try
        {
            if (!Directory.Exists(directory)) return;
            foreach (var dll in Directory.GetFiles(directory, "*.dll", SearchOption.AllDirectories))
            {
                PhonemizerProcesser.LoadExtensionPhonemizer(dll);
            }
        }
        catch
        {
            // Swallow: a malformed phonemizer dll must not block engine startup.
        }
    }
}