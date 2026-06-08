using System;
using System.IO;

namespace DiffsingerSSM;

/// <summary>
/// Minimal RIFF/WAVE reader for the cache files written by Diffsinger3rdApi's
/// <c>WaveFileWriter.CreateWaveFile16</c>.  Those files are always
/// PCM 16-bit, mono, 44.1 kHz, but we keep the implementation general enough to
/// also handle IEEE-float-encoded WAV in case a future Diffsinger3rdApi version
/// changes its writer.
///
/// We deliberately do NOT pull in NAudio / Diffsinger3rdApi's audio library:
///   * Diffsinger3rdApi's stripped AudioLibrary.NAudio.Wave doesn't expose a
///     WaveFileReader at all,
///   * ChoristaDsForTuneLab.dll has one but it's <c>internal</c>.
/// A 60-line RIFF parser is cheaper than dragging another dll into our tlx.
///
/// Returned samples are always single-channel float in [-1.0, 1.0].
/// Multi-channel files are mixed down by averaging.
/// </summary>
internal static class WavReader
{
    public sealed class Result
    {
        public required int SampleRate { get; init; }
        public required float[] Samples { get; init; }
    }

    public static Result Read(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        if (new string(br.ReadChars(4)) != "RIFF")
            throw new InvalidDataException("Not a RIFF file: " + path);
        _ = br.ReadInt32(); // overall size
        if (new string(br.ReadChars(4)) != "WAVE")
            throw new InvalidDataException("Not a WAVE file: " + path);

        ushort audioFormat = 0;
        ushort channels = 0;
        int sampleRate = 0;
        ushort bitsPerSample = 0;
        byte[]? rawData = null;

        while (fs.Position < fs.Length)
        {
            string chunkId = new(br.ReadChars(4));
            int chunkSize = br.ReadInt32();
            if (chunkId == "fmt ")
            {
                audioFormat   = br.ReadUInt16();
                channels      = br.ReadUInt16();
                sampleRate    = br.ReadInt32();
                _             = br.ReadInt32(); // byte rate
                _             = br.ReadUInt16(); // block align
                bitsPerSample = br.ReadUInt16();
                int rest = chunkSize - 16;
                if (rest > 0) br.ReadBytes(rest);
            }
            else if (chunkId == "data")
            {
                rawData = br.ReadBytes(chunkSize);
                break;
            }
            else
            {
                br.ReadBytes(chunkSize);
            }
        }

        if (rawData == null)
            throw new InvalidDataException("Missing 'data' chunk in " + path);
        if (channels == 0 || sampleRate == 0)
            throw new InvalidDataException("Missing 'fmt ' chunk in " + path);

        return new Result
        {
            SampleRate = sampleRate,
            Samples    = Decode(rawData, audioFormat, channels, bitsPerSample),
        };
    }

    private static float[] Decode(byte[] data, ushort audioFormat, ushort channels, ushort bps)
    {
        int frameSize = channels * (bps / 8);
        if (frameSize == 0)
            throw new InvalidDataException($"Invalid wav format (channels={channels}, bps={bps}).");
        int frames = data.Length / frameSize;
        var output = new float[frames];

        switch (audioFormat)
        {
            case 1 when bps == 16:
                for (int i = 0; i < frames; i++)
                {
                    float sum = 0f;
                    for (int c = 0; c < channels; c++)
                    {
                        short s = BitConverter.ToInt16(data, i * frameSize + c * 2);
                        sum += s / 32768f;
                    }
                    output[i] = sum / channels;
                }
                break;
            case 3 when bps == 32:
                for (int i = 0; i < frames; i++)
                {
                    float sum = 0f;
                    for (int c = 0; c < channels; c++)
                        sum += BitConverter.ToSingle(data, i * frameSize + c * 4);
                    output[i] = sum / channels;
                }
                break;
            default:
                throw new InvalidDataException(
                    $"Unsupported wav format (audioFormat={audioFormat}, bps={bps}).");
        }

        return output;
    }
}