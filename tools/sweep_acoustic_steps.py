"""Steps-vs-quality sweep on the deployed acoustic ONNX.

Reuses the same dummy phrase as profile_voicebank.py.  For each step count
in {20, 10, 5, 2}, runs acoustic with a fixed random condition + identical
noise (np seed) and reports:

  * wall ms (median of 5 runs)
  * max abs diff vs steps=20 baseline (mel)
  * mean abs diff vs steps=20 baseline (mel)
  * SNR vs steps=20 baseline (mel, dB)

Then runs the vocoder once per steps to produce wav files
`tools/sweep_steps_<n>.wav` for ear test.

NOTE: input phoneme/f0 are dummy, so the wav won't sing real lyrics.  But
the SAME dummy input is used across steps, so any audible difference comes
solely from sampler step count.
"""
from __future__ import annotations

import time
from pathlib import Path
from statistics import median

import numpy as np
import onnxruntime as ort

try:
    from scipy.io import wavfile  # type: ignore
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


BANK = Path(r"D:\OpenUtau for diffsinger\Singers\SSM_test_opencpop")
ACO_PATH      = BANK / "aco_testssm2.onnx"
VOCODER_PATH  = BANK / "dsvocoder" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"

OUT_DIR = Path(r"I:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-main-SSM\tools\sweep_out")

N_TOKENS  = 96
N_FRAMES  = 258
HIDDEN    = 256

STEPS_LIST = [20, 10, 5, 2]


def _build_ph_dur(n_tokens: int, n_frames: int) -> np.ndarray:
    base = n_frames // n_tokens
    rem = n_frames - base * n_tokens
    arr = np.full((n_tokens,), base, dtype=np.int64)
    arr[:rem] += 1
    return arr.reshape(1, n_tokens)


def _bench(sess: ort.InferenceSession, feeds: dict, n: int = 5):
    sess.run(None, feeds)
    times = []
    last_out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)
        last_out = out
    return median(times), last_out


def _snr_db(ref: np.ndarray, sig: np.ndarray) -> float:
    noise = ref - sig
    rs = float(np.sum(ref ** 2))
    ns = float(np.sum(noise ** 2))
    if ns == 0.0:
        return float("inf")
    return 10.0 * np.log10(rs / ns)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(20251208)

    ph_dur = _build_ph_dur(N_TOKENS, N_FRAMES)
    tokens = np.full((1, N_TOKENS), 1, dtype=np.int64)
    languages = np.zeros((1, N_TOKENS), dtype=np.int64)
    f0 = np.full((1, N_FRAMES), 220.0, dtype=np.float32)
    breath = np.zeros((1, N_FRAMES), np.float32)
    voicing = np.zeros((1, N_FRAMES), np.float32)
    tension = np.zeros((1, N_FRAMES), np.float32)
    gender = np.zeros((1, N_FRAMES), np.float32)
    velocity = np.ones((1, N_FRAMES), np.float32)
    spk_frm = np.zeros((1, N_FRAMES, HIDDEN), np.float32)
    depth_0 = np.array(0.6, dtype=np.float32)

    aco = ort.InferenceSession(str(ACO_PATH), providers=["CPUExecutionProvider"])
    voc = ort.InferenceSession(str(VOCODER_PATH), providers=["CPUExecutionProvider"])

    base_feeds = {
        "tokens": tokens, "durations": ph_dur, "f0": f0,
        "breathiness": breath, "voicing": voicing, "tension": tension,
        "gender": gender, "velocity": velocity,
        "spk_embed": spk_frm, "languages": languages,
        "depth": depth_0,
    }

    print(f"{'steps':>5s} {'wall ms':>10s} {'speedup':>8s}  {'maxAbs':>9s}  {'meanAbs':>9s}  {'SNR(dB)':>8s}")
    print("-" * 60)
    ref_mel = None
    ref_ms = None
    for steps in STEPS_LIST:
        feeds = dict(base_feeds, steps=np.array(steps, dtype=np.int64))
        ms, out = _bench(aco, feeds)
        mel = out[0]

        if ref_mel is None:
            ref_mel = mel.copy()
            ref_ms = ms
            speedup = 1.0
            mx = mn = 0.0
            snr = float("inf")
        else:
            diff = np.abs(mel - ref_mel)
            mx = float(diff.max())
            mn = float(diff.mean())
            snr = _snr_db(ref_mel, mel)
            speedup = ref_ms / ms

        print(f"{steps:5d} {ms:10.1f} {speedup:7.2f}x  {mx:9.4f}  {mn:9.4f}  {snr:8.2f}")

        wav = voc.run(None, {"mel": mel.astype(np.float32), "f0": f0})[0][0]
        if HAVE_SCIPY:
            wav_int16 = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav_int16 * 32767).astype(np.int16)
            wavfile.write(str(OUT_DIR / f"sweep_steps_{steps:02d}.wav"), 44100, wav_int16)
        else:
            np.save(OUT_DIR / f"sweep_steps_{steps:02d}.npy", wav)

    out_msg = "wav" if HAVE_SCIPY else "npy"
    print(f"\nartifacts: {OUT_DIR}\\sweep_steps_<n>.{out_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())