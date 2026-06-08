"""Profile the deployed voicebank ONNX stages on the local box.

Models and their real input contracts (from tools/dump_onnx_io.py):

  linguistic : tokens(n_tokens), word_div(n_words), word_dur(n_words), languages(n_tokens)
  dur        : encoder_out(n_tokens,256), x_masks(n_tokens), ph_midi(n_tokens), spk_embed(n_tokens,256)
  pitch      : encoder_out(n_tokens,256), ph_dur(n_tokens), note_midi(n_notes), note_rest(n_notes),
               note_dur(n_notes), pitch(n_frames), expr(n_frames), retake(n_frames),
               spk_embed(n_frames,256), steps(1,)
  variance   : encoder_out(n_tokens,256), ph_dur(n_tokens), pitch(n_frames), breathiness(n_frames),
               voicing(n_frames), tension(n_frames), retake(n_frames,3), spk_embed(n_frames,256),
               steps(1,)
  acoustic   : tokens(n_tokens), durations(n_tokens), f0(n_frames), breathiness(n_frames),
               voicing(n_frames), tension(n_frames), gender(n_frames), velocity(n_frames),
               spk_embed(n_frames,256), languages(n_tokens), depth(scalar), steps(scalar)
  vocoder    : mel(n_frames,128), f0(n_frames)

Phrase chosen: 96 phonemes, 24 notes / words, 258 frames (~3 s @ 44.1k/hop=512).
ph_dur is constructed so it sums exactly to n_frames.

Each session is benched with 1 warm-up + 5 measured runs, median reported.
Provider defaults to CPU but `--gpu` switches to CUDAExecutionProvider when available.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import median

import numpy as np
import onnxruntime as ort


BANK = Path(r"D:\OpenUtau for diffsinger\Singers\SSM_test_opencpop")

ACO_PATH      = BANK / "aco_testssm2.onnx"
VOCODER_PATH  = BANK / "dsvocoder" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
LING_PATH     = BANK / "dspitch" / "var_testssm2.linguistic.onnx"
DUR_PATH      = BANK / "dsdur"   / "var_testssm2.dur.onnx"
PITCH_PATH    = BANK / "dspitch" / "var_testssm2.pitch.onnx"
VAR_PATH      = BANK / "dsvariance" / "var_testssm2.variance.onnx"


N_TOKENS  = 96
N_WORDS   = 24
N_NOTES   = 24
N_FRAMES  = 258
HIDDEN    = 256
N_MEL     = 128
DIFF_STEPS = 20


def _build_ph_dur(n_tokens: int, n_frames: int) -> np.ndarray:
    base = n_frames // n_tokens
    rem = n_frames - base * n_tokens
    arr = np.full((n_tokens,), base, dtype=np.int64)
    arr[:rem] += 1
    return arr.reshape(1, n_tokens)


def _bench(sess: ort.InferenceSession, feeds: dict, n: int = 5) -> float:
    sess.run(None, feeds)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)
    return median(times)


def _make_session(path: Path, provider: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=[provider])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="run on CUDAExecutionProvider")
    args = ap.parse_args()

    provider = "CUDAExecutionProvider" if args.gpu else "CPUExecutionProvider"
    available = ort.get_available_providers()
    if args.gpu and provider not in available:
        print(f"[warn] CUDAExecutionProvider not available, falling back to CPU. avail={available}")
        provider = "CPUExecutionProvider"
    print(f"Provider: {provider}")
    print(f"Voicebank: {BANK}")
    print(f"Phrase: tokens={N_TOKENS}  words={N_WORDS}  notes={N_NOTES}  frames={N_FRAMES} (~3s)")
    print()

    ph_dur = _build_ph_dur(N_TOKENS, N_FRAMES)
    word_dur = _build_ph_dur(N_WORDS, N_FRAMES)
    note_dur = _build_ph_dur(N_NOTES, N_FRAMES)
    word_div = np.full((1, N_WORDS), N_TOKENS // N_WORDS, dtype=np.int64)
    word_div[0, : (N_TOKENS - (N_TOKENS // N_WORDS) * N_WORDS)] += 1

    def zeros(shape, dtype): return np.zeros(shape, dtype=dtype)

    tokens     = np.full((1, N_TOKENS), 1,    dtype=np.int64)
    languages  = zeros((1, N_TOKENS), np.int64)
    ph_midi    = np.full((1, N_TOKENS), 60,   dtype=np.int64)
    spk_tok    = zeros((1, N_TOKENS, HIDDEN), np.float32)
    spk_frm    = zeros((1, N_FRAMES, HIDDEN), np.float32)
    enc_out    = zeros((1, N_TOKENS, HIDDEN), np.float32)
    x_masks    = zeros((1, N_TOKENS), np.bool_)
    note_midi  = np.full((1, N_NOTES), 60.0,  dtype=np.float32)
    note_rest  = zeros((1, N_NOTES), np.bool_)
    pitch_frm  = np.full((1, N_FRAMES), 60.0, dtype=np.float32)
    expr_frm   = np.ones((1, N_FRAMES), dtype=np.float32)
    retake_1d  = np.ones((1, N_FRAMES), dtype=np.bool_)
    retake_3d  = np.ones((1, N_FRAMES, 3), dtype=np.bool_)
    f0         = np.full((1, N_FRAMES), 220.0, dtype=np.float32)
    breath     = zeros((1, N_FRAMES), np.float32)
    voicing    = zeros((1, N_FRAMES), np.float32)
    tension    = zeros((1, N_FRAMES), np.float32)
    gender     = zeros((1, N_FRAMES), np.float32)
    velocity   = np.ones((1, N_FRAMES), dtype=np.float32)
    mel        = zeros((1, N_FRAMES, N_MEL), np.float32)
    steps_1    = np.array([DIFF_STEPS], dtype=np.int64)
    steps_0    = np.array(DIFF_STEPS, dtype=np.int64)
    depth_0    = np.array(0.6, dtype=np.float32)

    rows = []

    # linguistic
    sess = _make_session(LING_PATH, provider)
    feeds = {"tokens": tokens, "word_div": word_div, "word_dur": word_dur, "languages": languages}
    rows.append(("linguistic", _bench(sess, feeds)))
    enc_out_real = sess.run(None, feeds)[0]   # 用真实 encoder_out 喂下游，避免下游模型遇到全零卡死

    # dur
    sess = _make_session(DUR_PATH, provider)
    feeds = {"encoder_out": enc_out_real, "x_masks": x_masks, "ph_midi": ph_midi, "spk_embed": spk_tok}
    rows.append(("dur", _bench(sess, feeds)))

    # pitch
    sess = _make_session(PITCH_PATH, provider)
    feeds = {
        "encoder_out": enc_out_real, "ph_dur": ph_dur,
        "note_midi": note_midi, "note_rest": note_rest, "note_dur": note_dur,
        "pitch": pitch_frm, "expr": expr_frm, "retake": retake_1d,
        "spk_embed": spk_frm, "steps": steps_1,
    }
    rows.append(("pitch", _bench(sess, feeds)))

    # variance
    sess = _make_session(VAR_PATH, provider)
    feeds = {
        "encoder_out": enc_out_real, "ph_dur": ph_dur,
        "pitch": pitch_frm, "breathiness": breath, "voicing": voicing, "tension": tension,
        "retake": retake_3d, "spk_embed": spk_frm, "steps": steps_1,
    }
    rows.append(("variance", _bench(sess, feeds)))

    # acoustic
    sess = _make_session(ACO_PATH, provider)
    feeds = {
        "tokens": tokens, "durations": ph_dur, "f0": f0,
        "breathiness": breath, "voicing": voicing, "tension": tension,
        "gender": gender, "velocity": velocity,
        "spk_embed": spk_frm, "languages": languages,
        "depth": depth_0, "steps": steps_0,
    }
    rows.append(("acoustic", _bench(sess, feeds)))

    # vocoder
    sess = _make_session(VOCODER_PATH, provider)
    feeds = {"mel": mel, "f0": f0}
    rows.append(("vocoder", _bench(sess, feeds)))

    total = sum(ms for _, ms in rows)
    audio_sec = N_FRAMES * 512 / 44100.0

    print(f"{'stage':14s} {'wall ms':>10s} {'%total':>8s}")
    print("-" * 36)
    for name, ms in rows:
        print(f"{name:14s} {ms:10.1f} {ms/total*100:7.1f}%")
    print("-" * 36)
    print(f"{'TOTAL':14s} {total:10.1f}")
    print(f"audio: {audio_sec:.2f}s    RTF (lower=faster) = {total/1000.0/audio_sec:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())