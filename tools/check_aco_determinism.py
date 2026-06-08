"""Check whether the acoustic ONNX uses non-deterministic noise (RandomNormal
without a fixed seed).  If the same input gives different mel on two runs,
then the SNR comparison across steps is meaningless — the differences are
noise re-sampling, not sampler-truncation error.
"""
from pathlib import Path
import numpy as np
import onnxruntime as ort

BANK = Path(r"D:\OpenUtau for diffsinger\Singers\SSM_test_opencpop")
ACO_PATH = BANK / "aco_testssm2.onnx"

N_TOKENS, N_FRAMES, HIDDEN = 96, 258, 256


def main() -> int:
    sess = ort.InferenceSession(str(ACO_PATH), providers=["CPUExecutionProvider"])
    ph = N_FRAMES // N_TOKENS
    ph_dur = np.full((N_TOKENS,), ph, dtype=np.int64)
    ph_dur[: N_FRAMES - ph * N_TOKENS] += 1
    feeds = {
        "tokens":     np.full((1, N_TOKENS), 1, np.int64),
        "durations":  ph_dur.reshape(1, -1),
        "f0":         np.full((1, N_FRAMES), 220.0, np.float32),
        "breathiness":np.zeros((1, N_FRAMES), np.float32),
        "voicing":    np.zeros((1, N_FRAMES), np.float32),
        "tension":    np.zeros((1, N_FRAMES), np.float32),
        "gender":     np.zeros((1, N_FRAMES), np.float32),
        "velocity":   np.ones((1, N_FRAMES), np.float32),
        "spk_embed":  np.zeros((1, N_FRAMES, HIDDEN), np.float32),
        "languages":  np.zeros((1, N_TOKENS), np.int64),
        "depth":      np.array(0.6, dtype=np.float32),
        "steps":      np.array(20, dtype=np.int64),
    }
    a = sess.run(None, feeds)[0]
    b = sess.run(None, feeds)[0]
    diff = np.abs(a - b)
    print("Same-input two-run check (steps=20):")
    print(f"  shape: {a.shape}")
    print(f"  max abs diff : {diff.max():.4f}")
    print(f"  mean abs diff: {diff.mean():.4f}")
    print(f"  → noise-determinism: {'NO (seed not pinned)' if diff.max() > 1e-3 else 'YES'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())