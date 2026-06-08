"""Smoke-test the freshly exported var ONNX models at multiple sequence
lengths.  Goal: confirm SSM dense path is happy with dynamic L (no more
mamba_rev/Reshape_11 crash).

Runs on the cloud env that has the diff_onnx environment.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

BASE = "/root/autodl-tmp/Diffsinger-main-SSM/onnx_out_var_testssm2"


def _make_inputs(sess: ort.InferenceSession, L: int, vocab: int):
    """Build a dummy input dict matching whatever the session expects."""
    tokens    = np.random.randint(1, vocab, size=(1, L), dtype=np.int64)
    languages = np.zeros((1, L), dtype=np.int64)
    # word_div  = how many phonemes belong to the same word (use 1-per-token here)
    word_div  = np.ones((1, L), dtype=np.int64)
    # word_dur  = duration per word in frames; 200 ≈ 2.3s @ hop=512/44.1k
    word_dur  = (np.ones((1, L), dtype=np.int64) * 50)
    # ph_dur    = phoneme dur in frames (only for non-linguistic)
    ph_dur    = np.full((1, L), 50, dtype=np.int64)
    # note_midi etc. used by pitch/dur predictors
    note_midi = np.full((1, L), 60.0, dtype=np.float32)
    note_rest = np.zeros((1, L), dtype=bool)
    note_dur  = np.full((1, L), 50, dtype=np.int64)
    pitch     = np.full((1, L * 50), 60.0, dtype=np.float32)
    expr      = np.ones((1, L * 50), dtype=np.float32)
    retake    = np.ones((1, L * 50), dtype=bool)
    spk_embed = np.zeros((1, L, 256), dtype=np.float32)
    encoder_out = np.zeros((1, L, 256), dtype=np.float32)
    steps     = np.array([20], dtype=np.int64)

    pool = {
        "tokens": tokens, "languages": languages, "word_div": word_div,
        "word_dur": word_dur, "ph_dur": ph_dur,
        "note_midi": note_midi, "note_rest": note_rest, "note_dur": note_dur,
        "pitch": pitch, "expr": expr, "retake": retake,
        "spk_embed": spk_embed, "encoder_out": encoder_out,
        "steps": steps,
    }
    return {i.name: pool[i.name] for i in sess.get_inputs() if i.name in pool}


def main():
    onnx_files = [
        f"{BASE}/var_testssm2.linguistic.onnx",
        f"{BASE}/var_testssm2.dur.onnx",
        f"{BASE}/var_testssm2.pitch.onnx",
        f"{BASE}/var_testssm2.variance.onnx",
    ]

    for path in onnx_files:
        print(f"\n=== {os.path.basename(path)} ===")
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for inp in sess.get_inputs():
            print(f"  in : {inp.name:18s} {inp.shape} {inp.type}")
        for out in sess.get_outputs():
            print(f"  out: {out.name:18s} {out.shape} {out.type}")

        # phonemes vocab
        with open(f"{BASE}/var_testssm2.phonemes.json") as f:
            ph = json.load(f)
        vocab = max(ph.values()) + 1

        for L in (16, 80, 256):
            try:
                inputs = _make_inputs(sess, L, vocab)
                missing = [i.name for i in sess.get_inputs() if i.name not in inputs]
                if missing:
                    print(f"  L={L:3d}: SKIP (need inputs we didn't pre-build: {missing})")
                    continue
                out = sess.run(None, inputs)
                print(f"  L={L:3d}: OK, output shapes = {[o.shape for o in out]}")
            except Exception as e:
                print(f"  L={L:3d}: FAIL — {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()