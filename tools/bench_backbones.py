"""Benchmark: lynxnet 6L x 1024 vs mamba_backbone 12L x 512 single forward on ORT CPU.

Goal: quantify whether swapping aco backbone from LYNXNet to MambaBackbone
would buy throughput on the SAME ORT CPU runtime that the deployed voicebank uses.

Method:
  1. Build each backbone with random weights (no ckpt needed: we measure compute, not quality).
  2. Trace-export each to ONNX (DIFFSINGER_USE_MAMBA3=0 for SimpleSSM dense path, matching
     the deployed export pipeline).
  3. Run on ORT CPU (1 warmup + 5 measured), median per-call ms across T in {128,256,512,1024}.

Inputs match the production aco diffusion sub-graph contract:
  spec: [1, 1, 128, T]
  diffusion_step: [1, 1]
  cond: [1, 256, T]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from statistics import median

# Force SimpleSSM dense path (matches deployed export). Must be set BEFORE module import.
os.environ.setdefault("DIFFSINGER_USE_MAMBA3", "0")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.hparams import set_hparams, hparams  # noqa: E402

# Provide minimal hparams so MambaBackbone reads cond_dim=256 cleanly.
hparams.update({"hidden_size": 256})

from modules.backbones.lynxnet import LYNXNet  # noqa: E402
from modules.backbones.mamba_backbone import MambaBackbone  # noqa: E402

import onnxruntime as ort  # noqa: E402


def export_onnx(model: torch.nn.Module, T: int, out_path: Path) -> None:
    model.eval()
    spec = torch.randn(1, 1, 128, T)
    step = torch.tensor([0.5], dtype=torch.float32)
    cond = torch.randn(1, 256, T)
    torch.onnx.export(
        model,
        (spec, step, cond),
        str(out_path),
        opset_version=17,
        input_names=["spec", "step", "cond"],
        output_names=["out"],
        dynamic_axes={"spec": {3: "T"}, "cond": {2: "T"}, "out": {3: "T"}},
        do_constant_folding=True,
    )


def bench(onnx_path: Path, T: int, n_runs: int = 5) -> float:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    spec = rng.standard_normal((1, 1, 128, T), dtype=np.float32)
    step = np.array([0.5], dtype=np.float32)
    cond = rng.standard_normal((1, 256, T), dtype=np.float32)
    feed = {"spec": spec, "step": step, "cond": cond}
    sess.run(None, feed)  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return median(times)


def main():
    out_dir = ROOT / "artifacts_local" / "backbone_bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[build] LYNXNet 6L x 1024 ...")
    lynx = LYNXNet(
        in_dims=128, n_feats=1,
        num_layers=6, num_channels=1024, expansion_factor=2, kernel_size=31,
        activation="PReLU", dropout=0.0, strong_cond=True,
    )
    print(f"  params = {sum(p.numel() for p in lynx.parameters()) / 1e6:.2f} M")

    print("[build] MambaBackbone 12L x 512 ...")
    mamba = MambaBackbone(
        in_dims=128, n_feats=1,
        num_layers=12, num_channels=512,
        d_state=128, expand=2, dropout=0.0, strong_cond=True,
    )
    print(f"  params = {sum(p.numel() for p in mamba.parameters()) / 1e6:.2f} M")

    # Export once each at T=1 (dynamic axis), then bench at multiple T.
    lynx_path = out_dir / "lynxnet_6L_1024.onnx"
    mamba_path = out_dir / "mamba_12L_512.onnx"
    print(f"[export] {lynx_path}")
    export_onnx(lynx, T=64, out_path=lynx_path)
    print(f"  size = {lynx_path.stat().st_size/1e6:.1f} MB")
    print(f"[export] {mamba_path}")
    export_onnx(mamba, T=64, out_path=mamba_path)
    print(f"  size = {mamba_path.stat().st_size/1e6:.1f} MB")

    print()
    print(f"{'T':>6} | {'lynxnet ms':>12} | {'mamba ms':>10} | {'mamba/lynx':>10}")
    print("-" * 50)
    for T in (128, 256, 512, 1024):
        a = bench(lynx_path, T)
        b = bench(mamba_path, T)
        print(f"{T:>6} | {a:>12.1f} | {b:>10.1f} | {b/a:>10.2f}")


if __name__ == "__main__":
    main()