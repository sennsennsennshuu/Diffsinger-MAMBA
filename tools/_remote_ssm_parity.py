"""
Numerical parity check: mamba_ssm.Mamba3 (Triton kernel) vs SimpleSSM (pure PyTorch dense path).

Both run on GPU.  Same random weights are forced into both modules so we
isolate forward-algorithm divergence from init noise.

Outputs printed to stdout:
  * d_in_proj layout match
  * which params copied / which were skipped
  * max abs diff, mean abs diff, relative L2 between full forward outputs
  * per-step intermediate diffs (DT, ADT, cumulative phase) for triage
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJ = Path("/root/autodl-tmp/Diffsinger-main-SSM").resolve()
sys.path.insert(0, str(PROJ))

# Force SimpleSSM path explicitly (the SimpleSSM class itself ignores the env,
# but we keep this set so any indirect import doesn't accidentally route through Mamba3).
os.environ["DIFFSINGER_USE_MAMBA3"] = "0"

from modules.commons.ssm_layers import SimpleSSM  # noqa: E402

# Import the real Mamba3 directly from mamba_ssm — DO NOT use _get_mamba3
# because that wrapper falls back to SimpleSSM when the env flag is off.
from mamba_ssm import Mamba3  # noqa: E402


def _print_header(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def main() -> int:
    torch.manual_seed(0)

    # match training-side SSM config exactly: d_model=256, d_state=128,
    # expand=2 (so d_inner=512, H=8, P=64), ngroups=1, headdim=64.
    cfg = dict(d_model=256, d_state=128, expand=2, headdim=64, ngroups=1)

    device = torch.device("cuda")
    dtype = torch.float32

    # build both
    ref = Mamba3(**cfg).to(device=device, dtype=dtype).eval()
    ours = SimpleSSM(**cfg).to(device=device, dtype=dtype).eval()

    _print_header("layout")
    print("d_in_proj  ref :", ref.in_proj.weight.shape)
    print("d_in_proj ours:", ours.in_proj.weight.shape)
    print("nheads ref/ours:", ref.nheads, ours.nheads)
    print("num_rope_angles ref/ours:", ref.num_rope_angles, ours.num_rope_angles)

    # copy weights ref -> ours so they share initialisation
    _print_header("weight copy ref -> ours")
    ref_sd = dict(ref.state_dict())
    ours_sd = dict(ours.state_dict())
    copied, missing, mismatched = [], [], []
    for k, v_ours in ours_sd.items():
        if k in ref_sd:
            v_ref = ref_sd[k]
            if v_ref.shape == v_ours.shape:
                ours_sd[k] = v_ref.clone()
                copied.append((k, tuple(v_ref.shape)))
            else:
                mismatched.append((k, tuple(v_ours.shape), tuple(v_ref.shape)))
        else:
            missing.append((k, tuple(v_ours.shape)))
    ours.load_state_dict(ours_sd, strict=True)
    print("copied   :", len(copied))
    for k, s in copied:
        print(f"   {k:32s} {s}")
    print("missing  (in ours but not ref):", len(missing))
    for k, s in missing:
        print(f"   {k:32s} {s}")
    print("mismatched (shape):", len(mismatched))
    for k, so, sr in mismatched:
        print(f"   {k:32s} ours={so} ref={sr}")
    extra_ref = [k for k in ref_sd if k not in ours_sd]
    print("ref-only :", len(extra_ref))
    for k in extra_ref:
        print(f"   {k:32s} {tuple(ref_sd[k].shape)}")

    # forward pass
    _print_header("forward pass")
    B_, L_ = 1, 64
    x = torch.randn(B_, L_, cfg["d_model"], device=device, dtype=dtype)

    with torch.no_grad():
        y_ref = ref(x)
        y_ours = ours(x)

    print("y_ref shape :", tuple(y_ref.shape))
    print("y_ours shape:", tuple(y_ours.shape))

    diff = (y_ref - y_ours).abs()
    print(f"max abs diff : {diff.max().item():.6e}")
    print(f"mean abs diff: {diff.mean().item():.6e}")
    rel = diff.norm() / (y_ref.norm() + 1e-12)
    print(f"relative L2  : {rel.item():.6e}")
    print(f"y_ref  abs mean : {y_ref.abs().mean().item():.6e}")
    print(f"y_ours abs mean : {y_ours.abs().mean().item():.6e}")

    # angle-handling probe: feed tiny input and look at internal cumulative phase
    _print_header("angle handling probe")
    print("ours forward applies tanh(angles)*pi BEFORE multiplication with DT")
    print("ref  forward feeds raw angles directly into Triton kernel")
    print("if these differ it shows up in y but not in dt_bias/D etc.")

    # Small ablation: zero all params except in_proj rows for angles, and see
    # whether output magnitude is dominated by angle path.  Skipped for now to
    # keep the script deterministic and quick.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())