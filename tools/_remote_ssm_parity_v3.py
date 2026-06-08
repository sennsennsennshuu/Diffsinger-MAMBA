"""
Three-way parity probe: ref.forward(x) vs kernel-direct(x) vs SimpleSSM(x).

Goal: determine where the 1.21 relative-L2 discrepancy originates.

Hypothesis A: my hand-rolled kernel call is missing a step that ref.forward applies.
Hypothesis B: SimpleSSM is correct vs the kernel and ref.forward is doing something extra.
Hypothesis C: bf16 cast inside mamba3_siso_combined accidentally interacts with fp32 weights differently across paths.

This script eliminates all three by running all three paths in one process,
on the same x, with the same weights.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJ = Path("/root/autodl-tmp/Diffsinger-main-SSM").resolve()
sys.path.insert(0, str(PROJ))

os.environ["DIFFSINGER_USE_MAMBA3"] = "0"

from einops import rearrange  # noqa: E402

from modules.commons.ssm_layers import SimpleSSM  # noqa: E402
from mamba_ssm import Mamba3  # noqa: E402
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined  # noqa: E402


def diff(a, b, tag):
    d = (a - b).abs()
    rel = d.norm() / (a.norm() + 1e-12)
    print(f"  {tag:32s} max={d.max().item():.4e} mean={d.mean().item():.4e} relL2={rel.item():.4e}")


def main() -> int:
    torch.manual_seed(0)
    cfg = dict(d_model=256, d_state=128, expand=2, headdim=64, ngroups=1)
    device = torch.device("cuda")
    dtype = torch.float32

    ref = Mamba3(**cfg).to(device=device, dtype=dtype).eval()
    ours = SimpleSSM(**cfg).to(device=device, dtype=dtype).eval()
    ours.load_state_dict({k: v.clone() for k, v in ref.state_dict().items()}, strict=True)

    B_, L_ = 1, 64
    x = torch.randn(B_, L_, cfg["d_model"], device=device, dtype=dtype)

    with torch.no_grad():
        y_ref = ref(x)
        y_ours = ours(x)

        # hand-rolled kernel call (copy of ref.forward up through kernel + out_proj)
        zxBCdtAtrap = ref.in_proj(x)
        z_r, xv_r, B_r, C_r, dd_dt_r, dd_A_r, trap_r, angles_r = torch.split(
            zxBCdtAtrap,
            [
                ref.d_inner, ref.d_inner,
                ref.d_state * ref.num_bc_heads * ref.mimo_rank,
                ref.d_state * ref.num_bc_heads * ref.mimo_rank,
                ref.nheads, ref.nheads, ref.nheads,
                ref.num_rope_angles,
            ],
            dim=-1,
        )
        z_r = rearrange(z_r, "b l (h p) -> b l h p", p=ref.headdim)
        xv_r = rearrange(xv_r, "b l (h p) -> b l h p", p=ref.headdim)
        B_r = rearrange(B_r, "b l (r g n) -> b l r g n", r=ref.mimo_rank, g=ref.num_bc_heads)
        C_r = rearrange(C_r, "b l (r g n) -> b l r g n", r=ref.mimo_rank, g=ref.num_bc_heads)
        trap_r = rearrange(trap_r, "b l h -> b h l")

        _A = -F.softplus(dd_A_r.float()).clamp(max=-ref.A_floor)
        DT_r = F.softplus(dd_dt_r + ref.dt_bias)
        ADT_r = _A * DT_r
        DT_r = rearrange(DT_r, "b l n -> b n l")
        ADT_r = rearrange(ADT_r, "b l n -> b n l")
        angles_full = angles_r.unsqueeze(-2).expand(-1, -1, ref.nheads, -1).float()

        B_n = ref.B_norm(B_r)
        C_n = ref.C_norm(C_r)

        y_kernel = mamba3_siso_combined(
            Q=C_n.squeeze(2),
            K=B_n.squeeze(2),
            V=xv_r,
            ADT=ADT_r,
            DT=DT_r,
            Trap=trap_r,
            Q_bias=ref.C_bias.squeeze(1),
            K_bias=ref.B_bias.squeeze(1),
            Angles=angles_full,
            D=ref.D,
            Z=z_r,
            chunk_size=ref.chunk_size,
        )
        y_kernel = rearrange(y_kernel, "b l h p -> b l (h p)").float()
        y_kernel_out = ref.out_proj(y_kernel.to(x.dtype))

    print("Three-way comparison (lower = more aligned):")
    diff(y_ref,         y_kernel_out, "ref.forward vs kernel-direct")
    diff(y_kernel_out,  y_ours,       "kernel-direct vs SimpleSSM")
    diff(y_ref,         y_ours,       "ref.forward vs SimpleSSM")

    # also probe: are y_ref and y_kernel_out close in absolute scale?
    print()
    print(f"  |y_ref|        mean: {y_ref.abs().mean().item():.4e}")
    print(f"  |y_kernel_out| mean: {y_kernel_out.abs().mean().item():.4e}")
    print(f"  |y_ours|       mean: {y_ours.abs().mean().item():.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())