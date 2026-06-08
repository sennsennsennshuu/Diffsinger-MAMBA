"""
Stage-2 parity probe.

Strategy: bypass our SimpleSSM forward and feed the *exact same*
intermediate tensors (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z)
to BOTH:
  (a) mamba3_siso_combined  (Triton kernel, ground truth)
  (b) SimpleSSM._forward_dense + the (D + qk_diag)*V tail and silu(z) gate

If (a) ≈ (b) here, the bug is in _forward_dense or the tail.  If they
match, the earlier divergence must be in the in_proj split / RoPE prep
(Bs, Cs, B_bias, C_bias, RMSNorm, angle accumulation).
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

from modules.commons.ssm_layers import SimpleSSM  # noqa: E402
from mamba_ssm import Mamba3  # noqa: E402

from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined  # noqa: E402


def _print_h(s):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


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

    # ------------------------------------------------------------------
    # Build ref Mamba3 intermediates by hand (copy of Mamba3.forward up
    # through to the kernel call).
    # ------------------------------------------------------------------
    with torch.no_grad():
        zxBCdtAtrap = ref.in_proj(x)
        from einops import rearrange
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

        # call kernel directly
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
        # y_kernel: (B, L, H, P)
        y_kernel = rearrange(y_kernel, "b l h p -> b l (h p)").float()
        y_kernel_out = ref.out_proj(y_kernel.to(x.dtype))

    # ------------------------------------------------------------------
    # Run SimpleSSM forward on the same x.  Already verified earlier
    # that y_kernel_out ≈ ref.forward(x) (modulo bf16 noise).
    # ------------------------------------------------------------------
    with torch.no_grad():
        y_ours = ours(x)

    _print_h("kernel-direct vs SimpleSSM (same input x)")
    diff = (y_kernel_out - y_ours).abs()
    print(f"max abs : {diff.max().item():.4e}")
    print(f"mean abs: {diff.mean().item():.4e}")
    rel = diff.norm() / y_kernel_out.norm()
    print(f"rel L2  : {rel.item():.4e}")

    # ------------------------------------------------------------------
    # Now feed ours's intermediates into kernel and compare with ours.
    # SimpleSSM uses Bs/Cs that are RMS-normed B/C (no rearrange to r,g).
    # Replicate ours's prefix.
    # ------------------------------------------------------------------
    with torch.no_grad():
        zx = ours.in_proj(x)
        z, x_v, Bs, Cs, dd_dt_o, dd_A_o, trap_o, angles_o = torch.split(
            zx,
            [
                ours.d_inner, ours.d_inner,
                ours.d_state * ours.num_bc_heads * ours.mimo_rank,
                ours.d_state * ours.num_bc_heads * ours.mimo_rank,
                ours.nheads, ours.nheads, ours.nheads,
                ours.num_rope_angles,
            ],
            dim=-1,
        )
        z = z.reshape(B_, L_, ours.nheads, ours.headdim)
        V = x_v.reshape(B_, L_, ours.nheads, ours.headdim)
        Bs_n = ours.B_norm(Bs)
        Cs_n = ours.C_norm(Cs)
        H, S = ours.nheads, ours.d_state

        K_pre = Bs_n.unsqueeze(2) + ours.B_bias.reshape(H, S).view(1, 1, H, S)
        Q_pre = Cs_n.unsqueeze(2) + ours.C_bias.reshape(H, S).view(1, 1, H, S)

    print()
    print("Bs (ours) vs B normed (ref) numerical agreement:")
    diff_b = (Bs_n.view(B_, L_, H, S) - B_n.squeeze(2).expand(-1, -1, H, -1)).abs()
    print(f"  max abs: {diff_b.max().item():.4e}")
    diff_c = (Cs_n.view(B_, L_, H, S) - C_n.squeeze(2).expand(-1, -1, H, -1)).abs()
    print(f"C (similar): max abs: {diff_c.max().item():.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())