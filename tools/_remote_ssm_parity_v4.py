"""Stage v4: instrument ref.forward to capture intermediates and compare with manual rebuild."""
from __future__ import annotations
import os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

PROJ = Path("/root/autodl-tmp/Diffsinger-main-SSM").resolve()
sys.path.insert(0, str(PROJ))
os.environ["DIFFSINGER_USE_MAMBA3"] = "0"

from einops import rearrange
from mamba_ssm import Mamba3
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


def main() -> int:
    torch.manual_seed(0)
    cfg = dict(d_model=256, d_state=128, expand=2, headdim=64, ngroups=1)
    device = torch.device("cuda")
    dtype = torch.float32

    ref = Mamba3(**cfg).to(device=device, dtype=dtype).eval()

    B_, L_ = 1, 64
    x = torch.randn(B_, L_, cfg["d_model"], device=device, dtype=dtype)

    captured = {}
    orig_kernel = mamba3_siso_combined
    import mamba_ssm.modules.mamba3 as m3mod

    def spy_kernel(*args, **kw):
        for name in ["Q","K","V","ADT","DT","Trap","Q_bias","K_bias","Angles","D","Z"]:
            v = kw.get(name)
            captured[name] = v.detach().clone() if isinstance(v, torch.Tensor) else v
        captured["chunk_size"] = kw.get("chunk_size", 64)
        out = orig_kernel(*args, **kw)
        captured["kernel_out"] = (out if not isinstance(out, tuple) else out[0]).detach().clone()
        return out

    m3mod.mamba3_siso_combined = spy_kernel

    with torch.no_grad():
        y_ref = ref(x)

    print(f"y_ref      |.| mean = {y_ref.abs().mean().item():.4e}")
    print(f"kernel_out |.| mean = {captured['kernel_out'].abs().mean().item():.4e}")

    # Now rebuild manually from the same x and compare each intermediate.
    with torch.no_grad():
        zxBCdtAtrap = ref.in_proj(x)
        z, xv, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                ref.d_inner, ref.d_inner,
                ref.d_state * ref.num_bc_heads * ref.mimo_rank,
                ref.d_state * ref.num_bc_heads * ref.mimo_rank,
                ref.nheads, ref.nheads, ref.nheads, ref.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=ref.headdim)
        xv = rearrange(xv, "b l (h p) -> b l h p", p=ref.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=ref.mimo_rank, g=ref.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=ref.mimo_rank, g=ref.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")
        _A = -F.softplus(dd_A.float()).clamp(max=-ref.A_floor)
        DT = F.softplus(dd_dt + ref.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")
        angles_full = angles.unsqueeze(-2).expand(-1, -1, ref.nheads, -1).float()
        B_n = ref.B_norm(B)
        C_n = ref.C_norm(C)

    # extra ADT debug
    _A_manual = -F.softplus(dd_A.float())
    print()
    print("ADT debug:")
    print(f"  ref.A_floor = {ref.A_floor}")
    print(f"  dd_A pre-softplus  abs mean = {dd_A.abs().mean().item():.4e}, min={dd_A.min().item():.4e}, max={dd_A.max().item():.4e}")
    print(f"  -softplus(dd_A)   abs mean = {_A_manual.abs().mean().item():.4e}, min={_A_manual.min().item():.4e}, max={_A_manual.max().item():.4e}")
    _A_clamped = _A_manual.clamp(max=-ref.A_floor)
    print(f"  after clamp(max=-A_floor) abs mean = {_A_clamped.abs().mean().item():.4e}, min={_A_clamped.min().item():.4e}, max={_A_clamped.max().item():.4e}")
    print(f"  ADT (manual)       abs mean = {ADT.abs().mean().item():.4e}")
    # what kernel actually got
    print(f"  ADT (captured/ref) abs mean = {captured['ADT'].abs().mean().item():.4e}")
    print(f"  captured ADT permuted back = {rearrange(captured['ADT'], 'b n l -> b l n').shape}")
    # element-wise compare
    cap_back = rearrange(captured['ADT'], 'b n l -> b l n')
    print(f"  manual ADT shape (b l n): {ADT.permute(0,2,1).shape}")
    pre_perm_diff = (cap_back - ADT.permute(0,2,1)).abs()
    print(f"  diff abs mean: {pre_perm_diff.mean().item():.4e}")
    print(f"  diff abs max:  {pre_perm_diff.max().item():.4e}")

    def cmp(name, a, b):
        a, b = a.float(), b.float()
        d = (a - b).abs()
        rel = d.norm() / (a.norm() + 1e-12)
        eq = "OK" if rel < 1e-4 else "DIFF"
        print(f"  [{eq}] {name:14s} ref|.|={a.abs().mean().item():.3e}  manual|.|={b.abs().mean().item():.3e}  rel={rel.item():.3e}  max={d.max().item():.3e}")

    print()
    print("intermediate comparison (kernel-input snapshots vs manual reconstruction):")
    cmp("Q",      captured["Q"],      C_n.squeeze(2))
    cmp("K",      captured["K"],      B_n.squeeze(2))
    cmp("V",      captured["V"],      xv)
    cmp("ADT",    captured["ADT"],    ADT)
    cmp("DT",     captured["DT"],     DT)
    cmp("Trap",   captured["Trap"],   trap)
    cmp("Q_bias", captured["Q_bias"], ref.C_bias.squeeze(1))
    cmp("K_bias", captured["K_bias"], ref.B_bias.squeeze(1))
    cmp("Angles", captured["Angles"], angles_full)
    cmp("D",      captured["D"],      ref.D)
    cmp("Z",      captured["Z"],      z)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())