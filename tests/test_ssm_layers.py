"""SSM Layers Unit Test - Verifies all SSM components work correctly."""
import sys
sys.path.insert(0, r"i:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-main-SSM")

import torch

print("PyTorch:", torch.__version__)

# Import SSM layers (uses SimpleSSM fallback if mamba-ssm not installed)
from modules.commons.ssm_layers import (
    BiMambaBlock, GatedMambaFFN, MambaEncoder, 
    MambaResidualBlock, MambaBackbone
)
print("SSM layers imported successfully (fallback: SimpleSSM)")

# ── Test 1: BiMambaBlock shape ──
block = BiMambaBlock(c=256, num_heads=2, dropout=0.1, kernel_size=9, act="gelu")
x = torch.randn(2, 50, 256)
out = block(x)
assert out.shape == (2, 50, 256), f"BiMambaBlock shape: {out.shape}"
print(f"[PASS] BiMambaBlock shape: {out.shape}")

# ── Test 2: BiMambaBlock padding mask ──
mask = torch.zeros(2, 50, dtype=torch.bool)
mask[:, -10:] = True
out_masked = block(x, encoder_padding_mask=mask)
assert out_masked.shape == (2, 50, 256)
assert (out_masked[:, -10:, :].abs().max() < 1e-5), f"Padding not zeroed: max={out_masked[:, -10:, :].abs().max()}"
print(f"[PASS] BiMambaBlock padding: max padded={out_masked[:, -10:, :].abs().max():.2e}")

# ── Test 3: GatedMambaFFN ──
ffn = GatedMambaFFN(hidden_size=256, filter_size=1024, kernel_size=9, dropout=0.1, act="gelu")
x_ffn = torch.randn(2, 50, 256)
out_ffn = ffn(x_ffn)
assert out_ffn.shape == (2, 50, 256), f"GatedMambaFFN shape: {out_ffn.shape}"
print(f"[PASS] GatedMambaFFN shape: {out_ffn.shape}")

# ── Test 4: MambaEncoder ──
encoder = MambaEncoder(hidden_size=256, num_layers=2, ffn_kernel_size=9, ffn_act="gelu", dropout=0.1)
out_enc = encoder(x, padding_mask=mask)
assert out_enc.shape == (2, 50, 256), f"MambaEncoder shape: {out_enc.shape}"
print(f"[PASS] MambaEncoder shape: {out_enc.shape}")

# ── Test 5: MambaEncoder return_hiddens ──
hiddens = encoder(x, padding_mask=mask, return_hiddens=True)
assert hiddens.shape == (2, 2, 50, 256), f"MambaEncoder hiddens shape: {hiddens.shape}"
print(f"[PASS] MambaEncoder hiddens: {hiddens.shape}")

# ── Test 6: MambaResidualBlock ──
res_block = MambaResidualBlock(dim=128, cond_dim=256, d_state=16)
x_c = torch.randn(2, 128, 30)  # (B, C, T) channel-first
cond = torch.randn(2, 256, 30)
diff_step = torch.randn(2, 128, 30)
out_res = res_block(x_c, cond, diff_step)
assert out_res.shape == (2, 128, 30), f"MambaResidualBlock shape: {out_res.shape}"
print(f"[PASS] MambaResidualBlock shape: {out_res.shape}")

# ── Test 7: MambaBackbone interface (matches LYNXNet) ──
backbone = MambaBackbone(in_dims=1, n_feats=1, num_layers=4, num_channels=128, d_state=16)
spec = torch.randn(2, 1, 1, 30)  # (B, F=1, M=1, T=30)
step = torch.rand(2, 1)           # (B, 1)
cond_bb = torch.randn(2, 256, 30)  # (B, H=256, T=30)
out_bb = backbone(spec, step, cond_bb)
assert out_bb.shape == spec.shape, f"MambaBackbone shape: {out_bb.shape} vs {spec.shape}"
print(f"[PASS] MambaBackbone shape: {out_bb.shape}")

# ── Test 8: MambaBackbone with mel-like dimensions (128 mel bins) ──
backbone2 = MambaBackbone(in_dims=128, n_feats=1, num_layers=2, num_channels=64, d_state=16)
spec2 = torch.randn(2, 1, 128, 20)  # (B, F=1, M=128, T=20) simulating 128 mel bins
step2 = torch.rand(2, 1)
cond_bb2 = torch.randn(2, 256, 20)
out_bb2 = backbone2(spec2, step2, cond_bb2)
assert out_bb2.shape == spec2.shape, f"MambaBackbone mel shape: {out_bb2.shape} vs {spec2.shape}"
print(f"[PASS] MambaBackbone 128-mel shape: {out_bb2.shape}")

print("\n" + "=" * 50)
print(" ALL 8 SSM LAYER TESTS PASSED ")
print("=" * 50)
