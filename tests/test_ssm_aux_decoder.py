"""SSM Auxiliary Decoder Test — Verifies MambaAuxDecoder interface."""
import sys
sys.path.insert(0, r"i:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-main-SSM")

import torch
from modules.aux_decoder.mamba_aux_decoder import MambaAuxDecoder

print("PyTorch:", torch.__version__)

# ── Test 1: Basic shape ──
decoder = MambaAuxDecoder(
    in_dims=256, out_dims=128,  # hidden_size=256 → 128 mel bins
    num_channels=256, num_layers=3, kernel_size=7,
    d_state=16, d_conv=4, expand=2, dropout_rate=0.0
)
x = torch.randn(2, 100, 256)  # (B=2, T=100, H=256)
out = decoder(x)
assert out.shape == (2, 100, 128), f"MambaAuxDecoder shape: {out.shape}"
print(f"[PASS] MambaAuxDecoder shape: {out.shape}")

# ── Test 2: Short sequence ──
x_short = torch.randn(1, 10, 256)
out_short = decoder(x_short)
assert out_short.shape == (1, 10, 128)
print(f"[PASS] MambaAuxDecoder short seq: {out_short.shape}")

# ── Test 3: infer flag (kept for compat) ──
out_infer = decoder(x, infer=True)
assert out_infer.shape == (2, 100, 128)
print(f"[PASS] MambaAuxDecoder infer flag: PASS")

# ── Test 4: Output is finite ──
assert torch.isfinite(out).all(), "Output contains NaN/Inf!"
print(f"[PASS] MambaAuxDecoder finite output: PASS")

# ── Test 5: ConvNeXt-compatible interface ──
from modules.aux_decoder import build_aux_decoder
aux_dec = build_aux_decoder(
    in_dims=256, out_dims=128,
    aux_decoder_arch='mamba',
    aux_decoder_args={
        'num_channels': 256, 'num_layers': 2,
        'kernel_size': 7, 'd_state': 16, 'd_conv': 4,
        'expand': 2, 'dropout_rate': 0.0
    }
)
out_reg = aux_dec(x)
assert out_reg.shape == (2, 100, 128), f"Registered decoder shape: {out_reg.shape}"
print(f"[PASS] Registered via build_aux_decoder(): {out_reg.shape}")

# ── Test 6: Gradient flow ──
decoder_grad = MambaAuxDecoder(
    in_dims=64, out_dims=32, num_channels=64, num_layers=2,
    kernel_size=3, d_state=16, dropout_rate=0.0
)
x_grad = torch.randn(1, 20, 64, requires_grad=True)
out_grad = decoder_grad(x_grad)
loss = out_grad.sum()
loss.backward()
assert x_grad.grad is not None, "No gradient flowing back!"
assert torch.isfinite(x_grad.grad).all(), "Gradient contains NaN/Inf!"
print(f"[PASS] MambaAuxDecoder gradient flow: grad_norm={x_grad.grad.norm().item():.4f}")

print("\n" + "=" * 50)
print(" ALL 6 MambaAuxDecoder TESTS PASSED ")
print("=" * 50)
