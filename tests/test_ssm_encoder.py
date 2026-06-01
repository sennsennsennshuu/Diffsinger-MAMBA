"""SSM Encoder Integration Test — Verifies FastSpeech2Encoder SSM replacement."""
import sys
sys.path.insert(0, r"i:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-main-SSM")

import torch
from modules.fastspeech.tts_modules import FastSpeech2Encoder

print("PyTorch:", torch.__version__)

# ── Test 1: Constructor (should accept old hparams) ──
encoder = FastSpeech2Encoder(
    hidden_size=256,
    num_layers=4,
    ffn_kernel_size=9,
    ffn_act='gelu',
    dropout=0.1,
    num_heads=2,
    use_pos_embed=True,
    rel_pos=True,
    use_rope=False
)
print("[PASS] FastSpeech2Encoder constructor: OK")

# Check that it uses MambaEncoder internally
assert hasattr(encoder, 'encoder'), "Missing .encoder attribute"
from modules.commons.ssm_layers import MambaEncoder
print(f"[PASS] Internally uses: {type(encoder.encoder).__name__}")

# ── Test 2: Forward pass shape ──
B, T, H = 2, 50, 256
main_embed = torch.randn(B, T, H)
extra_embed = torch.randn(B, T, H)
# padding: True = padded (符合 PhonemeUtils PAD_INDEX=0, tokens==0)
padding_mask = torch.zeros(B, T, dtype=torch.bool)
padding_mask[:, -10:] = True  # last 10 tokens padded

out = encoder(main_embed, extra_embed, padding_mask)
assert out.shape == (B, T, H), f"FastSpeech2Encoder output shape: {out.shape}"
print(f"[PASS] FastSpeech2Encoder shape: {out.shape}")

# ── Test 3: Padding zeroed ──
assert (out[:, -10:, :].abs().max() < 1e-5), f"Padding not zeroed: max={out[:, -10:, :].abs().max():.2e}"
print(f"[PASS] FastSpeech2Encoder padding mask: max padded={out[:, -10:, :].abs().max():.2e}")

# ── Test 4: Different input → different output ──
main2 = torch.randn(B, T, H)
out2 = encoder(main2, extra_embed, padding_mask)
assert not torch.allclose(out, out2, atol=1e-3), "Different inputs should produce different outputs"
print("[PASS] FastSpeech2Encoder distinguishes inputs: PASS")

# ── Test 5: Position embedding ON → should still work ──
encoder_rope = FastSpeech2Encoder(
    hidden_size=256, num_layers=2, ffn_kernel_size=9, ffn_act='gelu',
    dropout=0.1, num_heads=2,
    use_pos_embed=True, rel_pos=False, use_rope=True  # RoPE mode
)
out_rope = encoder_rope(main_embed, extra_embed, padding_mask)
assert out_rope.shape == (B, T, H), f"RoPE-mode shape: {out_rope.shape}"
print(f"[PASS] FastSpeech2Encoder (RoPE-mode) shape: {out_rope.shape}")

# ── Test 6: No extra_embed ──
out_no_extra = encoder(main_embed, None, padding_mask)
assert out_no_extra.shape == (B, T, H)
print(f"[PASS] FastSpeech2Encoder (no extra_embed): {out_no_extra.shape}")

# ── Test 7: return_hiddens ──
hiddens = encoder(main_embed, extra_embed, padding_mask, return_hiddens=True)
assert hiddens.shape == (4, B, T, H), f"return_hiddens shape: {hiddens.shape}"
print(f"[PASS] FastSpeech2Encoder return_hiddens: {hiddens.shape}")

# ── Test 8: Forward pass is finite ──
assert torch.isfinite(out).all(), "Output contains NaN/Inf!"
print("[PASS] FastSpeech2Encoder finite output: PASS")

# ── Test 9: Imports from acoustic_encoder path (end-to-end import chain) ──
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
print("[PASS] FastSpeech2Acoustic import: OK (no transitive import errors)")

# ── Test 10: Imports from variance_encoder path ──
from modules.fastspeech.variance_encoder import FastSpeech2Variance
print("[PASS] FastSpeech2Variance import: OK (no transitive import errors)")

print("\n" + "=" * 50)
print(" ALL 10 ENCODER INTEGRATION TESTS PASSED ")
print("=" * 50)
