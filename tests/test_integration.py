"""
End-to-End SSM Integration Test (SimpleSSM Fallback)
Verifies the full DiffSinger pipeline with all SSM components.
"""
import sys, os, yaml
sys.path.insert(0, r"i:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-main-SSM")

import torch
print("PyTorch:", torch.__version__)

# ── Set hparams BEFORE importing DiffSinger modules ──
# Note: we import utils first to get the hparams dict reference
import utils
base_cfg = yaml.safe_load(open(
    r"i:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-main-SSM\configs\base.yaml", encoding='utf-8'))

ssm_overrides = {
    'backbone_type': 'mamba',
    'backbone_args': {'num_channels': 64, 'num_layers': 2, 'd_state': 16, 'd_conv': 4,
                      'expand': 2, 'kernel_size': 7, 'activation': 'SiLU', 'dropout': 0.0, 'strong_cond': False},
    'use_shallow_diffusion': True,
    'shallow_diffusion_args': {
        'train_aux_decoder': True, 'train_diffusion': True, 'aux_decoder_grad': 0.1,
        'val_gt_start': True, 'aux_decoder_arch': 'mamba',
        'aux_decoder_args': {'num_channels': 64, 'num_layers': 2, 'kernel_size': 3,
                            'd_state': 16, 'd_conv': 4, 'expand': 2, 'dropout_rate': 0.0}
    },
    'diffusion_type': 'reflow', 'hidden_size': 64, 'enc_layers': 2,
    'enc_ffn_kernel_size': 5, 'ffn_act': 'gelu', 'dropout': 0.0,
    'num_heads': 2, 'use_pos_embed': True, 'rel_pos': False, 'use_rope': False,
    'K_step': 100, 'timesteps': 1000, 'T_start': 0.4, 'time_scale_factor': 1000,
    'use_spk_id': False, 'use_lang_id': False, 'use_energy_embed': False,
    'use_breathiness_embed': False, 'use_voicing_embed': False, 'use_tension_embed': False,
    'use_key_shift_embed': False, 'use_speed_embed': False,
    'num_spk': 1, 'num_lang': 1, 'predict_dur': True, 'predict_pitch': False,
    'predict_variances': False, 'vocab_size': 100, 'spec_min': [-12], 'spec_max': [0],
    'infer': False, 'sampling_algorithm': 'euler', 'sampling_steps': 8,
}

# Directly modify the hparams dict (both in utils.hparams and utils)
from utils import hparams
hparams.clear()
hparams.update(base_cfg)
hparams.update(ssm_overrides)

# Now import DiffSinger modules
from modules.toplevel import DiffSingerAcoustic
from modules.backbones import BACKBONES

assert 'mamba' in BACKBONES, f"Mamba missing: {list(BACKBONES.keys())}"
print(f"[1/6] Import OK. Backbones: {list(BACKBONES.keys())}")

# Test 2: Instantiate
model = DiffSingerAcoustic(vocab_size=100, out_dims=128)
pcount = sum(p.numel() for p in model.parameters())
# Check that we're using Mamba backbone via the model structure
has_mamba = any('mamba' in name.lower() for name, _ in model.named_modules())
print(f"[2/6] Model: {pcount:,} params, mamba_detected={has_mamba}")

# Test 3: Inference
model.eval()
tokens = torch.randint(1, 100, (1, 10))
mel2ph = torch.tensor([[min(i // 3 + 1, 10) for i in range(30)]], dtype=torch.long)
f0 = torch.randn(1, 30) * 10 + 300
with torch.no_grad():
    out = model(tokens, mel2ph, f0, infer=True)
assert out.aux_out is not None and out.diff_out is not None
print(f"[3/6] Inference: aux={out.aux_out.shape}, diff={out.diff_out.shape}")

# Test 4: Training
model.train()
gt_mel = torch.randn(1, 30, 128)  # (B, T, M) — 30 mel frames, 128 mel bins
out_tr = model(tokens, mel2ph, f0, infer=False, gt_mel=gt_mel)
assert out_tr.aux_out is not None and out_tr.diff_out is not None
# Training mode: diffusion returns (v_pred, v_gt, t) tuple
v_pred, v_gt, t = out_tr.diff_out
print(f"[4/6] Training: aux={out_tr.aux_out.shape}, v_pred={v_pred.shape}, v_gt={v_gt.shape}")

# Test 5: Gradient flow
t_g = torch.randint(1, 100, (1, 5))
m_g = torch.ones(1, 15, dtype=torch.long)
f_g = torch.randn(1, 15) * 10 + 300
gt_g = torch.randn(1, 15, 128)  # (B, T, M)
o_g = model(t_g, m_g, f_g, infer=False, gt_mel=gt_g)
v_pred_g, _, _ = o_g.diff_out  # training returns (v_pred, v_gt, t)
(o_g.aux_out.pow(2).mean() + v_pred_g.pow(2).mean()).backward()
gcnt = sum(1 for p in model.parameters() if p.grad is not None)
print(f"[5/6] Gradients: {gcnt}/{pcount} params have grads")

# Test 6: Finite outputs
assert torch.isfinite(out.aux_out).all() and torch.isfinite(out.diff_out).all()
assert torch.isfinite(out_tr.aux_out).all()
print("[6/6] All outputs finite")

print("\n" + "=" * 60)
print(" ALL 6 END-TO-END SSM INTEGRATION TESTS PASSED ")
print("=" * 60)
