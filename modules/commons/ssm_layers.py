"""
SSM (State Space Model) layers for DiffSinger.
Replaces all Transformer components with Mamba3-based SSM variants.

Architecture:
- BiMambaBlock: Bidirectional Mamba3 encoder block (replaces EncSALayer)
- GatedMambaFFN: Gated Conv1d FFN (replaces TransformerFFNLayer)  
- MambaEncoder: Stack of BiMambaBlock (replaces FastSpeech2Encoder)
- MambaResidualBlock: SSM diffusion residual block (replaces LYNXNet/WaveNet blocks)
- MambaBackbone: Full SSM diffusion backbone (replaces LYNXNet/WaveNet)

SSM Library: mamba-ssm>=2.3.0 (state-spaces/mamba)
- Mamba3: d_state=128 (default), headdim=64, no local conv kernel
- Mamba3 is the latest generation with MIMO mode, rotary embeddings,
  and structured SSD attention — unified SSM + attention framework.

Fallback: Pure PyTorch SimpleSSM for CPU / non-CUDA environments.

References:
- https://github.com/state-spaces/mamba
- https://github.com/hustvl/Vim (bidirectional Mamba)
- https://github.com/tyshiwo1/DiM-DiffusionMamba (SSM diffusion backbone)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Mamba3 import with fallback ───────────────────────────────────────
_MAMBA3_AVAILABLE = False

try:
    from mamba_ssm import Mamba3
    _MAMBA3_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════
# Pure PyTorch SSM Fallback (CPU / non-CUDA)
# Mamba3-compatible simplified selective SSM implementation.
# ═══════════════════════════════════════════════════════════════════════

class SimpleSSM(nn.Module):
    """
    Simplified selective SSM in pure PyTorch.
    Used as fallback when mamba-ssm (Mamba3) is not available.
    
    Implements the core selective scan algorithm:
        h_t = A * h_{t-1} + B * x_t
        y_t = C * h_t + D * x_t
    
    Where A, B, C are input-dependent (selective), matching Mamba3's
    architecture pattern without MIMO or RoPE extensions.
    
    Constructor matches Mamba3's key parameters for drop-in compatibility.
    """
    def __init__(self, d_model, d_state=128, expand=2, 
                 headdim=64, ngroups=1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        
        d_inner = int(d_model * expand)
        assert d_inner % headdim == 0, f"d_inner ({d_inner}) must be divisible by headdim ({headdim})"
        self.nheads = d_inner // headdim
        self.d_inner = d_inner
        
        # Input projection (matching Mamba3's in_proj pattern)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # SSM parameters (per-head, matching Mamba3)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(self.nheads, 1))
        )
        self.D = nn.Parameter(torch.ones(self.nheads))
        
        # dt projection per head
        self.dt_proj = nn.Linear(1, self.nheads, bias=True)
        dt_init = torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        self.dt_proj.bias = nn.Parameter(dt_init)
        
        # Head-wise B, C projections
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)  where D = d_model, returns (B, L, D).
        
        GPU-friendly vectorized parallel scan using cumprod/cumsum.
        Batches over state dimensions to balance memory vs parallelism.
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)
        
        P = self.headdim
        H = self.nheads
        
        # dt: (B, L, H), clamped for stability
        dt = F.softplus(self.dt_proj(x_proj.mean(dim=-1, keepdim=True)))  # (B, L, H)
        dt = dt.clamp(max=20.0)
        
        # B, C: (B, L, d_state)
        bc = self.x_proj(x_proj)  # (B, L, 2*d_state)
        Bp, Cp = bc.chunk(2, dim=-1)  # each (B, L, d_state)
        
        A_neg = -torch.exp(self.A_log).to(x.device).reshape(self.nheads, self.d_state)  # (H, d_state)
        S = self.d_state
        
        # Reshape to (B*H, L, P) for vectorized heads
        xh = x_proj.reshape(B, L, H, P).transpose(1, 2).reshape(B * H, L, P)
        dt_h = dt.transpose(1, 2).reshape(B * H, L)  # (BH, L)
        A_h = A_neg.reshape(-1, S).repeat(B, 1)  # (BH, d_state)
        
        # Expand B, C from (B, L, S) to (BH, L, S)
        B_h = Bp.unsqueeze(1).expand(B, self.nheads, L, S).reshape(B * self.nheads, L, S).contiguous()  # (BH, L, S)
        C_h = Cp.unsqueeze(1).expand(B, self.nheads, L, S).reshape(B * self.nheads, L, S).contiguous()  # (BH, L, S)
        
        # State scan: process all 128 state dims in one shot
        # CHUNK=128 = single ONNX node group (~63 nodes after simplify vs ~1,280 for CHUNK=1)
        CHUNK = 128
        y = torch.zeros(B * H, L, P, device=x.device, dtype=x.dtype)
        
        for s0 in range(0, S, CHUNK):
            s_end = min(s0 + CHUNK, S)
            csz = s_end - s0
            
            # a_t = exp(dt_h * A_h[:, s0:s_end]): (BH, L, csz)
            # Use exp(cumsum(log(a_t))) instead of cumprod(a_t) for ONNX compatibility
            log_a_t = dt_h.unsqueeze(-1) * A_h[:, None, s0:s_end]  # (BH, L, csz)
            cumsum_log_a = torch.cumsum(log_a_t, dim=1)  # (BH, L, csz)
            cumprod_a = torch.exp(cumsum_log_a)  # (BH, L, csz)
            
            zeros = torch.zeros(B * H, 1, csz, device=x.device, dtype=x.dtype)
            cumprod_a_pad = torch.exp(torch.cat([zeros, cumsum_log_a[:, :-1, :]], dim=1))  # (BH, L, csz)
            cumprod_a_pad = cumprod_a_pad.clamp(min=1e-12)  # prevent div-by-zero
            
            # u = B * x: (BH, L, csz, P)
            u = B_h[:, :, s0:s_end, None] * xh[:, :, None, :]  # (BH, L, csz, P)
            
            # h = cumprod_a * cumsum(u / cumprod_a_pad, dim=1)
            h_c = cumprod_a[:, :, :, None] * torch.cumsum(
                u / cumprod_a_pad[:, :, :, None], dim=1
            )  # (BH, L, csz, P)
            
            # y += sum over state dims: C * h → (BH, L, P)
            y = y + (C_h[:, :, s0:s_end, None] * h_c).sum(dim=2)  # (BH, L, P)
        
        # Reshape back
        y = y.reshape(B, H, L, P).transpose(1, 2).reshape(B, L, -1)  # (B, L, d_inner)
        
        # D skip + gate
        y = y + self.D.mean() * x_proj
        y = y * F.silu(z)
        
        return self.out_proj(y)  # (B, L, D)


# ═══════════════════════════════════════════════════════════════════════
# Mamba3 Block Selection (unified)
# ═══════════════════════════════════════════════════════════════════════

def _get_mamba3(d_model, d_state=128, expand=2, headdim=64, ngroups=1, **kwargs):
    """Get Mamba3 block, falling back to SimpleSSM if mamba-ssm not installed."""
    if _MAMBA3_AVAILABLE:
        return Mamba3(
            d_model=d_model, d_state=d_state, expand=expand,
            headdim=headdim, ngroups=ngroups,
            **{k: v for k, v in kwargs.items() if k in ('chunk_size', 'rope_fraction')}
        )
    return SimpleSSM(
        d_model=d_model, d_state=d_state, expand=expand,
        headdim=headdim, ngroups=ngroups
    )


# ═══════════════════════════════════════════════════════════════════════
# Gated Mamba FFN
# ═══════════════════════════════════════════════════════════════════════

class GatedMambaFFN(nn.Module):
    """
    Gated Feed-Forward Network for Mamba3 blocks.
    Replaces TransformerFFNLayer.
    
    Architecture: Conv1d(d → 4d*2) → SwiGLU → Conv1d(4d → d) → Dropout
    
    Input: (B, T, C)
    Output: (B, T, C)
    
    Constructor matches TransformerFFNLayer signature for compatibility.
    """
    def __init__(self, hidden_size, filter_size, kernel_size=1, dropout=0., act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        
        inner_dim = filter_size
        if act == 'relu':
            self.act_fn = nn.ReLU()
            self.ffn_1 = nn.Conv1d(hidden_size, inner_dim, kernel_size, padding=kernel_size // 2)
        elif act == 'gelu':
            self.act_fn = nn.GELU()
            self.ffn_1 = nn.Conv1d(hidden_size, inner_dim, kernel_size, padding=kernel_size // 2)
        elif act == 'silu' or act == 'swish':
            self.act_fn = nn.SiLU()
            self.ffn_1 = nn.Conv1d(hidden_size, inner_dim, kernel_size, padding=kernel_size // 2)
        elif act == 'swiglu':
            self.act_fn = None
            self.ffn_1 = nn.Conv1d(hidden_size, inner_dim * 2, kernel_size, padding=kernel_size // 2)
        else:
            raise ValueError(f"'{act}' is not a valid activation")
        
        self.ffn_2 = nn.Linear(inner_dim, hidden_size)
        nn.init.xavier_uniform_(self.ffn_2.weight)
        if self.ffn_2.bias is not None:
            nn.init.constant_(self.ffn_2.bias, 0.)

    def forward(self, x):
        x = self.ffn_1(x.transpose(1, 2)).transpose(1, 2)
        x = x * self.kernel_size ** -0.5
        
        if self.act == 'swiglu':
            out, gate = x.chunk(2, dim=-1)
            x = out * F.silu(gate)
        else:
            x = self.act_fn(x)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
# Bidirectional Mamba3 Block
# ═══════════════════════════════════════════════════════════════════════

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba3 encoder block.
    Replaces EncSALayer (Self-Attention + FFN transformer layer).
    
    Uses Vision Mamba-style bidirectional scan with Mamba3:
        Forward Mamba3 + Reverse Mamba3 -> merge -> residual
    
    Architecture:
        x → LayerNorm → FwdMamba3 + RevMamba3(flipped) → avg → Dropout → Residual
        x → LayerNorm → GatedMambaFFN → Dropout → Residual
    
    Input: (B, T, C)
    Output: (B, T, C)
    
    Constructor matches EncSALayer signature for compatibility.
    """
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, act='gelu', rotary_embed=None):
        super().__init__()
        self.dropout = dropout
        d_model = c
        
        # Mamba3 parameters: d_state=128 default, no d_conv (Mamba3 has no local conv)
        d_state = 128
        expand = 2
        # Map kernel_size to headdim hint (Mamba3 uses headdim=64 default)
        headdim = 64
        
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba_fwd = _get_mamba3(d_model, d_state=d_state, expand=expand, headdim=headdim)
        self.mamba_rev = _get_mamba3(d_model, d_state=d_state, expand=expand, headdim=headdim)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GatedMambaFFN(
            hidden_size=d_model,
            filter_size=4 * d_model,
            kernel_size=kernel_size,
            dropout=relu_dropout,
            act=act
        )
        
        self.num_heads = num_heads
        self.use_rope = rotary_embed is not None

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.norm1.training = layer_norm_training
            self.norm2.training = layer_norm_training
        
        # ── Bidirectional Mamba3 (replaces self-attention) ──
        residual = x
        x_norm = self.norm1(x)
        
        x_fwd = self.mamba_fwd(x_norm)
        x_rev = self.mamba_rev(torch.flip(x_norm, dims=[1]))
        x_rev = torch.flip(x_rev, dims=[1])
        
        x = (x_fwd + x_rev) * 0.5
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.float())[..., None]
        
        # ── FFN ──
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.float())[..., None]
        
        return x


# ═══════════════════════════════════════════════════════════════════════
# Mamba3 Encoder (Stack of BiMambaBlock)
# ═══════════════════════════════════════════════════════════════════════

class BiMambaAttnWrapper(nn.Module):
    """
    Transformer Self-Attention 层，接口兼容 BiMambaBlock。
    用于混合 Encoder：与 BiMambaBlock 交替排列在同一 MambaEncoder 中。

    forward(x, encoder_padding_mask=None, **kwargs) → Tensor
    输入输出形状：[B, T, C] (batch_first)
    """

    def __init__(self, d_model, num_heads, dropout, kernel_size, act):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model * 4, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(d_model * 4, d_model, kernel_size=1),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        # [B, T, C] ← 与 BiMambaBlock 相同格式
        residual = x
        x_norm = self.norm1(x)
        attn_mask = None
        if encoder_padding_mask is not None:
            attn_mask = encoder_padding_mask.bool()
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        x = residual + self.dropout1(x_attn)

        residual = x
        x_norm2 = self.norm2(x)
        x_ffn = x_norm2.transpose(1, 2)  # [B,T,C] → [B,C,T]
        x_ffn = self.dropout2(self.ffn(x_ffn))
        x_ffn = x_ffn.transpose(1, 2)  # [B,C,T] → [B,T,C]
        x = residual + x_ffn

        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.float())[:, :, None]
        return x


def _build_attn_layer(hidden_size, num_heads, dropout, kernel_size, act):
    """构造一个标准 Transformer Self-Attention 层（用于混合 Encoder）"""
    return BiMambaAttnWrapper(
        d_model=hidden_size, num_heads=num_heads,
        dropout=dropout, kernel_size=kernel_size, act=act,
    )


class MambaEncoder(nn.Module):
    """
    Stack of bidirectional Mamba3 blocks forming a full encoder.
    Replaces the FastSpeech2Encoder's internal transformer stack.

    Supports mixed layer types via `layer_types`:
      - None / ['mamba']*N → pure Mamba3 (backward compatible)
      - ['mamba','mamba','attention','attention'] → hybrid
    """

    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, ffn_act='gelu',
                 dropout=0.1, num_heads=2, rotary_embed=None, layer_types=None):
        super().__init__()
        if layer_types is None:
            layer_types = ['mamba'] * num_layers
        if len(layer_types) != num_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) != num_layers ({num_layers})"
            )

        layers = []
        for lt in layer_types:
            if lt == 'mamba':
                layers.append(BiMambaBlock(
                    c=hidden_size, num_heads=num_heads, dropout=dropout,
                    attention_dropout=0.0, relu_dropout=dropout,
                    kernel_size=ffn_kernel_size, act=ffn_act, rotary_embed=rotary_embed,
                ))
            elif lt == 'attention':
                layers.append(_build_attn_layer(
                    hidden_size=hidden_size, num_heads=num_heads,
                    dropout=dropout, kernel_size=ffn_kernel_size, act=ffn_act,
                ))
            else:
                raise ValueError(f"Unknown layer type: {lt!r}, expected 'mamba' or 'attention'")
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, padding_mask=None, attn_mask=None, return_hiddens=False):
        nonpadding_mask = 1 - padding_mask.float()[:, :, None] if padding_mask is not None else None
        
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask)
            if nonpadding_mask is not None:
                x = x * nonpadding_mask
            if return_hiddens:
                hiddens.append(x)
        
        x = self.layer_norm(x)
        if nonpadding_mask is not None:
            x = x * nonpadding_mask
        
        if return_hiddens:
            return torch.stack(hiddens, 0)  # [L, B, T, C]
        return x


# ═══════════════════════════════════════════════════════════════════════
# SSM Diffusion Backbone (Mamba3)
# ═══════════════════════════════════════════════════════════════════════

class MambaResidualBlock(nn.Module):
    """
    SSM-based residual block for diffusion backbone using Mamba3.
    Replaces both LYNXNetResidualLayer and WaveNet ResidualBlock.
    
    Condition injection: Gated Modulation
        gate, bias = cond_proj(cond).chunk(2)
        x = x * sigmoid(gate) + bias
    
    Args:
        dim: channel dimension (e.g. 512)
        cond_dim: condition dimension from encoder (e.g. 256)
        d_state: SSM state size (Mamba3 default: 128)
        expand: channel expansion factor
        dropout: dropout rate
    """
    def __init__(self, dim, cond_dim, d_state=128, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        
        self.cond_proj = nn.Conv1d(cond_dim, dim * 2, 1)
        self.diff_proj = nn.Conv1d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)
        
        # Mamba3 backbone: headdim=64 is the default
        self.mamba = _get_mamba3(dim, d_state=d_state, expand=expand, headdim=64)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, cond, diffusion_step):
        """
        x: (B, C, T) channel-first like LYNXNet
        cond: (B, H, T) condition from encoder
        diffusion_step: (B, C, T) diffusion step embedding
        Returns: (B, C, T)
        """
        residual = x
        
        # Gated condition modulation
        cond_mod = self.cond_proj(cond)
        gate, bias = cond_mod.chunk(2, dim=1)
        x = x * torch.sigmoid(gate) + bias
        
        # Diffusion step injection
        x = x + self.diff_proj(diffusion_step)
        
        # Mamba3 processing
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, C, T)
        
        x = residual + x
        return x


class MambaBackbone(nn.Module):
    """
    Mamba3-based diffusion backbone replacing both LYNXNet and WaveNet.
    
    Architecture:
        Input(spec) → Conv1d(F*M → C) → GELU
        DiffStep → SinusoidalPosEmb → MLP → (B, C, T)
        
        [MambaResidualBlock × num_layers]
            x → GatedCondModulation(cond) → +DiffStep → Mamba3 → Residual
        
        Output → LayerNorm → Conv1d(C → F*M) → Reshape
    
    Interface matches LYNXNet/WaveNet exactly:
        forward(spec, diffusion_step, cond) → velocity_prediction
    """
    def __init__(self, in_dims, n_feats, *, num_layers=12, num_channels=512,
                 d_state=128, d_conv=4, expand=2, kernel_size=31,
                 activation='PReLU', dropout=0.0, strong_cond=False):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.strong_cond = strong_cond
        
        self.input_projection = nn.Conv1d(in_dims * n_feats, num_channels, 1)
        nn.init.kaiming_normal_(self.input_projection.weight)
        
        from modules.commons.common_layers import SinusoidalPosEmb
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        
        import sys
        _hp = sys.modules.get('utils.hparams')
        cond_dim = getattr(_hp, 'hparams', {}).get('hidden_size', 256) if _hp else 256
        
        self.residual_layers = nn.ModuleList([
            MambaResidualBlock(
                dim=num_channels, cond_dim=cond_dim,
                d_state=d_state, expand=expand, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Conv1d(num_channels, in_dims * n_feats, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        spec:           [B, F, M, T] diffusion sample (F=1 typically)
        diffusion_step: [B, 1] diffusion timestep
        cond:           [B, H, T] condition signal from encoder
        Returns:        [B, F, M, T] velocity field prediction
        """
        if self.n_feats == 1:
            x = spec[:, 0]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)
        
        x = self.input_projection(x)
        if not self.strong_cond:
            x = F.gelu(x)
        
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = diffusion_step.squeeze(1).unsqueeze(-1)
        
        for layer in self.residual_layers:
            x = layer(x, cond, diffusion_step)
        
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.output_projection(x)
        
        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        
        return x
