"""
SSM (State Space Model) layers for DiffSinger.
Replaces all Transformer components with Mamba3-based SSM variants.

Architecture:
- BiMambaBlock: Bidirectional Mamba3 encoder block (replaces EncSALayer)
- GatedMambaFFN: Gated Conv1d FFN (replaces TransformerFFNLayer)  
- MambaEncoder: Stack of BiMambaBlock (replaces FastSpeech2Encoder)
- MambaResidualBlock: SSM diffusion residual block (replaces LYNXNet/WaveNet blocks)
- MambaBackbone: Full SSM diffusion backbone (replaces LYNXNet/WaveNet)

SSM Runtime:
- Training: set DIFFSINGER_USE_MAMBA3=1 to train with mamba-ssm Mamba3 (CUDA).
- ONNX export: SimpleSSM loads the same Mamba3-layout weights with strict=True,
  forward decomposed into standard ONNX ops (Conv, MatMul, Exp, CumSum, etc.).
- No custom ONNX ops or DLL required — exported ONNX runs in any standard
  ONNX Runtime environment including OpenUtau out of the box.

References:
- https://github.com/state-spaces/mamba
- https://github.com/hustvl/Vim (bidirectional Mamba)
- https://github.com/tyshiwo1/DiM-DiffusionMamba (SSM diffusion backbone)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Mamba3 import with explicit opt-in ─────────────────────────────────
_USE_MAMBA3 = os.environ.get('DIFFSINGER_USE_MAMBA3', '').lower() in {'1', 'true', 'yes'}
_MAMBA3_AVAILABLE = False

if _USE_MAMBA3:
    try:
        from mamba_ssm import Mamba3
        _MAMBA3_AVAILABLE = True
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# Pure PyTorch SSM Fallback (CPU / non-CUDA)
# Mamba3 architecture mirror for ONNX-exportable parameter loading.
#
# Parameter layout EXACTLY matches mamba-ssm Mamba3 so that a
# Mamba3-trained checkpoint loads with strict=True into this class.
# Forward pass uses the SSD diagonal-scan algorithm via cumsum/exp,
# which is ONNX-traceable and produces meaningful output (though not
# bit-identical to Mamba3's CUDA kernel).
# ═══════════════════════════════════════════════════════════════════════

class _RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (matches Mamba3 internals)."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # x: (..., d)
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms).to(x.dtype) * self.weight


class SimpleSSM(nn.Module):
    """
    Pure-PyTorch SSM mirroring mamba_ssm.Mamba3 (SISO, default config) exactly.

    Parameter names/shapes match `mamba_ssm.modules.mamba3.Mamba3` so a
    Mamba-3 trained checkpoint loads with strict=True:

      in_proj.weight  (d_in_proj, d_model)
        d_in_proj = 2·d_inner + 2·d_state·ngroups·mimo_rank
                    + 3·nheads + num_rope_angles
      B_bias          (nheads, mimo_rank, d_state)
      C_bias          (nheads, mimo_rank, d_state)
      B_norm.weight   (d_state,)
      C_norm.weight   (d_state,)
      dt_bias         (nheads,)
      D               (nheads,)
      out_proj.weight (d_model, d_inner)

    Forward replicates the SISO combined Triton kernel using only
    ONNX-traceable ops (Linear / matmul / cumsum / exp / sin / cos /
    softplus / sigmoid / tanh / floor / where / pad). No custom op.
    """

    def __init__(self, d_model, d_state=128, expand=2,
                 headdim=64, ngroups=1, rope_fraction=0.5,
                 mimo_rank=1, A_floor=1e-4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim

        d_inner = int(d_model * expand)
        assert d_inner % headdim == 0, \
            f"d_inner ({d_inner}) must be divisible by headdim ({headdim})"
        self.d_inner = d_inner
        self.nheads = d_inner // headdim
        self.num_bc_heads = ngroups
        self.mimo_rank = int(mimo_rank)
        self.A_floor = A_floor

        # RoPE config (Mamba-3 default rope_fraction=0.5)
        assert rope_fraction in (0.5, 1.0)
        self.rotary_dim_divisor = int(2 / rope_fraction)
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        assert self.num_rope_angles > 0

        H = self.nheads
        S = self.d_state
        # Order: [z, x, B, C, dd_dt, dd_A, trap, angles]
        d_in_proj = (2 * d_inner
                     + 2 * S * self.num_bc_heads * self.mimo_rank
                     + 3 * H
                     + self.num_rope_angles)
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        self.B_bias = nn.Parameter(torch.ones(H, self.mimo_rank, S))
        self.C_bias = nn.Parameter(torch.ones(H, self.mimo_rank, S))
        self.B_norm = _RMSNorm(S)
        self.C_norm = _RMSNorm(S)

        self.dt_bias = nn.Parameter(torch.zeros(H))
        self.D = nn.Parameter(torch.ones(H))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    @staticmethod
    def _safe_clamp(x):
        """NaN/Inf/value clamp via torch.where (ONNX-traceable)."""
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.where(torch.isinf(x) & (x > 0), torch.full_like(x, 1e4), x)
        x = torch.where(torch.isinf(x) & (x < 0), torch.full_like(x, -1e4), x)
        return torch.clamp(x, min=-1e4, max=1e4)

    @staticmethod
    def _rope_pair(x, cos, sin):
        """Apply RoPE on the last dim by splitting it into pairs.

        x: (..., S) where S is even.
        cos, sin: (..., S//2).  pairs (x0, x1) become (x0·c - x1·s, x0·s + x1·c).
        """
        *prefix, S = x.shape
        x_pair = x.reshape(*prefix, S // 2, 2)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        return torch.stack((o0, o1), dim=-1).reshape(*prefix, S)

    def forward(self, x):
        """
        x: (B, L, D)  →  (B, L, D).

        Two equivalent execution paths share the same prefix (in_proj split,
        RoPE, decay accumulation).  They diverge only in how the time-domain
        SSM update is computed.

          eager / training:     chunked SSD (O(L·S) memory).
          torch.jit.tracing()   dense O(L²) form, no Python-side branches.
                                ONNX gets fully dynamic shapes, no fused
                                "chunk size" constants in Reshape/Pad.

        Both paths are mathematically the same; we verified max diff ~1e-4
        fp32 against the dense form once and lock the contract here.
        """
        z, V, Q_rot, K_scaled, ADT, da_cs, qk_diag = self._forward_prefix(x)
        if torch.jit.is_tracing():
            out_off = self._forward_dense(Q_rot, K_scaled, V, ADT, da_cs)
        else:
            out_off = self._forward_chunked(Q_rot, K_scaled, V, ADT, da_cs)

        H = self.nheads
        out = out_off + (self.D.view(1, 1, H, 1) + qk_diag.unsqueeze(-1)) * V
        out = out * F.silu(z)
        B, L = x.shape[0], x.shape[1]
        out = out.reshape(B, L, self.d_inner)
        out = self.out_proj(out)
        return self._safe_clamp(out)

    # ── shared front-end ────────────────────────────────────────────────
    def _forward_prefix(self, x):
        B, L, _ = x.shape
        H = self.nheads
        P = self.headdim
        S = self.d_state
        R = self.num_rope_angles
        d_inner = self.d_inner

        x = self._safe_clamp(x)

        zxBCdtA = self.in_proj(x)
        z, x_v, Bs, Cs, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtA,
            [d_inner, d_inner,
             S * self.num_bc_heads * self.mimo_rank,
             S * self.num_bc_heads * self.mimo_rank,
             H, H, H, R],
            dim=-1,
        )

        z = z.reshape(B, L, H, P)
        V = x_v.reshape(B, L, H, P)

        Bs = self._safe_clamp(self.B_norm(Bs))
        Cs = self._safe_clamp(self.C_norm(Cs))
        B_bias_h = self.B_bias.reshape(H, S)
        C_bias_h = self.C_bias.reshape(H, S)
        K_pre = Bs.unsqueeze(2) + B_bias_h.view(1, 1, H, S)
        Q_pre = Cs.unsqueeze(2) + C_bias_h.view(1, 1, H, S)

        DT = F.softplus(dd_dt + self.dt_bias).clamp(min=1e-6, max=20.0)
        # NOTE: parenthesise the unary minus.  `-F.softplus(x).clamp(max=-A)`
        # parses as `-(softplus(x).clamp(max=-A))`, which collapses A_neg to
        # +A_floor and kills the SSM decay.  See the parity probe in
        # tools/_remote_ssm_parity_v4.py for the regression evidence.
        A_neg = (-F.softplus(dd_A)).clamp(max=-self.A_floor)
        ADT = (A_neg * DT).clamp(min=-60.0, max=0.0)

        trap_s = torch.sigmoid(trap)
        gamma = DT * trap_s
        zero_pad = DT.new_zeros(B, 1, H)
        DT_next = torch.cat([DT[:, 1:, :], zero_pad], dim=1)
        trap_next = torch.cat([trap_s[:, 1:, :], zero_pad], dim=1)
        scale = DT_next * (1.0 - trap_next) + gamma
        qk_diag = (Q_pre * K_pre).sum(dim=-1) * gamma

        angles_pure = torch.tanh(angles) * math.pi
        vals = angles_pure.unsqueeze(2) * DT.unsqueeze(-1)
        ang_cs = torch.cumsum(vals, dim=1)
        two_pi = 2.0 * math.pi
        ang_cs = ang_cs - two_pi * torch.floor(ang_cs / two_pi)
        pad_dim = S // 2 - R
        if pad_dim > 0:
            ang_cs = F.pad(ang_cs, (0, pad_dim), value=0.0)
        cos_b = torch.cos(ang_cs)
        sin_b = torch.sin(ang_cs)

        K_rot = self._rope_pair(K_pre, cos_b, sin_b)
        Q_rot = self._rope_pair(Q_pre, cos_b, sin_b)
        K_scaled = self._safe_clamp(K_rot * scale.unsqueeze(-1))
        Q_rot = self._safe_clamp(Q_rot)

        da_cs = torch.cumsum(ADT, dim=1)                      # (B, L, H)

        return z, V, Q_rot, K_scaled, ADT, da_cs, qk_diag

    # ── ONNX-friendly dense path ────────────────────────────────────────
    def _forward_dense(self, Q_rot, K_scaled, V, ADT, da_cs):
        """O(L²) reference equivalent of chunked SSD.

        Semantics match the chunked path's <c>out_off</c>: a strictly causal
        (t' < t) contribution.  The self (t' == t) contribution is added by
        the caller via <c>(D + qk_diag) * V</c>.  Mathematically:

            out_off[b, t, h, p] = Σ_{t' < t}
                exp(da_cs[b,t,h] - da_cs[b,t',h])
                · <Q_rot[b,t,h,:], K_scaled[b,t',h,:]>
                · V[b,t',h,p]

        Trace-friendly: only model-fixed dims (H, S, P) are constants;
        the sequence axis stays dynamic.
        """
        # logits[b, t_q, t_k, h] = <Q_rot[b,t_q,h,:], K_scaled[b,t_k,h,:]>
        logits = torch.einsum('bthd,bshd->btsh', Q_rot, K_scaled)

        # decay[b, t_q, t_k, h] = exp(da_cs[b,t_q,h] - da_cs[b,t_k,h])
        log_dec = da_cs.unsqueeze(2) - da_cs.unsqueeze(1)         # (B, L_q, L_k, H)
        log_dec = log_dec.clamp(min=-60.0, max=0.0)

        # strictly causal mask: t_k < t_q.  Built from da_cs to inherit the
        # dynamic sequence length from the trace; never materialised as a
        # python int.
        L_arange = torch.arange(da_cs.shape[1], device=da_cs.device)
        # row = t_q, col = t_k → t_k < t_q
        causal = (L_arange.unsqueeze(1) > L_arange.unsqueeze(0)).to(logits.dtype)
        causal = causal.unsqueeze(0).unsqueeze(-1)                 # (1, L_q, L_k, 1)

        attn = logits * torch.exp(log_dec) * causal                # (B, L_q, L_k, H)
        out  = torch.einsum('btsh,bshp->bthp', attn, V)            # (B, L_q, H, P)
        return out

    # ── eager-mode chunked path ─────────────────────────────────────────
    def _forward_chunked(self, Q_rot, K_scaled, V, ADT, da_cs):
        B, L, H, S = Q_rot.shape
        P = V.shape[-1]

        chunk = 64 if L >= 64 else int(L)
        if chunk == 0:
            chunk = 1
        pad = (-L) % chunk
        if pad:
            Q_rot    = F.pad(Q_rot,    (0, 0, 0, 0, 0, pad))
            K_scaled = F.pad(K_scaled, (0, 0, 0, 0, 0, pad))
            V        = F.pad(V,        (0, 0, 0, 0, 0, pad))
            ADT_p    = F.pad(ADT,      (0, 0, 0, pad))
            da_cs    = F.pad(da_cs,    (0, 0, 0, pad))
        else:
            ADT_p    = ADT
        L_pad = L + pad
        n_chunks = L_pad // chunk

        Qc = Q_rot   .reshape(B, n_chunks, chunk, H, S)
        Kc = K_scaled.reshape(B, n_chunks, chunk, H, S)
        Vc = V       .reshape(B, n_chunks, chunk, H, P)
        ADTc  = ADT_p.reshape(B, n_chunks, chunk, H)
        DAc   = da_cs.reshape(B, n_chunks, chunk, H)
        chunk_end_log_decay = ADTc.sum(dim=2)                 # (B, n, H)

        idx = torch.arange(chunk, device=Q_rot.device)
        causal = (idx.unsqueeze(0) < idx.unsqueeze(1)).to(Qc.dtype)
        chunk_start_da = DAc[:, :, :1, :] - ADTc[:, :, :1, :]
        rel_da = DAc - chunk_start_da
        log_dec_local = rel_da.unsqueeze(3) - rel_da.unsqueeze(2)
        log_dec_local = log_dec_local.permute(0, 1, 2, 4, 3).contiguous()
        log_dec_local = log_dec_local.clamp(min=-60.0, max=0.0)
        decay_local = torch.exp(log_dec_local)
        logits_local = torch.einsum('bnchs,bndhs->bnchd', Qc, Kc)
        attn_local = logits_local * decay_local * causal.view(1, 1, chunk, 1, chunk)
        out_local = torch.einsum('bnchd,bndhp->bnchp', attn_local, Vc)

        kv_chunk = torch.einsum('bnchs,bnchp->bnhsp', Kc, Vc)

        state = kv_chunk.new_zeros(B, H, S, P)
        out_cross = kv_chunk.new_zeros(B, n_chunks, chunk, H, P)
        for n in range(n_chunks):
            if n > 0:
                prev_end_log = chunk_end_log_decay[:, n - 1, :]
                state = state * torch.exp(prev_end_log).clamp(min=0.0, max=1.0).view(B, H, 1, 1)
                state = state + kv_chunk[:, n - 1]
            local_decay = torch.exp(rel_da[:, n].clamp(min=-60.0, max=0.0))
            q_state = torch.einsum('bchs,bhsp->bchp', Qc[:, n], state)
            out_cross[:, n] = q_state * local_decay.unsqueeze(-1)

        out_off = (out_local + out_cross).reshape(B, L_pad, H, P)
        return out_off[:, :L]


# ═══════════════════════════════════════════════════════════════════════
# Mamba3 Block Selection (unified)
# ═══════════════════════════════════════════════════════════════════════

def _get_mamba3(d_model, d_state=128, expand=2, headdim=64, ngroups=1, **kwargs):
    """Get Mamba3 block.

    Training (DIFFSINGER_USE_MAMBA3=1): mamba-ssm Mamba3 (CUDA).
    Export / fallback:          SimpleSSM — pure PyTorch, ONNX-exportable
                                with strictly standard ops (no custom op / DLL needed).
    """
    _mamba3_kw = {k: v for k, v in kwargs.items()
                  if k in ('chunk_size', 'rope_fraction', 'd_conv', 'dt_rank')}
    if _MAMBA3_AVAILABLE:
        return Mamba3(
            d_model=d_model, d_state=d_state, expand=expand,
            headdim=headdim, ngroups=ngroups, **_mamba3_kw,
        )
    return SimpleSSM(
        d_model=d_model, d_state=d_state, expand=expand,
        headdim=headdim, ngroups=ngroups, **_mamba3_kw,
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
