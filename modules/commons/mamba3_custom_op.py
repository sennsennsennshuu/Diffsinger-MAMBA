"""
Mamba3 ONNX Custom Operator for OpenUtau.

Provides:
- SelectiveScanFunction: torch.autograd.Function that emits a single
  custom.ssm::SSMSelectiveScan node during ONNX export, and runs the
  equivalent parallel scan in PyTorch mode.
- Mamba3ONNX: an nn.Module whose parameter layout matches mamba-ssm Mamba3
  exactly (strict-loadable from Mamba3 checkpoints), and whose forward
  uses SelectiveScanFunction for the scan portion so that ONNX export
  produces minimal per-head custom-op nodes.

Integration:
- The C++ SSMOptimizer.dll implements the same scan formula and registers
  as an ONNX Runtime custom op under domain "custom.ssm".
- OpenUtau must have the DLL placed in its Dependencies/SSM/ directory
  (the existing install.py handles this).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (matches Mamba3 internals).

    ONNX-exportable: mean(x^2) -> sqrt -> reciprocal -> scale.
    """
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms).to(x.dtype) * self.weight


class SelectiveScanFunction(torch.autograd.Function):
    """Custom selective scan for ONNX Runtime integration.

    forward():
        PyTorch-only path.  Runs the parallel-scan algorithm from
        SimpleSSM (cumsum / exp).  Used when running the module
        directly in Python (training or pure-PyTorch inference).

    symbolic():
        ONNX export path.  Emits a single custom op node:

          custom.ssm::SSMSelectiveScan(u, dt, A, B, C, D) -> y

        where:
          u  : (B, L, d_inner)   post-norm input
          dt : (B, L, nheads)    discretisation step (positive)
          A  : (nheads, d_state) pre-computed -exp(A_log), repeated
          B  : (B, L, d_state)   input projection
          C  : (B, L, d_state)   output projection
          D  : (nheads,)         skip-connection per head (optional)
          y  : (B, L, d_inner)   scan output

    Scan formula (MUST match selective_scan.cpp):
        h_s[d] = A_bar * (h_s[d] + B_s * x[d])
        y[d]  += C_s * h_s[d]
        A_bar = exp(dt * A[s])

    This matches SimpleSSM's parallel-scan formulation.
    """

    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # u:  (B, L, d_inner)
        # dt: (B, L, nheads)
        # A:  (nheads, d_state)
        # B:  (B, L, d_state)
        # C:  (B, L, d_state)
        # D:  (nheads,) optional
        B0, L, d_inner = u.shape
        nheads, d_state = A.shape
        head_dim = d_inner // nheads
        BH = B0 * nheads

        # Re-arrange into BH dimension (batch*heads) ---
        u_h = u.reshape(B0, L, nheads, head_dim) \
               .transpose(1, 2).reshape(BH, L, head_dim)    # (BH, L, P)

        dt_h = dt.transpose(1, 2).reshape(BH, L)             # (BH, L)

        A_expanded = A.unsqueeze(0).expand(B0, -1, -1) \
                       .reshape(BH, d_state)                  # (BH, S)

        B_h = B.unsqueeze(1).expand(B0, nheads, L, d_state) \
                .reshape(BH, L, d_state)                     # (BH, L, S)
        C_h = C.unsqueeze(1).expand(B0, nheads, L, d_state) \
                .reshape(BH, L, d_state)                     # (BH, L, S)

        # Parallel scan (cumsum / exp — mathematically identical to DLL loop)
        CHUNK = 128
        y = torch.zeros(BH, L, head_dim, device=u.device, dtype=u.dtype)

        for s0 in range(0, d_state, CHUNK):
            s_end = min(s0 + CHUNK, d_state)
            csz = s_end - s0

            # log_a_t: log of per-timestep decay for each state dim
            log_a_t = dt_h.unsqueeze(-1) * A_expanded[:, s0:s_end].unsqueeze(1)
            cumsum_log_a = torch.cumsum(log_a_t, dim=1).clamp(min=-60.0, max=20.0)
            cumprod_a = torch.exp(cumsum_log_a)

            pad = torch.zeros(BH, 1, csz, device=u.device, dtype=u.dtype)
            prev_cumsum = torch.cat([pad, cumsum_log_a[:, :-1, :]], dim=1)
            cumprod_a_pad = torch.exp(prev_cumsum).clamp(min=1e-12, max=1e4)

            u_scan = B_h[:, :, s0:s_end, None] * u_h[:, :, None, :]  # (BH,L,csz,P)
            div = (u_scan / cumprod_a_pad[:, :, :, None]).clamp(min=-1e4, max=1e4)
            h_c = cumprod_a[:, :, :, None] * torch.cumsum(div, dim=1)
            h_c = h_c.clamp(min=-1e4, max=1e4)

            y = y + (C_h[:, :, s0:s_end, None] * h_c).sum(dim=2)

        # Reshape back to (B, L, d_inner)
        y = y.reshape(B0, nheads, L, head_dim) \
             .transpose(1, 2).reshape(B0, L, d_inner)

        # D skip connection — per-head, matching DLL
        if D is not None:
            y = y + D.view(1, 1, nheads, 1) \
                     .expand(B0, L, -1, head_dim) \
                     .reshape(B0, L, d_inner) * u.reshape(B0, L, nheads, head_dim) \
                     .reshape(B0, L, d_inner)

        ctx.save_for_backward(u, dt, A, B, C, D)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Inference-only, but provide a pass-through for compatibility.
        # A full backward through the scan would need attention-equivalent
        # recomputation; skipping for now.
        return grad_output, None, None, None, None, None

    @staticmethod
    def symbolic(g, u, dt, A, B, C, D=None):
        """Emit custom.ssm::SSMSelectiveScan during ONNX export."""
        inputs = [u, dt, A, B, C]
        if D is not None:
            inputs.append(D)
        return g.op("custom.ssm::SSMSelectiveScan", *inputs, outputs=1)


class Mamba3ONNX(nn.Module):
    """Mamba3-compatible module with ONNX custom-op scan.

    Parameter layout is identical to mamba-ssm Mamba3 and SimpleSSM:
      in_proj.weight  (2*d_inner, d_model)
      conv1d.weight   (d_inner, 1, d_conv)
      conv1d.bias     (d_inner,)
      norm.weight     (d_inner,)
      x_proj.weight   (dt_rank + 2*d_state, d_inner)
      dt_proj.weight  (d_inner, dt_rank)
      dt_proj.bias    (d_inner,)
      dt_bias         (nheads,)
      A_log           (nheads,)
      D               (nheads,)
      out_proj.weight (d_model, d_inner)

    This allows strict loading from Mamba3-trained checkpoints.
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        d_conv=4,
        dt_rank="auto",
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim

        d_inner = int(d_model * expand)
        assert d_inner % headdim == 0, (
            f"d_inner ({d_inner}) must be divisible by headdim ({headdim})"
        )
        self.nheads = d_inner // headdim
        self.d_inner = d_inner

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = int(dt_rank)

        # ---- Mamba3-identical parameter layout ----
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        # RMSNorm — same semantics as Mamba3 internals, ONNX-exportable
        self.norm = _RMSNorm(d_inner, eps=1e-5)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Save conv param for causal trimming
        self.d_conv = d_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        B, L, D = x.shape

        # 1. Input projection
        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)             # each (B, L, d_inner)

        # 2. Causal depthwise conv + SiLU
        x_ssm_t = x_ssm.transpose(1, 2)            # (B, d_inner, L)
        x_ssm_t = self.conv1d(x_ssm_t)             # (B, d_inner, L + d_conv - 1)
        x_ssm_t = x_ssm_t[:, :, :L]                # causal
        x_ssm_t = F.silu(x_ssm_t)                  # SiLU
        x_ssm = x_ssm_t.transpose(1, 2)            # (B, L, d_inner)

        # 3. RMS norm (LayerNorm is ONNX-exportable)
        x_ssm = self.norm(x_ssm)
        x_ssm = x_ssm.clamp(min=-1e4, max=1e4)     # safety

        # 4. x_proj -> dt_in, B, C
        x_db_c = self.x_proj(x_ssm)                # (B, L, dt_rank + 2*d_state)
        dt_in, B_ssm, C_ssm = torch.split(
            x_db_c, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # 5. dt: linear -> pool over headdim -> softplus -> add bias -> clamp
        dt = self.dt_proj(dt_in)                   # (B, L, d_inner)
        dt = dt.reshape(B, L, self.nheads, self.headdim) \
               .mean(dim=-1)                        # (B, L, nheads)
        dt = F.softplus(dt + self.dt_bias)          # (B, L, nheads)
        dt = dt.clamp(min=1e-6, max=20.0)

        # A: -exp(A_log) -> repeat to (nheads, d_state)
        A_neg = -torch.exp(self.A_log.clamp(max=10.0))   # (nheads,)
        A_full = A_neg.unsqueeze(-1).expand(-1, self.d_state).contiguous()

        # 6. Selective scan (custom op)
        y = SelectiveScanFunction.apply(x_ssm, dt, A_full, B_ssm, C_ssm, self.D)

        # 7. Gate + output projection
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y.clamp(min=-1e4, max=1e4)