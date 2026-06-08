# SSM Selective Scan — Specification

This document is the **single source of truth** for the SSM forward used by
DiffSinger-SSM. It is the contract that `SimpleSSM` (Python reference),
`SSMSelectiveScan` ONNX custom op (CPU/GPU), and any future fused kernel must
all satisfy.

When the spec and code disagree, the spec wins; code is updated to match.

## Scope

This spec covers the **SISO** (`mimo_rank=1`, `ngroups=1`) configuration used
by every SSM layer in `aco_testssm2` / `var_testssm2`:

```
d_model = 256       hidden width into / out of the layer
d_inner = 512       expanded width = expand · d_model, expand=2
headdim = 64        per-head dim
H       = 8         num heads = d_inner / headdim
S       = 128       state size (d_state)
P       = 64        per-head value/output dim (== headdim)
ngroups = 1
mimo_rank = 1
rope_fraction = 0.5
A_floor = 1e-4
```

Every tensor index uses the layout `(B, L, ...)` with `B = batch`,
`L = sequence length` (dynamic).

## Inputs

The SSM block consumes a single tensor

```
x : (B, L, d_model)    fp32
```

and the trained parameters

```
in_proj.weight  : (d_in_proj, d_model)   d_in_proj = 2*d_inner + 2*S + 3*H + R
B_bias          : (H, 1, S)              MIMO rank=1, treat as (H, S) below
C_bias          : (H, 1, S)
B_norm.weight   : (S,)
C_norm.weight   : (S,)
dt_bias         : (H,)
D               : (H,)
out_proj.weight : (d_model, d_inner)
```

with `R = num_rope_angles = floor(S * rope_fraction / 2) = 32`.

## Output

```
y : (B, L, d_model)    fp32
```

## Forward pipeline

The forward decomposes into a deterministic prefix, a causal SSM core, and a
post-mix. CPU/GPU implementations may fuse, batch, or block any of the
internal stages, but the externally observable output must equal the
reference within numerical tolerance (see § Tolerances).

### Stage 0 — input clamp

```
x ← clamp_safe(x)            (NaN→0, +Inf→1e4, −Inf→−1e4, |·|≤1e4)
```

`clamp_safe` is applied wherever the spec marks a value as "clamped". It is
implemented as `where(isnan, 0, where(isinf, ±1e4, clamp(x, -1e4, 1e4)))`.

### Stage 1 — projection split

Compute `proj = x · in_proj.weightᵀ`, `proj : (B, L, d_in_proj)`, then split
along the last axis in this exact order:

```
z       :  d_inner    (B, L, d_inner)   → reshape (B, L, H, P)
x_v     :  d_inner    (B, L, d_inner)   → reshape (B, L, H, P)         = V
B_in    :  S          (B, L, S)         (ngroups=1, mimo_rank=1)
C_in    :  S          (B, L, S)
dd_dt   :  H          (B, L, H)
dd_A    :  H          (B, L, H)
trap    :  H          (B, L, H)
angles  :  R          (B, L, R)         R = 32
```

### Stage 2 — B / C normalisation

```
Bs = clamp_safe( RMSNorm(B_in, B_norm.weight) )       (B, L, S)
Cs = clamp_safe( RMSNorm(C_in, C_norm.weight) )       (B, L, S)

K_pre = Bs[:,:,None,:] + B_bias[None,None,:,:]        (B, L, H, S)
Q_pre = Cs[:,:,None,:] + C_bias[None,None,:,:]        (B, L, H, S)
```

`RMSNorm(v, w) = (v / sqrt(mean(v², dim=-1, keepdim=True) + eps)) * w`,
`eps = 1e-5`, computed in fp32 and cast back to the input dtype.

### Stage 3 — discrete dynamics

```
DT     = softplus(dd_dt + dt_bias) clamped to [1e-6, 20.0]    (B, L, H)
A_neg  = (-softplus(dd_A)) clamped to (-∞, -A_floor]           (B, L, H)
ADT    = (A_neg * DT) clamped to [-60.0, 0.0]                  (B, L, H)
```

> Critical: `A_neg` parses as `(-softplus(...)).clamp(max=-A_floor)`. The
> alternative parse `-(softplus(...).clamp(max=-A_floor))` is wrong (it
> collapses every entry to `+A_floor` then negates). The wrong parse is the
> root cause that motivated this spec; see `tools/_remote_ssm_parity_v4.py`.

### Stage 4 — trapezoid scale

```
γ           = DT * sigmoid(trap)                              (B, L, H)
DT_next     = shift_left(DT, fill=0)                          DT[:,1:]∥0
trap_next   = shift_left(sigmoid(trap), fill=0)
scale       = DT_next * (1 - trap_next) + γ                   (B, L, H)
qk_diag     = (Q_pre · K_pre).sum(-1) * γ                     (B, L, H)
```

`shift_left(t, fill=0)` shifts `t` along the L axis by 1 step toward index 0
and fills the last position with the constant. For `t : (B, L, H)`,
`shift_left(t)[:, i, :] = t[:, i+1, :]` for `i<L-1`, `= 0` for `i=L-1`.

### Stage 5 — RoPE angles

```
α       = tanh(angles) · π                                    (B, L, R)
v       = α[:,:,None,:] · DT[:,:,:,None]                      (B, L, H, R)
ang_cs  = cumsum(v, dim=L)                                    (B, L, H, R)
ang_cs  = ang_cs − 2π · floor(ang_cs / 2π)                    wrap into [0, 2π)
ang_pad = pad ang_cs with zeros to (B, L, H, S/2)             pad_dim = S/2 − R = 32
cos_b   = cos(ang_pad)                                        (B, L, H, S/2)
sin_b   = sin(ang_pad)                                        (B, L, H, S/2)

K_rot   = rope_pair(K_pre, cos_b, sin_b)                      (B, L, H, S)
Q_rot   = clamp_safe( rope_pair(Q_pre, cos_b, sin_b) )        (B, L, H, S)
K_scaled= clamp_safe( K_rot · scale[:,:,:,None] )             (B, L, H, S)
```

`rope_pair(x, c, s)` views the last dim of `x` as `S/2` 2-vectors and rotates
each pair:

```
x_pair = reshape(x, (..., S/2, 2))
o0 = x_pair[..., 0] · c − x_pair[..., 1] · s
o1 = x_pair[..., 0] · s + x_pair[..., 1] · c
y  = reshape(stack(o0, o1, axis=-1), (..., S))
```

### Stage 6 — selective scan core (the kernel)

`da_cs = cumsum(ADT, dim=L)`  →  `(B, L, H)`.

The strictly-causal off-diagonal output is

```
out_off[b, t, h, p] = Σ_{t' < t} exp(da_cs[b,t,h] - da_cs[b,t',h])
                              · ⟨Q_rot[b,t,h,:], K_scaled[b,t',h,:]⟩
                              · V[b,t',h,p]
```

with `⟨·,·⟩` the inner product over the state dim S. The exponent argument
must be clamped to `[-60.0, 0.0]` before applying `exp` to suppress overflow
and to enforce causality decay.

The reference `SimpleSSM._forward_dense` computes this as the unfused dense
matmul. A fused implementation must produce the same `out_off` up to the
tolerances below.

### Stage 7 — post-mix

```
out  = out_off + (D[None,None,:,None] + qk_diag[:,:,:,None]) · V        (B, L, H, P)
out  = out · silu(z)                                                    (B, L, H, P)
out  = reshape(out, (B, L, d_inner))
y    = clamp_safe( out · out_proj.weightᵀ )                             (B, L, d_model)
```

## Tolerances

Reference: `SimpleSSM._forward_dense` running in fp32 on CPU.

| Implementation              | abs tol  | rel L2 tol |
|-----------------------------|----------|------------|
| `SimpleSSM` chunked (eager) | 5e-4     | 5e-4       |
| ONNX-exported SimpleSSM     | 5e-4     | 5e-4       |
| `mamba_ssm.Mamba3` (CUDA)   | 5e-3     | 5e-3       |
| Fused custom op (CPU/GPU)   | 5e-4     | 5e-4       |

The Mamba3 CUDA reference uses a different chunk decomposition and slightly
different `tanh_approx`; its tolerance is intentionally looser. The mandatory
target for any new fused kernel is to match `_forward_dense` (i.e. the spec
above) to 5e-4 fp32, **not** to match Mamba3 exactly.

## ONNX custom op signature

The op `com.diffsinger::SSMSelectiveScan` takes 7 inputs and returns 1.

```
inputs:
  Q       : (B, L, H, S)   fp32   = Q_rot       (post-RoPE, post clamp)
  K       : (B, L, H, S)   fp32   = K_scaled    (post-RoPE, post-scale, post clamp)
  V       : (B, L, H, P)   fp32                                          
  ADT     : (B, L, H)      fp32   = clipped A·Δt                          (unused if da_cs given)
  da_cs   : (B, L, H)      fp32   = cumsum(ADT, dim=L)                   
  qk_diag : (B, L, H)      fp32   = (Q_pre·K_pre).sum(-1) · γ            
  D       : (H,)           fp32   = D parameter                          

attributes:
  causal       : int = 1            (must be 1 in this spec)
  exp_clip_lo  : float = -60.0
  exp_clip_hi  : float = 0.0

output:
  Y       : (B, L, H, P)   fp32   = out_off + (D + qk_diag) · V
```

The op produces only the `out_off + (D + qk_diag) · V` term. The exporter is
responsible for emitting `silu(z)`, the inner reshape and `out_proj`. This
keeps the op stateless and fusable later without re-baking the post-mix.

## Implementation notes

- **CPU custom op (p2-cpu)**. Tile L by 64, accumulate state-prefix sums in
  fp32, expose AVX2 path that vectorises the S=128 inner reduction as 16
  fp32 lanes × 8. Multi-head dimension H=8 is the OpenMP parallel axis. No
  AVX-512 dependency.
- **GPU custom op (p2-cuda)**. Use the chunked SSD recurrence (chunk=64),
  one warp per (batch, head) tile. Avoid materialising the (L,L) decay
  matrix. Exact same tolerance target as CPU.
- **Reference baseline**. SimpleSSM dense path in `modules/commons/ssm_layers.py`.
  Already validated against `mamba_ssm.Mamba3` to relL2 ≈ 3.4e-3 fp32 and
  against the chunked path to ≈ 1.2e-4 fp32.

## Verification points

- A **parity test** must compare Q,K,V,ADT,da_cs,qk_diag,D against the
  reference for the same random weights and inputs at L ∈ {128, 256, 512,
  1024}.
- A **perf gate** must show the fused op at L=512 ≤ 60 ms on a single
  modern x86 core, ≤ 5 ms on a desktop NVIDIA GPU, both fp32.
- An **end-to-end test** must re-export `aco_testssm2` with the SSM-fast
  flag, then validate that mel RMSE vs the dense ONNX is ≤ 1e-3 over the
  reference voicebank phrase.