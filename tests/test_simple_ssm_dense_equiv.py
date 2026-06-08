"""
Sanity check: dense (trace path) and chunked (eager path) of SimpleSSM
must produce numerically equivalent outputs.

Run before re-exporting any onnx so we catch regressions in the prefix +
dense rewrite locally instead of waiting for an ORT roundtrip.
"""
import os, sys, math, torch, torch.nn.functional as F

# Force the SimpleSSM path even if mamba_ssm is importable on this box.
os.environ['DIFFSINGER_USE_MAMBA3'] = '0'

# 项目根 = 当前文件向上两级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.commons.ssm_layers import SimpleSSM


def main():
    torch.manual_seed(0)
    d_model, d_state, expand, headdim = 256, 128, 2, 64
    ssm = SimpleSSM(
        d_model=d_model,
        d_state=d_state,
        expand=expand,
        headdim=headdim,
        ngroups=1,
    ).eval()

    # exercise both the L < 64 (no padding) and L > 64 (padded chunks) regimes,
    # plus a length that's exactly divisible (should exercise the no-pad branch
    # of the chunked path).
    cases = [
        ('short, padded',         (1, 16)),
        ('exactly one chunk',     (1, 64)),
        ('off-grid, two chunks',  (1, 80)),
        ('clean two chunks',      (2, 128)),
        ('three chunks ish',      (1, 200)),
    ]

    print(f"{'case':<24}{'shape':<14}{'max|Δ|':<14}{'mean|Δ|':<14}")
    for label, (B, L) in cases:
        x = torch.randn(B, L, d_model)
        with torch.no_grad():
            # eager == chunked
            y_chunked = ssm(x)

            # trace == dense.  Force the dense path by trick: override jit.is_tracing
            # for the duration of one call.
            orig = torch.jit.is_tracing
            try:
                torch.jit.is_tracing = lambda: True
                y_dense = ssm(x)
            finally:
                torch.jit.is_tracing = orig

        diff = (y_dense - y_chunked).abs()
        print(f"{label:<24}{str(tuple(x.shape)):<14}"
              f"{diff.max().item():.3e}    {diff.mean().item():.3e}")
        assert diff.max().item() < 1e-3, \
            f"DENSE != CHUNKED for {label}: max|Δ|={diff.max().item():.3e}"

    print('\nOK: dense path is numerically equivalent to chunked path.')


if __name__ == '__main__':
    main()