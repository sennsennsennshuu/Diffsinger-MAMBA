"""TDD Tests for C++ vs PyTorch Selective Scan Math Consistency

Tests that the C++ selective_scan implementation matches SimpleSSM's output.
Following TDD: Write failing test first, then fix C++ code.
"""
import numpy as np
import pytest
import torch
import sys
import ctypes
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _get_dll_path():
    """Get path to SSMOptimizer.dll"""
    dll_path = Path(__file__).parent.parent / ".build" / "ssm_optimizer" / "cpp" / "build" / "Release" / "SSMOptimizer.dll"
    if not dll_path.exists():
        pytest.skip(f"SSMOptimizer.dll not found at {dll_path}")
    return str(dll_path)


def _load_dll():
    """Load SSMOptimizer.dll and return ctypes interface"""
    dll_path = _get_dll_path()
    try:
        lib = ctypes.CDLL(dll_path)
        return lib
    except OSError as e:
        pytest.skip(f"Cannot load SSMOptimizer.dll: {e}")


def _simple_ssm_reference(x_np, dt_np, A_np, B_np, C_np, D_val=0.0):
    """
    Reference implementation matching SimpleSSM's forward logic.
    
    SimpleSSM uses vectorized parallel scan:
      u = B * x  (outer product: (L, S) * (L, P) -> (L, S, P))
      h = cumprod_a * cumsum(u / cumprod_a_pad, dim=1)
      y = sum(C * h, dim=1)  # sum over state dims
    
    This is for a single head (BH=1).
    
    Args:
        x_np: (L, P) input
        dt_np: (L,) timestep
        A_np: (S,) A matrix (negative)
        B_np: (L, S) B matrix
        C_np: (L, S) C matrix
        D_val: float D skip connection
    
    Returns:
        y_np: (L, P) output
    """
    L = x_np.shape[0]
    P = x_np.shape[1]
    S = A_np.shape[0]
    
    x = torch.from_numpy(x_np).float()
    dt = torch.from_numpy(dt_np).float().squeeze()  # ensure (L,)
    A = torch.from_numpy(A_np).float()
    B = torch.from_numpy(B_np).float()
    C = torch.from_numpy(C_np).float()
    
    dt_h = dt.unsqueeze(-1)  # (L, 1)
    A_h = A.unsqueeze(0)  # (1, S)
    
    y = torch.zeros(L, P)
    
    # Process each state dimension (CHUNK=1 like SimpleSSM)
    for s in range(S):
        # log_a_t = dt * A_s: (L, 1)
        log_a_t = dt_h * A_h[0, s]  # (L, 1)
        cumsum_log_a = torch.cumsum(log_a_t, dim=0)  # (L, 1)
        cumprod_a = torch.exp(cumsum_log_a)  # (L, 1)
        
        zeros = torch.zeros(1, 1)
        cumprod_a_pad = torch.exp(torch.cat([zeros, cumsum_log_a[:-1, :]], dim=0))  # (L, 1)
        cumprod_a_pad = cumprod_a_pad.clamp(min=1e-12)
        
        # u = B_s * x: (L, P)
        u = B[:, s:s+1] * x  # (L, 1) * (L, P) = (L, P)
        
        # h = cumprod_a * cumsum(u / cumprod_a_pad, dim=0)
        h_c = cumprod_a * torch.cumsum(u / cumprod_a_pad, dim=0)  # (L, P)
        
        # y += C_s * h
        y = y + C[:, s:s+1] * h_c  # (L, 1) * (L, P) = (L, P)
    
    # D skip connection
    y = y + D_val * x
    
    return y.numpy()


class TestCppMathConsistency:
    """Test C++ selective_scan matches SimpleSSM math exactly"""
    
    def test_cpp_matches_pytorch_single_head(self):
        """C++ selective_scan output must match SimpleSSM reference
        
        This test FAILS with current C++ implementation because
        C++ uses x_t[s % head_dim] (scalar) instead of x_t (vector).
        """
        lib = _load_dll()
        
        # Set up function signature
        # void SSM_SelectiveScan_F32(
        #   const float* x, const float* dt, const float* A,
        #   const float* B, const float* C, const float* D,
        #   float* output,
        #   int batch, int seq_len, int d_inner, int n_heads,
        #   int d_state, int head_dim, int use_simd)
        lib.SSM_SelectiveScan_F32.restype = None
        lib.SSM_SelectiveScan_F32.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # dt
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.POINTER(ctypes.c_float),  # D
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,  # batch
            ctypes.c_int,  # seq_len
            ctypes.c_int,  # d_inner
            ctypes.c_int,  # n_heads
            ctypes.c_int,  # d_state
            ctypes.c_int,  # head_dim
            ctypes.c_int,  # use_simd
        ]
        
        # Test parameters (matching SimpleSSM defaults)
        batch = 1
        seq_len = 100
        d_model = 256
        d_inner = 512  # expand=2
        n_heads = 8
        head_dim = d_inner // n_heads  # 64
        d_state = 128
        
        # Generate random inputs
        np.random.seed(42)
        x_d_model = np.random.randn(batch, seq_len, d_model).astype(np.float32)  # (B,L,d_model) for SimpleSSM
        x_d_inner = np.random.randn(batch, seq_len, d_inner).astype(np.float32)  # (B,L,d_inner) for C++
        dt = np.random.uniform(0.1, 5.0, (batch, seq_len, n_heads)).astype(np.float32)
        A = -np.exp(np.random.randn(n_heads, d_state).astype(np.float32))  # A_neg
        B = np.random.randn(batch, seq_len, d_state).astype(np.float32)
        C = np.random.randn(batch, seq_len, d_state).astype(np.float32)
        D = np.random.randn(n_heads).astype(np.float32)
        
        # PyTorch reference (using SimpleSSM model with correct d_model input)
        from modules.commons.ssm_layers import SimpleSSM
        model = SimpleSSM(d_model=d_model, d_state=d_state, expand=2, headdim=head_dim, ngroups=1)
        model.eval()
        
        x_torch = torch.from_numpy(x_d_model).float()
        with torch.no_grad():
            ref_output = model(x_torch).numpy()
        
        # C++ output
        output = np.zeros_like(x_d_inner)
        
        x_ptr = x_d_inner.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dt_ptr = dt.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        D_ptr = D.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        lib.SSM_SelectiveScan_F32(
            x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, out_ptr,
            batch, seq_len, d_inner, n_heads, d_state, head_dim, 0  # use_simd=0
        )
        
        # Compare
        max_diff = np.max(np.abs(ref_output - output))
        
        # Tolerance: 1e-3 for float32 precision differences
        assert max_diff < 1e-3, \
            f"C++ output differs from PyTorch: max_diff={max_diff:.6f} (tolerance: 1e-3)"
    
    def test_cpp_matches_pytorch_core_scan(self):
        """Test core scan loop matches (without projection layers)
        
        Isolates the selective scan math from input/output projections.
        Uses the same weights for both implementations.
        """
        lib = _load_dll()
        
        lib.SSM_SelectiveScan_F32.restype = None
        lib.SSM_SelectiveScan_F32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        
        # Small test case: single head
        batch = 1
        seq_len = 50
        d_inner = 64  # single head
        n_heads = 1
        head_dim = 64
        d_state = 16  # small for clarity
        
        np.random.seed(123)
        x = np.random.randn(batch, seq_len, d_inner).astype(np.float32)
        dt = np.random.uniform(0.1, 5.0, (batch, seq_len, n_heads)).astype(np.float32)
        A = -np.exp(np.random.randn(n_heads, d_state).astype(np.float32))
        B = np.random.randn(batch, seq_len, d_state).astype(np.float32)
        C = np.random.randn(batch, seq_len, d_state).astype(np.float32)
        D = np.random.randn(n_heads).astype(np.float32)
        
        # PyTorch reference (core scan only)
        ref_y = _simple_ssm_reference(
            x[0], dt[0], A[0], B[0], C[0], D_val=float(D[0])
        )
        
        # C++ output
        output = np.zeros_like(x)
        lib.SSM_SelectiveScan_F32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            D.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch, seq_len, d_inner, n_heads, d_state, head_dim, 0
        )
        
        max_diff = np.max(np.abs(ref_y - output[0]))
        mean_ref = np.mean(np.abs(ref_y))
        
        # Check relative error (< 5%) since parallel vs sequential scan
        # has different numerical precision characteristics
        rel_err = max_diff / (mean_ref + 1e-8)
        assert rel_err < 0.05, \
            f"Core scan mismatch: max_diff={max_diff:.6f}, rel_err={rel_err:.4f} (tolerance: 5%)"
    
    def test_cpp_simd_matches_scalar(self):
        """SIMD version should produce same result as scalar version"""
        lib = _load_dll()
        
        lib.SSM_SelectiveScan_F32.restype = None
        lib.SSM_SelectiveScan_F32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        
        batch = 1
        seq_len = 100
        d_inner = 256
        n_heads = 4
        head_dim = 64
        d_state = 64
        
        np.random.seed(456)
        x = np.random.randn(batch, seq_len, d_inner).astype(np.float32)
        dt = np.random.uniform(0.1, 5.0, (batch, seq_len, n_heads)).astype(np.float32)
        A = -np.exp(np.random.randn(n_heads, d_state).astype(np.float32))
        B = np.random.randn(batch, seq_len, d_state).astype(np.float32)
        C = np.random.randn(batch, seq_len, d_state).astype(np.float32)
        D = np.random.randn(n_heads).astype(np.float32)
        
        # Scalar version
        output_scalar = np.zeros_like(x)
        lib.SSM_SelectiveScan_F32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            D.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_scalar.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch, seq_len, d_inner, n_heads, d_state, head_dim, 0  # use_simd=0
        )
        
        # SIMD version
        output_simd = np.zeros_like(x)
        lib.SSM_SelectiveScan_F32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            D.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_simd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch, seq_len, d_inner, n_heads, d_state, head_dim, 1  # use_simd=1
        )
        
        max_diff = np.max(np.abs(output_scalar - output_simd))
        assert max_diff < 1e-5, \
            f"SIMD vs scalar mismatch: max_diff={max_diff:.6f} (tolerance: 1e-5)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
