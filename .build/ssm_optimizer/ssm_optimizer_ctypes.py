"""
SSM Optimizer - Python ctypes wrapper

This module provides a Python interface to the SSMOptimizer.dll
using ctypes for integration with DiffSinger/ONNX Runtime.
"""

import ctypes
import os
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

# Try to find the DLL
def _find_dll():
    """Find SSMOptimizer.dll in common locations."""
    # Check if in OpenUtau Dependencies
    openutau_path = Path("C:/Users/Asus/Documents/OpenUtau/Dependencies/SSM")
    if (openutau_path / "SSMOptimizer.dll").exists():
        return str(openutau_path / "SSMOptimizer.dll")
    
    # Check current directory
    current = Path(__file__).parent / "cpp" / "build" / "SSMOptimizer.dll"
    if current.exists():
        return str(current)
    
    # Check if in PATH
    try:
        dll = ctypes.CDLL("SSMOptimizer.dll")
        return "SSMOptimizer.dll"
    except OSError:
        pass
    
    return None


class SSMConfig(ctypes.Structure):
    """Configuration structure matching C API."""
    _fields_ = [
        ("use_simd", ctypes.c_int),
        ("use_openmp", ctypes.c_int),
        ("chunk_size", ctypes.c_int),
        ("num_threads", ctypes.c_int),
    ]
    
    def __init__(self, use_simd: bool = True, use_openmp: bool = True, 
                 chunk_size: int = 64, num_threads: int = 0):
        self.use_simd = int(use_simd)
        self.use_openmp = int(use_openmp)
        self.chunk_size = chunk_size
        self.num_threads = num_threads


class SSMOptimizer:
    """
    High-performance SSM selective scan optimizer.
    
    This class provides an optimized CPU implementation of the SSM selective scan
    algorithm using SIMD (AVX2/AVX512) and OpenMP parallelization.
    """
    
    _instance = None
    
    def __new__(cls, dll_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, dll_path: Optional[str] = None):
        if self._initialized:
            return
        
        # Load DLL
        if dll_path is None:
            dll_path = _find_dll()
        
        if dll_path is None or not os.path.exists(dll_path):
            raise RuntimeError(
                "SSMOptimizer.dll not found. Please build the C++ library first "
                "or ensure it's in C:/Users/Asus/Documents/OpenUtau/Dependencies/SSM/"
            )
        
        self._dll = ctypes.CDLL(dll_path)
        self._setup_functions()
        self._initialized = True
    
    def _setup_functions(self):
        """Setup ctypes function signatures."""
        # Version functions
        self._dll.SSM_GetVersion.restype = ctypes.c_char_p
        self._dll.SSM_GetMajorVersion.restype = ctypes.c_int
        self._dll.SSM_GetMinorVersion.restype = ctypes.c_int
        self._dll.SSM_GetPatchVersion.restype = ctypes.c_int
        
        # Config functions
        self._dll.SSM_GetDefaultConfig.argtypes = [ctypes.POINTER(SSMConfig)]
        self._dll.SSM_SetGlobalConfig.argtypes = [ctypes.POINTER(SSMConfig)]
        
        # Selective scan function
        self._dll.SSM_SelectiveScan_F32.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # dt
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.POINTER(ctypes.c_float),  # D (optional)
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # seq_len
            ctypes.c_int,  # d_inner
            ctypes.c_int,  # n_heads
            ctypes.c_int,  # d_state
            ctypes.c_int,  # head_dim
        ]
        self._dll.SSM_SelectiveScan_F32.restype = ctypes.c_int
    
    @property
    def version(self) -> str:
        """Get library version string."""
        return self._dll.SSM_GetVersion().decode('utf-8')
    
    @property
    def version_tuple(self) -> Tuple[int, int, int]:
        """Get library version as tuple."""
        return (
            self._dll.SSM_GetMajorVersion(),
            self._dll.SSM_GetMinorVersion(),
            self._dll.SSM_GetPatchVersion()
        )
    
    def get_default_config(self) -> SSMConfig:
        """Get default configuration."""
        config = SSMConfig()
        self._dll.SSM_GetDefaultConfig(ctypes.byref(config))
        return config
    
    def set_config(self, config: SSMConfig):
        """Set global configuration."""
        self._dll.SSM_SetGlobalConfig(ctypes.byref(config))
    
    def selective_scan(
        self,
        input: np.ndarray,
        dt: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Execute optimized selective scan.
        
        Args:
            input: Input tensor [batch, seq_len, d_inner]
            dt: Time delta [batch, seq_len, n_heads]
            A: State transition [n_heads, d_state]
            B: Input matrix [batch, seq_len, d_state]
            C: Output matrix [batch, seq_len, d_state]
            D: Skip connection [n_heads] (optional)
            
        Returns:
            Output tensor [batch, seq_len, d_inner]
        """
        # Ensure contiguous arrays
        input = np.ascontiguousarray(input, dtype=np.float32)
        dt = np.ascontiguousarray(dt, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        C = np.ascontiguousarray(C, dtype=np.float32)
        
        batch_size, seq_len, d_inner = input.shape
        n_heads, d_state = A.shape
        head_dim = d_inner // n_heads
        
        output = np.empty((batch_size, seq_len, d_inner), dtype=np.float32)
        
        D_ptr = None
        if D is not None:
            D = np.ascontiguousarray(D, dtype=np.float32)
            D_ptr = D.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        ret = self._dll.SSM_SelectiveScan_F32(
            input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            D_ptr,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size,
            seq_len,
            d_inner,
            n_heads,
            d_state,
            head_dim
        )
        
        if ret != 0:
            raise RuntimeError(f"SSM selective scan failed with error code {ret}")
        
        return output


class SimpleSSMReference:
    """
    Pure NumPy reference implementation of SimpleSSM.
    
    This matches the PyTorch SimpleSSM implementation for testing.
    """
    
    @staticmethod
    def selective_scan(
        input: np.ndarray,
        dt: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reference implementation of selective scan.
        
        Computes: h_t = A_bar * h_{t-1} + B_bar * x_t
                  y_t = C_t * h_t + D * x_t
        """
        batch_size, seq_len, d_inner = input.shape
        n_heads, d_state = A.shape
        head_dim = d_inner // n_heads
        
        output = np.zeros_like(input)
        
        for b in range(batch_size):
            for h in range(n_heads):
                h_state = np.zeros(d_state, dtype=np.float32)
                
                for t in range(seq_len):
                    x_t = input[b, t, h * head_dim:(h + 1) * head_dim]
                    dt_t = dt[b, t, h]
                    B_t = B[b, t]
                    C_t = C[b, t]
                    
                    # Compute A_bar and B_bar
                    A_bar = np.exp(dt_t * A[h])
                    B_bar = dt_t * B_t
                    
                    # Update state
                    x_sum = np.mean(x_t)
                    h_state = A_bar * h_state + B_bar * x_sum
                    
                    # Compute output
                    y_val = np.sum(C_t * h_state)
                    
                    # Add skip connection
                    if D is not None:
                        y_val = y_val + D[h] * x_t
                    else:
                        y_val = np.full(head_dim, y_val, dtype=np.float32)
                    
                    output[b, t, h * head_dim:(h + 1) * head_dim] = y_val
        
        return output


def is_available() -> bool:
    """Check if SSMOptimizer DLL is available."""
    return _find_dll() is not None


def get_optimizer() -> Optional[SSMOptimizer]:
    """Get SSM optimizer instance if available."""
    try:
        return SSMOptimizer()
    except RuntimeError:
        return None


# Convenience function for direct usage
def selective_scan(
    input: np.ndarray,
    dt: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: Optional[np.ndarray] = None,
    use_optimized: bool = True
) -> np.ndarray:
    """
    Execute selective scan with optional optimization.
    
    Args:
        input: Input tensor [batch, seq_len, d_inner]
        dt: Time delta [batch, seq_len, n_heads]
        A: State transition [n_heads, d_state]
        B: Input matrix [batch, seq_len, d_state]
        C: Output matrix [batch, seq_len, d_state]
        D: Skip connection [n_heads] (optional)
        use_optimized: Use C++ optimized implementation if available
        
    Returns:
        Output tensor [batch, seq_len, d_inner]
    """
    if use_optimized:
        optimizer = get_optimizer()
        if optimizer is not None:
            return optimizer.selective_scan(input, dt, A, B, C, D)
    
    # Fall back to reference implementation
    return SimpleSSMReference.selective_scan(input, dt, A, B, C, D)
