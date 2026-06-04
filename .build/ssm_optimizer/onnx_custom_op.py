"""ONNX Runtime Custom Operator for SSM

This module provides a high-performance SSM selective scan implementation
as an ONNX Runtime custom operator.

Usage:
    import onnxruntime as ort
    from ssm_optimizer.onnx_custom_op import register_ssm_custom_ops
    
    # Register custom ops
    register_ssm_custom_ops()
    
    # Use in ONNX Runtime
    session = ort.InferenceSession("model.onnx")
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Path to the custom op library
CUSTOM_OP_PATH = Path("C:/Users/Asus/Documents/OpenUtau/Dependencies/SSM/SSMOptimizer.dll")


def register_ssm_custom_ops():
    """Register SSM custom ops with ONNX Runtime
    
    This function sets up the environment for ONNX Runtime to use
    the optimized SSM selective scan implementation.
    """
    if not CUSTOM_OP_PATH.exists():
        raise FileNotFoundError(
            f"SSM custom op library not found at {CUSTOM_OP_PATH}. "
            "Please build and install the SSM optimizer first."
        )
    
    # Set environment variable for ONNX Runtime
    os.environ["ORT_CUSTOM_OPS_LIB"] = str(CUSTOM_OP_PATH)
    
    return True


def is_ssm_optimizer_available() -> bool:
    """Check if SSM optimizer is installed and available"""
    return CUSTOM_OP_PATH.exists()


def get_ssm_optimizer_version() -> Optional[str]:
    """Get the version of the installed SSM optimizer"""
    if not is_ssm_optimizer_available():
        return None
    
    # Try to load the DLL and get version
    try:
        import ctypes
        lib = ctypes.CDLL(str(CUSTOM_OP_PATH))
        # Assuming the library exports a version function
        # lib.GetVersion.restype = ctypes.c_char_p
        # return lib.GetVersion().decode('utf-8')
        return "1.0.0"  # Placeholder
    except Exception:
        return None


class SSMCustomOpConfig:
    """Configuration for SSM custom operator"""
    
    def __init__(self):
        self.use_simd = True
        self.use_openmp = True
        self.chunk_size = 64
        self.num_threads = os.cpu_count() or 4
    
    def to_dict(self):
        return {
            "use_simd": self.use_simd,
            "use_openmp": self.use_openmp,
            "chunk_size": self.chunk_size,
            "num_threads": self.num_threads,
        }
