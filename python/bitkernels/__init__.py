"""
BitKernels: Optimized quantized neural network kernels

This package provides high-performance implementations of quantized neural network
operations, including BitLinear (ternary weight × INT8 activation).
"""

__version__ = "0.1.0"

from . import _C

# Export C++ functions
prepare_weights = _C.prepare_weights
bitlinear_forward = _C.bitlinear_forward
bitlinear_gemm = _C.bitlinear_gemm
bitlinear_gemv = _C.bitlinear_gemv
PackedWeights = _C.PackedWeights

__all__ = [
    "prepare_weights",
    "bitlinear_forward",
    "bitlinear_gemm",
    "bitlinear_gemv",
    "PackedWeights",
]

