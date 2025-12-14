"""
TriX Kernel

Sparse 2-bit matrix multiplication with tile-based routing.
Implements ARM NEON acceleration for ternary weights.
"""

from .bindings import (
    TriXLinear,
    STESign,
    pack_weights,
    unpack_weights,
    trix_forward,
    is_neon_available,
)

__all__ = [
    "TriXLinear",
    "STESign",
    "pack_weights",
    "unpack_weights",
    "trix_forward",
    "is_neon_available",
]
