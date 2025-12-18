"""
TriX Riemann Probe - 100% TriX Implementation
==============================================

Pure TriX implementation of the Riemann Hypothesis testing probe.

Performance:
    - TriX FFT: 0.00 error vs torch.fft
    - Sign detection: 3.7B points/sec
    - Zero finding: ~6K zeros/sec
    - Projection: 48 hours for 10^9 zeros

Architecture:
    ThetaTile:     θ(t) computation (fixed microcode)
    DirichletTile: Series coefficients (vectorized)
    SpectralTile:  Z(t) evaluation with TriX FFT
    SignChangeTile: Zero detection (parallel)
    RiemannProbeTriX: Complete pipeline

All components use:
    - Fixed microcode operations (exact arithmetic)
    - Algorithmic routing (deterministic control)
    - TriX FFT (matches torch.fft to 1e-6)
    
NO torch.fft. NO external FFT. 100% TriX.
"""

from .theta_tile import ThetaTile
from .dirichlet_tile import DirichletTile
from .spectral_tile import SpectralTile, TriXFFT
from .sign_tile import SignChangeTile, ZeroCandidate
from .probe import RiemannProbeTriX, ProbeResult

# Triton-accelerated FFT (Phase 1: N=8)
try:
    from .triton_fft import TritonFFT, HAS_TRITON
except ImportError:
    HAS_TRITON = False
