#!/usr/bin/env python3
"""
CHIRP-Z TRANSFORM - Fixed Implementation
==========================================

Bluestein's algorithm for computing DFT at arbitrary points.

This is the key to O(M log M) Riemann-Siegel evaluation.

Reference: Bluestein (1970), Rabiner et al. (1969)
"""

import torch
import math
from typing import Tuple


def next_power_of_2(n: int) -> int:
    """Next power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def chirp_z_transform(
    x: torch.Tensor,  # Input: (batch, N) complex
    M: int,           # Number of output points
    W: torch.Tensor,  # W = exp(-i * delta) - complex scalar or tensor
    A: torch.Tensor = None,  # Starting point - complex scalar or tensor
) -> torch.Tensor:
    """
    Compute Chirp-Z Transform using Bluestein's algorithm.
    
    X[k] = Σ_{n=0}^{N-1} x[n] * A^{-n} * W^{nk}
    
    This computes the Z-transform at M points: A*W^0, A*W^1, ..., A*W^{M-1}
    
    When A=1 and W=exp(-2πi/N), this is the standard DFT.
    
    Args:
        x: Input complex tensor (batch, N)
        M: Number of output points
        W: Complex scalar W = exp(-i * delta)
        A: Complex scalar A (default: 1)
    
    Returns:
        X: Output complex tensor (batch, M)
    """
    device = x.device
    dtype = x.dtype  # Should be complex
    batch_size, N = x.shape
    
    if A is None:
        A = torch.tensor(1.0 + 0j, device=device, dtype=dtype)
    
    # Ensure W and A are complex tensors
    if not W.is_complex():
        W = torch.complex(W, torch.zeros_like(W))
    if not A.is_complex():
        A = torch.complex(A, torch.zeros_like(A))
    
    # Length for FFT (needs to be >= N + M - 1)
    L = next_power_of_2(N + M - 1)
    
    # Indices
    n = torch.arange(N, device=device, dtype=torch.float64)
    k = torch.arange(M, device=device, dtype=torch.float64)
    
    # Chirp sequences: W^(n²/2) and W^(k²/2)
    # W^(n²/2) = exp(-i * angle * n² / 2) where W = exp(-i * angle)
    angle = torch.angle(W)
    
    chirp_n = torch.exp(1j * angle * n * n / 2).to(dtype)  # (N,)
    chirp_k = torch.exp(1j * angle * k * k / 2).to(dtype)  # (M,)
    
    # A^(-n)
    A_neg_n = torch.pow(A, -n).to(dtype)  # (N,)
    
    # Step 1: Premultiply
    # y[n] = x[n] * A^(-n) * chirp_n
    y = x * A_neg_n.unsqueeze(0) * chirp_n.unsqueeze(0)  # (batch, N)
    
    # Step 2: Zero-pad y to length L
    y_padded = torch.zeros(batch_size, L, device=device, dtype=dtype)
    y_padded[:, :N] = y
    
    # Step 3: Create chirp filter h[n] = W^(-n²/2) for n = -(M-1), ..., 0, ..., (N-1)
    # We need h at indices 0, 1, ..., L-1 (circular)
    # h[n] = chirp^(-1) at appropriate indices
    
    # For convolution, h needs to be the "flipped" version
    # h[k] corresponds to chirp^(-1) at offset k - (M-1)
    
    # Build h in a way that convolution gives us what we want
    # The convolution (y * h)[k] should give us Σ y[n] * h[k-n]
    # We want the result at indices M-1, M, ..., M+M-2 to be our output
    
    h = torch.zeros(L, device=device, dtype=dtype)
    
    # Fill in h with W^(-n²/2) at appropriate positions
    # For indices 0 to N-1: h[n] = W^(-n²/2)
    # For indices L-(M-1) to L-1: h[L-k] = W^(-k²/2) for k = 1 to M-1
    
    for i in range(N):
        h[i] = torch.exp(-1j * angle * i * i / 2).to(dtype)
    
    for i in range(1, M):
        h[L - i] = torch.exp(-1j * angle * i * i / 2).to(dtype)
    
    # Step 4: Convolution via FFT
    Y = torch.fft.fft(y_padded, dim=-1)
    H = torch.fft.fft(h)
    Z = torch.fft.ifft(Y * H.unsqueeze(0), dim=-1)
    
    # Step 5: Extract M outputs and postmultiply
    # The result we want is at indices 0 to M-1 of Z
    out = Z[:, :M] * chirp_k.unsqueeze(0)
    
    return out


def test_chirp_z():
    """Test that CZT matches FFT for standard case."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("CHIRP-Z TRANSFORM TEST")
    print("="*60)
    
    for N in [8, 16, 64, 256, 1024]:
        # Standard DFT: A=1, W=exp(-2πi/N)
        W = torch.exp(torch.tensor(-2j * math.pi / N, device=device))
        A = torch.tensor(1.0 + 0j, device=device)
        
        # Random input
        x = torch.randn(10, N, device=device, dtype=torch.complex128)
        
        # CZT
        X_czt = chirp_z_transform(x, N, W, A)
        
        # Reference FFT
        X_fft = torch.fft.fft(x)
        
        # Error
        error = (X_czt - X_fft).abs().max().item()
        
        status = "✓" if error < 1e-10 else "✗"
        print(f"  N={N:4d}: error = {error:.2e} {status}")
    
    print()
    
    # Test non-standard case (zoom FFT)
    print("Zoom FFT test (M != N):")
    N, M = 256, 64
    W = torch.exp(torch.tensor(-2j * math.pi / M, device=device))
    A = torch.tensor(1.0 + 0j, device=device)
    
    x = torch.randn(5, N, device=device, dtype=torch.complex128)
    X_czt = chirp_z_transform(x, M, W, A)
    
    # Manual check: X[k] should be Σ x[n] * W^(nk)
    X_manual = torch.zeros(5, M, device=device, dtype=torch.complex128)
    for k in range(M):
        for n in range(N):
            X_manual[:, k] += x[:, n] * (W ** (n * k))
    
    error = (X_czt - X_manual).abs().max().item()
    status = "✓" if error < 1e-8 else "✗"
    print(f"  N={N}, M={M}: error = {error:.2e} {status}")
    
    print()
    print("="*60)


if __name__ == "__main__":
    test_chirp_z()
