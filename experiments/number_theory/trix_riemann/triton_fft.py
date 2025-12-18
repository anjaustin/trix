#!/usr/bin/env python3
"""
TritonFFT - Compiled TriX for GPU
==================================

Triton-accelerated FFT with TriX philosophy:
- Precomputed twiddle table (no runtime trig)
- Cooley-Tukey DIT algorithm
- Structural routing compiled to index calculations

This is TriX compiled to silicon. Same algorithm, same twiddles,
different execution engine.

Phase 1: N=8 Proof of Concept
"""

import torch
import torch.nn as nn
import math
from typing import Tuple

# Check for Triton availability
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: Triton not available. TritonFFT will fall back to Python.")


# =============================================================================
# TWIDDLE TABLE (Precomputed - No Runtime Trig)
# =============================================================================

def compute_twiddle_table(N: int, device='cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute twiddle factors W_N^k = e^{-2πik/N}.
    
    These are the EXACT constants from our twiddle opcode philosophy.
    No np.cos/np.sin at runtime - computed once, used forever.
    
    Returns:
        W_re: cos(-2πk/N) for k=0..N-1
        W_im: sin(-2πk/N) for k=0..N-1
    """
    k = torch.arange(N, dtype=torch.float32, device=device)
    angles = -2.0 * math.pi * k / N
    W_re = torch.cos(angles)
    W_im = torch.sin(angles)
    return W_re, W_im


def compute_bitrev_indices(N: int, device='cuda') -> torch.Tensor:
    """
    Compute bit-reversal permutation indices.
    
    This is STRUCTURAL routing - determined entirely by position.
    """
    num_bits = int(math.log2(N))
    indices = torch.zeros(N, dtype=torch.int32, device=device)
    
    for i in range(N):
        rev = 0
        val = i
        for _ in range(num_bits):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        indices[i] = rev
    
    return indices


# =============================================================================
# TRITON KERNEL - N=8 FFT
# =============================================================================

if HAS_TRITON:
    
    @triton.jit
    def trix_fft_n8_kernel(
        # Input pointers
        X_re_ptr, X_im_ptr,
        # Output pointers
        Y_re_ptr, Y_im_ptr,
        # Twiddle table pointers
        W_re_ptr, W_im_ptr,
        # Batch stride
        batch_stride,
        # Number of batches
        num_batches,
    ):
        """
        N=8 Cooley-Tukey DIT FFT kernel.
        
        Each program handles one FFT in the batch.
        All 8 values loaded to registers, 3 stages computed, result stored.
        
        This is the TriX router compiled to silicon:
        - Twiddle selection = index calculation
        - Butterfly = exact arithmetic
        - No Python, no interpreter, pure GPU
        """
        # Which FFT in the batch?
        batch_idx = tl.program_id(0)
        
        if batch_idx >= num_batches:
            return
        
        # Base offset for this batch
        base = batch_idx * batch_stride
        
        # Load all 8 values (bit-reversed order built into load pattern)
        # Bit-reversal for N=8: [0,4,2,6,1,5,3,7]
        x0_re = tl.load(X_re_ptr + base + 0)
        x1_re = tl.load(X_re_ptr + base + 4)
        x2_re = tl.load(X_re_ptr + base + 2)
        x3_re = tl.load(X_re_ptr + base + 6)
        x4_re = tl.load(X_re_ptr + base + 1)
        x5_re = tl.load(X_re_ptr + base + 5)
        x6_re = tl.load(X_re_ptr + base + 3)
        x7_re = tl.load(X_re_ptr + base + 7)
        
        x0_im = tl.load(X_im_ptr + base + 0)
        x1_im = tl.load(X_im_ptr + base + 4)
        x2_im = tl.load(X_im_ptr + base + 2)
        x3_im = tl.load(X_im_ptr + base + 6)
        x4_im = tl.load(X_im_ptr + base + 1)
        x5_im = tl.load(X_im_ptr + base + 5)
        x6_im = tl.load(X_im_ptr + base + 3)
        x7_im = tl.load(X_im_ptr + base + 7)
        
        # Load twiddle factors (W_8^k for k=0..7)
        W0_re = tl.load(W_re_ptr + 0)  # W^0 = 1
        W0_im = tl.load(W_im_ptr + 0)
        W1_re = tl.load(W_re_ptr + 1)  # W^1 = (1-i)/sqrt(2)
        W1_im = tl.load(W_im_ptr + 1)
        W2_re = tl.load(W_re_ptr + 2)  # W^2 = -i
        W2_im = tl.load(W_im_ptr + 2)
        W3_re = tl.load(W_re_ptr + 3)  # W^3 = (-1-i)/sqrt(2)
        W3_im = tl.load(W_im_ptr + 3)
        
        # =====================================================================
        # STAGE 0: stride=1, groups of 2
        # Butterflies: (0,1), (2,3), (4,5), (6,7)
        # All use W^0 = 1
        # =====================================================================
        
        # Butterfly (0,1) with W^0
        t_re = x1_re * W0_re - x1_im * W0_im
        t_im = x1_re * W0_im + x1_im * W0_re
        y0_re = x0_re + t_re
        y0_im = x0_im + t_im
        y1_re = x0_re - t_re
        y1_im = x0_im - t_im
        
        # Butterfly (2,3) with W^0
        t_re = x3_re * W0_re - x3_im * W0_im
        t_im = x3_re * W0_im + x3_im * W0_re
        y2_re = x2_re + t_re
        y2_im = x2_im + t_im
        y3_re = x2_re - t_re
        y3_im = x2_im - t_im
        
        # Butterfly (4,5) with W^0
        t_re = x5_re * W0_re - x5_im * W0_im
        t_im = x5_re * W0_im + x5_im * W0_re
        y4_re = x4_re + t_re
        y4_im = x4_im + t_im
        y5_re = x4_re - t_re
        y5_im = x4_im - t_im
        
        # Butterfly (6,7) with W^0
        t_re = x7_re * W0_re - x7_im * W0_im
        t_im = x7_re * W0_im + x7_im * W0_re
        y6_re = x6_re + t_re
        y6_im = x6_im + t_im
        y7_re = x6_re - t_re
        y7_im = x6_im - t_im
        
        # =====================================================================
        # STAGE 1: stride=2, groups of 4
        # Butterflies: (0,2), (1,3), (4,6), (5,7)
        # (0,2) uses W^0, (1,3) uses W^2
        # =====================================================================
        
        # Butterfly (0,2) with W^0
        t_re = y2_re * W0_re - y2_im * W0_im
        t_im = y2_re * W0_im + y2_im * W0_re
        z0_re = y0_re + t_re
        z0_im = y0_im + t_im
        z2_re = y0_re - t_re
        z2_im = y0_im - t_im
        
        # Butterfly (1,3) with W^2
        t_re = y3_re * W2_re - y3_im * W2_im
        t_im = y3_re * W2_im + y3_im * W2_re
        z1_re = y1_re + t_re
        z1_im = y1_im + t_im
        z3_re = y1_re - t_re
        z3_im = y1_im - t_im
        
        # Butterfly (4,6) with W^0
        t_re = y6_re * W0_re - y6_im * W0_im
        t_im = y6_re * W0_im + y6_im * W0_re
        z4_re = y4_re + t_re
        z4_im = y4_im + t_im
        z6_re = y4_re - t_re
        z6_im = y4_im - t_im
        
        # Butterfly (5,7) with W^2
        t_re = y7_re * W2_re - y7_im * W2_im
        t_im = y7_re * W2_im + y7_im * W2_re
        z5_re = y5_re + t_re
        z5_im = y5_im + t_im
        z7_re = y5_re - t_re
        z7_im = y5_im - t_im
        
        # =====================================================================
        # STAGE 2: stride=4, groups of 8
        # Butterflies: (0,4), (1,5), (2,6), (3,7)
        # Uses W^0, W^1, W^2, W^3 respectively
        # =====================================================================
        
        # Butterfly (0,4) with W^0
        t_re = z4_re * W0_re - z4_im * W0_im
        t_im = z4_re * W0_im + z4_im * W0_re
        out0_re = z0_re + t_re
        out0_im = z0_im + t_im
        out4_re = z0_re - t_re
        out4_im = z0_im - t_im
        
        # Butterfly (1,5) with W^1
        t_re = z5_re * W1_re - z5_im * W1_im
        t_im = z5_re * W1_im + z5_im * W1_re
        out1_re = z1_re + t_re
        out1_im = z1_im + t_im
        out5_re = z1_re - t_re
        out5_im = z1_im - t_im
        
        # Butterfly (2,6) with W^2
        t_re = z6_re * W2_re - z6_im * W2_im
        t_im = z6_re * W2_im + z6_im * W2_re
        out2_re = z2_re + t_re
        out2_im = z2_im + t_im
        out6_re = z2_re - t_re
        out6_im = z2_im - t_im
        
        # Butterfly (3,7) with W^3
        t_re = z7_re * W3_re - z7_im * W3_im
        t_im = z7_re * W3_im + z7_im * W3_re
        out3_re = z3_re + t_re
        out3_im = z3_im + t_im
        out7_re = z3_re - t_re
        out7_im = z3_im - t_im
        
        # Store output
        tl.store(Y_re_ptr + base + 0, out0_re)
        tl.store(Y_re_ptr + base + 1, out1_re)
        tl.store(Y_re_ptr + base + 2, out2_re)
        tl.store(Y_re_ptr + base + 3, out3_re)
        tl.store(Y_re_ptr + base + 4, out4_re)
        tl.store(Y_re_ptr + base + 5, out5_re)
        tl.store(Y_re_ptr + base + 6, out6_re)
        tl.store(Y_re_ptr + base + 7, out7_re)
        
        tl.store(Y_im_ptr + base + 0, out0_im)
        tl.store(Y_im_ptr + base + 1, out1_im)
        tl.store(Y_im_ptr + base + 2, out2_im)
        tl.store(Y_im_ptr + base + 3, out3_im)
        tl.store(Y_im_ptr + base + 4, out4_im)
        tl.store(Y_im_ptr + base + 5, out5_im)
        tl.store(Y_im_ptr + base + 6, out6_im)
        tl.store(Y_im_ptr + base + 7, out7_im)


# =============================================================================
# TRITON FFT CLASS
# =============================================================================

class TritonFFT(nn.Module):
    """
    Triton-accelerated FFT with TriX philosophy.
    
    Features:
        - Precomputed twiddle table (no runtime trig)
        - Cooley-Tukey DIT algorithm
        - Batch support
        - Drop-in replacement for TriXFFT
    
    Currently supports: N=8
    """
    
    def __init__(self):
        super().__init__()
        self._twiddle_cache = {}
    
    def _get_twiddles(self, N: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached twiddle factors."""
        key = (N, device)
        if key not in self._twiddle_cache:
            self._twiddle_cache[key] = compute_twiddle_table(N, device)
        return self._twiddle_cache[key]
    
    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FFT using Triton kernel.
        
        Args:
            x_re: Real parts, shape (..., N) where N=8
            x_im: Imaginary parts, shape (..., N)
        
        Returns:
            X_re, X_im: FFT output, same shape
        """
        if not HAS_TRITON:
            raise RuntimeError("Triton not available")
        
        *batch_dims, N = x_re.shape
        
        if N != 8:
            raise ValueError(f"Currently only N=8 supported, got N={N}")
        
        device = x_re.device
        
        # Flatten batch dimensions
        batch_size = 1
        for d in batch_dims:
            batch_size *= d
        
        x_re_flat = x_re.reshape(batch_size, N).contiguous()
        x_im_flat = x_im.reshape(batch_size, N).contiguous()
        
        # Allocate output
        y_re = torch.empty_like(x_re_flat)
        y_im = torch.empty_like(x_im_flat)
        
        # Get twiddles
        W_re, W_im = self._get_twiddles(N, device)
        
        # Launch kernel
        grid = (batch_size,)
        trix_fft_n8_kernel[grid](
            x_re_flat, x_im_flat,
            y_re, y_im,
            W_re, W_im,
            N,  # batch_stride
            batch_size,
        )
        
        # Reshape output
        y_re = y_re.reshape(*batch_dims, N)
        y_im = y_im.reshape(*batch_dims, N)
        
        return y_re, y_im


# =============================================================================
# PYTHON FALLBACK (Reference)
# =============================================================================

class PythonFFT(nn.Module):
    """Python reference FFT for comparison."""
    
    def __init__(self):
        super().__init__()
        self._twiddle_cache = {}
        self._bitrev_cache = {}
    
    def _get_twiddles(self, N, device):
        key = (N, device)
        if key not in self._twiddle_cache:
            self._twiddle_cache[key] = compute_twiddle_table(N, device)
        return self._twiddle_cache[key]
    
    def _get_bitrev(self, N, device):
        key = (N, device)
        if key not in self._bitrev_cache:
            self._bitrev_cache[key] = compute_bitrev_indices(N, device)
        return self._bitrev_cache[key]
    
    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        *batch_dims, N = x_re.shape
        device = x_re.device
        num_stages = int(math.log2(N))
        
        W_re, W_im = self._get_twiddles(N, device)
        bitrev = self._get_bitrev(N, device)
        
        # Bit-reversal permutation (clone to avoid in-place issues)
        y_re = x_re[..., bitrev].clone()
        y_im = x_im[..., bitrev].clone()
        
        # Cooley-Tukey stages
        for stage in range(num_stages):
            stride = 1 << stage
            group_size = stride << 1
            
            # Clone for this stage to avoid read-write conflicts
            new_re = y_re.clone()
            new_im = y_im.clone()
            
            for k in range(0, N, group_size):
                for j in range(stride):
                    i = k + j
                    partner = i + stride
                    
                    # Twiddle index
                    tw_idx = (j * (N // group_size)) % N
                    
                    # Get twiddle
                    w_re = W_re[tw_idx]
                    w_im = W_im[tw_idx]
                    
                    # Get values
                    a_re = y_re[..., i]
                    a_im = y_im[..., i]
                    b_re = y_re[..., partner]
                    b_im = y_im[..., partner]
                    
                    # Complex multiply: W * b
                    wb_re = w_re * b_re - w_im * b_im
                    wb_im = w_re * b_im + w_im * b_re
                    
                    # Butterfly
                    new_re[..., i] = a_re + wb_re
                    new_im[..., i] = a_im + wb_im
                    new_re[..., partner] = a_re - wb_re
                    new_im[..., partner] = a_im - wb_im
            
            y_re = new_re
            y_im = new_im
        
        return y_re, y_im


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_triton_fft():
    """Test TritonFFT against torch.fft and Python reference."""
    
    print("="*60)
    print("TRITON FFT - PHASE 1 (N=8)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Triton available: {HAS_TRITON}")
    
    if not HAS_TRITON:
        print("Skipping Triton tests - not available")
        return False
    
    triton_fft = TritonFFT()
    python_fft = PythonFFT()
    
    # Test 1: Simple input
    print("\n[Test 1] Simple input [1,0,0,0,0,0,0,0]")
    x_re = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], device=device)
    x_im = torch.zeros(8, device=device)
    
    # Triton
    Y_re_triton, Y_im_triton = triton_fft(x_re, x_im)
    
    # torch.fft reference
    x_complex = torch.complex(x_re, x_im)
    Y_torch = torch.fft.fft(x_complex)
    Y_re_torch = Y_torch.real
    Y_im_torch = Y_torch.imag
    
    error_re = (Y_re_triton - Y_re_torch).abs().max().item()
    error_im = (Y_im_triton - Y_im_torch).abs().max().item()
    
    print(f"  Triton real: {Y_re_triton.tolist()}")
    print(f"  Torch real:  {Y_re_torch.tolist()}")
    print(f"  Max error (real): {error_re:.2e}")
    print(f"  Max error (imag): {error_im:.2e}")
    
    test1_pass = max(error_re, error_im) < 1e-5
    print(f"  Passed: {test1_pass}")
    
    # Test 2: Random input
    print("\n[Test 2] Random input")
    torch.manual_seed(42)
    x_re = torch.randn(8, device=device)
    x_im = torch.randn(8, device=device)
    
    Y_re_triton, Y_im_triton = triton_fft(x_re, x_im)
    
    x_complex = torch.complex(x_re, x_im)
    Y_torch = torch.fft.fft(x_complex)
    
    error_re = (Y_re_triton - Y_torch.real).abs().max().item()
    error_im = (Y_im_triton - Y_torch.imag).abs().max().item()
    
    print(f"  Max error (real): {error_re:.2e}")
    print(f"  Max error (imag): {error_im:.2e}")
    
    test2_pass = max(error_re, error_im) < 1e-5
    print(f"  Passed: {test2_pass}")
    
    # Test 3: Batch input
    print("\n[Test 3] Batch input (100 x 8)")
    x_re = torch.randn(100, 8, device=device)
    x_im = torch.randn(100, 8, device=device)
    
    Y_re_triton, Y_im_triton = triton_fft(x_re, x_im)
    
    x_complex = torch.complex(x_re, x_im)
    Y_torch = torch.fft.fft(x_complex)
    
    error_re = (Y_re_triton - Y_torch.real).abs().max().item()
    error_im = (Y_im_triton - Y_torch.imag).abs().max().item()
    
    print(f"  Max error (real): {error_re:.2e}")
    print(f"  Max error (imag): {error_im:.2e}")
    
    test3_pass = max(error_re, error_im) < 1e-5
    print(f"  Passed: {test3_pass}")
    
    # Test 4: Compare to Python reference (bit-exact?)
    print("\n[Test 4] Compare Triton to Python reference")
    x_re = torch.randn(8, device=device)
    x_im = torch.randn(8, device=device)
    
    Y_re_triton, Y_im_triton = triton_fft(x_re, x_im)
    Y_re_python, Y_im_python = python_fft(x_re.clone(), x_im.clone())
    
    error_re = (Y_re_triton - Y_re_python).abs().max().item()
    error_im = (Y_im_triton - Y_im_python).abs().max().item()
    
    print(f"  Max error vs Python (real): {error_re:.2e}")
    print(f"  Max error vs Python (imag): {error_im:.2e}")
    
    test4_pass = max(error_re, error_im) < 1e-5
    print(f"  Passed: {test4_pass}")
    
    # Test 5: Performance benchmark
    print("\n[Test 5] Performance benchmark")
    import time
    
    x_re = torch.randn(10000, 8, device=device)
    x_im = torch.randn(10000, 8, device=device)
    
    # Warmup
    for _ in range(10):
        _ = triton_fft(x_re, x_im)
    torch.cuda.synchronize()
    
    # Triton timing
    start = time.time()
    for _ in range(100):
        _ = triton_fft(x_re, x_im)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # Python timing
    start = time.time()
    for _ in range(10):  # Only 10 iterations (Python is slow)
        _ = python_fft(x_re.clone(), x_im.clone())
    torch.cuda.synchronize()
    python_time = (time.time() - start) * 10  # Scale to 100
    
    # torch.fft timing
    x_complex = torch.complex(x_re, x_im)
    start = time.time()
    for _ in range(100):
        _ = torch.fft.fft(x_complex)
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    triton_rate = 10000 * 100 / triton_time
    python_rate = 10000 * 100 / python_time
    torch_rate = 10000 * 100 / torch_time
    
    print(f"  Triton:  {triton_time:.3f}s ({triton_rate:,.0f} FFTs/sec)")
    print(f"  Python:  {python_time:.3f}s ({python_rate:,.0f} FFTs/sec)")
    print(f"  Torch:   {torch_time:.3f}s ({torch_rate:,.0f} FFTs/sec)")
    print(f"  Speedup vs Python: {python_time/triton_time:.1f}x")
    print(f"  Speedup vs Torch:  {torch_time/triton_time:.2f}x")
    
    test5_pass = triton_time < python_time
    print(f"  Passed (faster than Python): {test5_pass}")
    
    # Summary
    print("\n" + "="*60)
    all_passed = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass
    if all_passed:
        print("ALL TESTS PASSED")
        print("\nTritonFFT: Soul ported to silicon.")
    else:
        print("SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    test_triton_fft()
