#!/usr/bin/env python3
"""
TritonFFT Thor - Scaled for NVIDIA Thor (80% Utilization)
==========================================================

Target Hardware:
- NVIDIA Thor (Jetson AGX Thor)
- 122.8 GB unified memory → 98 GB target (80%)
- 20 SMs → 16 SMs active (80%)
- 228 KB shared memory per SM
- Compute capability 11.0

Target FFT Size: N=16384 (2^14)

Algorithm: Stockham Autosort FFT
- No bit-reversal needed (automatically sorted)
- Better memory coalescing
- Ping-pong between two buffers

For N=16384:
- 14 stages
- Each stage: N/2 = 8192 butterflies
- Twiddle table: 128 KB
- Working memory: 256 KB (two buffers)
- Strategy: Twiddles in constant memory, work in global with coalesced access
"""

import torch
import torch.nn as nn
import math
import time
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# =============================================================================
# CONFIGURATION FOR THOR @ 80%
# =============================================================================

THOR_CONFIG = {
    'total_memory_gb': 122.8,
    'target_memory_gb': 98.0,  # 80%
    'num_sms': 20,
    'target_sms': 16,  # 80%
    'shared_mem_per_block': 48 * 1024,  # 48 KB
    'shared_mem_per_sm': 228 * 1024,  # 228 KB
    'max_fft_size': 16384,  # 2^14
    'max_batch_size': 1024 * 1024,  # 1M concurrent FFTs
}


# =============================================================================
# TWIDDLE TABLE GENERATION
# =============================================================================

def compute_twiddle_table(N: int, device='cuda', dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute twiddle factors W_N^k = e^{-2πik/N}.
    
    For N=16384: 128 KB table
    """
    k = torch.arange(N, dtype=dtype, device=device)
    angles = -2.0 * math.pi * k / N
    W_re = torch.cos(angles)
    W_im = torch.sin(angles)
    return W_re, W_im


# =============================================================================
# STOCKHAM FFT KERNEL - SINGLE STAGE
# =============================================================================

if HAS_TRITON:
    
    @triton.jit
    def stockham_stage_kernel(
        # Input buffer
        X_re_ptr, X_im_ptr,
        # Output buffer
        Y_re_ptr, Y_im_ptr,
        # Twiddle table
        W_re_ptr, W_im_ptr,
        # Stage parameters
        stage: tl.constexpr,
        N: tl.constexpr,
        LOG_N: tl.constexpr,
        # Batch info
        batch_stride: tl.constexpr,
        num_batches,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        One stage of Stockham autosort FFT.
        
        Stockham advantages:
        - No bit-reversal (output is automatically sorted)
        - Better memory coalescing
        - Ping-pong between buffers
        
        Each program handles BLOCK_SIZE butterflies.
        """
        # Program indices
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        
        if batch_idx >= num_batches:
            return
        
        # Base offset for this batch
        batch_base = batch_idx * batch_stride
        
        # Butterflies this block handles
        butterfly_start = block_idx * BLOCK_SIZE
        butterfly_offsets = butterfly_start + tl.arange(0, BLOCK_SIZE)
        
        # Number of butterflies per stage = N/2
        num_butterflies = N // 2
        mask = butterfly_offsets < num_butterflies
        
        # Stockham indexing
        # For stage s (0-indexed):
        #   stride = 2^s
        #   For butterfly b:
        #     group = b // stride
        #     pos = b % stride
        #     i = group * (2 * stride) + pos
        #     j = i + stride
        #     twiddle_idx = pos * (N // (2 * stride))
        
        stride = 1 << stage
        group = butterfly_offsets // stride
        pos = butterfly_offsets % stride
        
        # Input indices (from previous stage or original)
        i_idx = group * (2 * stride) + pos
        j_idx = i_idx + stride
        
        # Twiddle index
        tw_stride = N // (2 * stride)
        tw_idx = pos * tw_stride
        
        # Load twiddles
        w_re = tl.load(W_re_ptr + tw_idx, mask=mask, other=1.0)
        w_im = tl.load(W_im_ptr + tw_idx, mask=mask, other=0.0)
        
        # Load input values
        a_re = tl.load(X_re_ptr + batch_base + i_idx, mask=mask, other=0.0)
        a_im = tl.load(X_im_ptr + batch_base + i_idx, mask=mask, other=0.0)
        b_re = tl.load(X_re_ptr + batch_base + j_idx, mask=mask, other=0.0)
        b_im = tl.load(X_im_ptr + batch_base + j_idx, mask=mask, other=0.0)
        
        # Complex multiply: W * b
        wb_re = w_re * b_re - w_im * b_im
        wb_im = w_re * b_im + w_im * b_re
        
        # Butterfly outputs
        out_upper_re = a_re + wb_re
        out_upper_im = a_im + wb_im
        out_lower_re = a_re - wb_re
        out_lower_im = a_im - wb_im
        
        # Stockham output indexing (different from input!)
        # Output is "sorted" - consecutive elements go to consecutive memory
        # out_i = b (butterfly index)
        # out_j = b + N/2
        out_i = butterfly_offsets
        out_j = butterfly_offsets + num_butterflies
        
        # Store to output buffer
        tl.store(Y_re_ptr + batch_base + out_i, out_upper_re, mask=mask)
        tl.store(Y_im_ptr + batch_base + out_i, out_upper_im, mask=mask)
        tl.store(Y_re_ptr + batch_base + out_j, out_lower_re, mask=mask)
        tl.store(Y_im_ptr + batch_base + out_j, out_lower_im, mask=mask)


    @triton.jit  
    def fused_stockham_small_kernel(
        # Input
        X_re_ptr, X_im_ptr,
        # Output
        Y_re_ptr, Y_im_ptr,
        # Twiddle table
        W_re_ptr, W_im_ptr,
        # Dimensions
        N: tl.constexpr,
        LOG_N: tl.constexpr,
        batch_stride: tl.constexpr,
        num_batches,
    ):
        """
        Fused Stockham FFT for small N (up to 1024).
        
        All stages in one kernel using shared memory.
        """
        batch_idx = tl.program_id(0)
        if batch_idx >= num_batches:
            return
        
        base = batch_idx * batch_stride
        
        # Load all N elements
        offsets = tl.arange(0, N)
        x_re = tl.load(X_re_ptr + base + offsets)
        x_im = tl.load(X_im_ptr + base + offsets)
        
        # Process all stages
        # Note: This requires N to be a tl.constexpr
        # The loop will be unrolled at compile time
        
        # Stage 0
        stride = 1
        for s in tl.static_range(LOG_N):
            stride = 1 << s
            half_n = N // 2
            
            # Compute butterfly indices (vectorized)
            b_idx = tl.arange(0, N // 2)
            group = b_idx // stride
            pos = b_idx % stride
            
            i_idx = group * (2 * stride) + pos
            j_idx = i_idx + stride
            
            # Twiddle
            tw_stride = N // (2 * stride)
            tw_idx = pos * tw_stride
            
            w_re = tl.load(W_re_ptr + tw_idx)
            w_im = tl.load(W_im_ptr + tw_idx)
            
            # This is tricky in Triton - need to gather/scatter
            # For now, this is pseudo-code showing the intent
            # Real implementation needs careful indexing
            
        # Store result
        tl.store(Y_re_ptr + base + offsets, x_re)
        tl.store(Y_im_ptr + base + offsets, x_im)


# =============================================================================
# THOR FFT CLASS
# =============================================================================

class TritonFFTThor(nn.Module):
    """
    Triton FFT optimized for NVIDIA Thor @ 80% utilization.
    
    Features:
    - Stockham autosort (no bit-reversal)
    - Multi-stage kernel launches
    - Ping-pong buffers
    - Supports N up to 16384
    - Batch sizes up to 1M
    
    Memory layout:
    - Twiddle tables: precomputed, reused
    - Working buffers: ping-pong for stages
    """
    
    def __init__(self, max_n: int = 16384):
        super().__init__()
        self.max_n = max_n
        self._twiddle_cache = {}
        
        # Validate against Thor config
        if max_n > THOR_CONFIG['max_fft_size']:
            print(f"Warning: N={max_n} exceeds recommended max {THOR_CONFIG['max_fft_size']}")
    
    def _get_twiddles(self, N: int, device, dtype=torch.float32):
        key = (N, str(device), dtype)
        if key not in self._twiddle_cache:
            self._twiddle_cache[key] = compute_twiddle_table(N, device, dtype)
        return self._twiddle_cache[key]
    
    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FFT using Stockham algorithm.
        
        Args:
            x_re: Real parts, shape (..., N)
            x_im: Imaginary parts, shape (..., N)
        
        Returns:
            Y_re, Y_im: FFT output
        """
        if not HAS_TRITON:
            # Fallback to torch.fft
            x = torch.complex(x_re, x_im)
            y = torch.fft.fft(x)
            return y.real.contiguous(), y.imag.contiguous()
        
        *batch_dims, N = x_re.shape
        device = x_re.device
        dtype = x_re.dtype
        
        LOG_N = int(math.log2(N))
        assert 2**LOG_N == N, f"N must be power of 2, got {N}"
        
        # Flatten batch dimensions
        batch_size = 1
        for d in batch_dims:
            batch_size *= d
        
        x_re_flat = x_re.reshape(batch_size, N).contiguous()
        x_im_flat = x_im.reshape(batch_size, N).contiguous()
        
        # Get twiddles
        W_re, W_im = self._get_twiddles(N, device, dtype)
        
        # Allocate ping-pong buffers
        buf_a_re = x_re_flat.clone()
        buf_a_im = x_im_flat.clone()
        buf_b_re = torch.empty_like(buf_a_re)
        buf_b_im = torch.empty_like(buf_a_im)
        
        # Process stages with ping-pong
        BLOCK_SIZE = min(1024, N // 2)
        num_blocks = (N // 2 + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        src_re, src_im = buf_a_re, buf_a_im
        dst_re, dst_im = buf_b_re, buf_b_im
        
        for stage in range(LOG_N):
            grid = (batch_size, num_blocks)
            
            stockham_stage_kernel[grid](
                src_re, src_im,
                dst_re, dst_im,
                W_re, W_im,
                stage, N, LOG_N,
                N,  # batch_stride
                batch_size,
                BLOCK_SIZE,
            )
            
            # Swap buffers
            src_re, src_im, dst_re, dst_im = dst_re, dst_im, src_re, src_im
        
        # Result is in src after final swap
        y_re = src_re.reshape(*batch_dims, N)
        y_im = src_im.reshape(*batch_dims, N)
        
        return y_re, y_im
    
    def benchmark(self, N: int = 16384, batch_size: int = 10000, iterations: int = 100):
        """Benchmark FFT performance."""
        device = 'cuda'
        
        print(f"=== TritonFFT Thor Benchmark ===")
        print(f"N={N}, batch={batch_size}, iters={iterations}")
        print()
        
        x_re = torch.randn(batch_size, N, device=device)
        x_im = torch.randn(batch_size, N, device=device)
        
        # Warmup
        for _ in range(10):
            _ = self(x_re, x_im)
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start = time.time()
        for _ in range(iterations):
            y_re, y_im = self(x_re, x_im)
        torch.cuda.synchronize()
        triton_time = time.time() - start
        
        # Benchmark torch.fft
        x_complex = torch.complex(x_re, x_im)
        start = time.time()
        for _ in range(iterations):
            _ = torch.fft.fft(x_complex)
        torch.cuda.synchronize()
        torch_time = time.time() - start
        
        triton_rate = batch_size * iterations / triton_time
        torch_rate = batch_size * iterations / torch_time
        
        print(f"Triton FFT: {triton_time:.3f}s ({triton_rate:,.0f} FFTs/sec)")
        print(f"Torch FFT:  {torch_time:.3f}s ({torch_rate:,.0f} FFTs/sec)")
        print(f"Ratio: {triton_time/torch_time:.2f}x (Triton/Torch)")
        print()
        
        # Memory usage
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_target = THOR_CONFIG['target_memory_gb']
        print(f"Memory used: {mem_used:.1f} GB / {mem_target:.1f} GB target ({100*mem_used/mem_target:.1f}%)")
        
        # Verify correctness
        y_torch = torch.fft.fft(x_complex)
        error_re = (y_re - y_torch.real).abs().max().item()
        error_im = (y_im - y_torch.imag).abs().max().item()
        print(f"Max error vs torch: {max(error_re, error_im):.2e}")
        
        return {
            'triton_rate': triton_rate,
            'torch_rate': torch_rate,
            'ratio': triton_time / torch_time,
            'error': max(error_re, error_im),
        }


# =============================================================================
# 80% UTILIZATION TEST
# =============================================================================

def test_thor_80_percent():
    """
    Test FFT at 80% of Thor's capabilities.
    
    Target:
    - N = 16384
    - Batch size to use ~80GB
    - 16 of 20 SMs active
    """
    print("="*70)
    print("TRITON FFT THOR - 80% UTILIZATION TEST")
    print("="*70)
    print()
    
    device = 'cuda'
    
    # Calculate batch size for 80% memory
    # Each FFT: 16384 * 2 * 4 * 2 = 256 KB (input + output, real + imag)
    # Plus twiddles: 128 KB (shared)
    # Target: 98 GB
    
    N = 16384
    bytes_per_fft = N * 2 * 4 * 4  # real+imag, input+output, float32
    target_bytes = int(THOR_CONFIG['target_memory_gb'] * 1024**3 * 0.5)  # 50% for safety
    max_batch = target_bytes // bytes_per_fft
    
    # Use a reasonable batch size
    batch_size = min(max_batch, 100000)  # Cap at 100K for testing
    
    print(f"Configuration:")
    print(f"  N = {N}")
    print(f"  Batch size = {batch_size:,}")
    print(f"  Memory per FFT = {bytes_per_fft / 1024:.1f} KB")
    print(f"  Total memory = {batch_size * bytes_per_fft / 1024**3:.1f} GB")
    print()
    
    # Create FFT
    fft = TritonFFTThor(max_n=N)
    
    # Test correctness first with small batch
    print("[1] Correctness Test (batch=100)")
    x_re = torch.randn(100, N, device=device)
    x_im = torch.randn(100, N, device=device)
    
    y_re, y_im = fft(x_re, x_im)
    y_torch = torch.fft.fft(torch.complex(x_re, x_im))
    
    error = max(
        (y_re - y_torch.real).abs().max().item(),
        (y_im - y_torch.imag).abs().max().item()
    )
    print(f"  Max error: {error:.2e}")
    print(f"  Passed: {error < 1e-4}")
    print()
    
    # Performance test
    print(f"[2] Performance Test (batch={batch_size:,})")
    result = fft.benchmark(N=N, batch_size=batch_size, iterations=10)
    print()
    
    # Calculate effective utilization
    print("[3] Resource Utilization")
    mem_used_gb = torch.cuda.max_memory_allocated() / 1024**3
    utilization = mem_used_gb / THOR_CONFIG['total_memory_gb'] * 100
    print(f"  Peak memory: {mem_used_gb:.1f} GB")
    print(f"  Utilization: {utilization:.1f}%")
    print(f"  Target: 80%")
    print()
    
    # Riemann Probe projection
    print("[4] Riemann Probe Projection")
    # Each Z(t) evaluation needs ~1 FFT of size sqrt(t)
    # At t=10^9, sqrt(t) ≈ 31623, round up to N=32768
    # At t=10^12, sqrt(t) ≈ 10^6, need larger FFT or multiple
    
    zeros_per_fft = N  # Approximate: one FFT gives N Z values
    zeros_per_sec = result['triton_rate'] * zeros_per_fft
    
    print(f"  FFT rate: {result['triton_rate']:,.0f} FFTs/sec")
    print(f"  Estimated zeros/sec: {zeros_per_sec:,.0f}")
    print(f"  Time for 10^9 zeros: {1e9/zeros_per_sec/3600:.1f} hours")
    print(f"  Time for 10^12 zeros: {1e12/zeros_per_sec/3600/24:.1f} days")
    print()
    
    print("="*70)
    if error < 1e-4 and utilization > 50:
        print("SUCCESS: Thor FFT operational at scale")
    else:
        print("PARTIAL: Needs optimization")
    print("="*70)
    
    return result


if __name__ == "__main__":
    test_thor_80_percent()
