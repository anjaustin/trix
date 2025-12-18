#!/usr/bin/env python3
"""
TritonFFT Large - Scalable FFT for N up to 16384
=================================================

Phase 2: Generalized Triton FFT using shared memory.

Architecture:
- Twiddle table in shared memory (the "Twiddle Tile")
- Working buffer in shared memory (stage results)
- Loop-based stages with tl.static_range
- Supports N = 8, 16, 32, ..., 16384

This is the Hollywood Squares FFT at Layer 1:
- Each kernel is a "Tile" 
- Shared memory is the "message bus"
- Stages are "wired" not "parallelized"
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: Triton not available")


# =============================================================================
# TWIDDLE TABLE GENERATION
# =============================================================================

def compute_twiddle_table(N: int, device='cuda', dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute twiddle factors W_N^k = e^{-2πik/N}.
    
    This is the "Twiddle Tile" - computed once, loaded to shared memory.
    No runtime trig. Ever.
    """
    k = torch.arange(N, dtype=dtype, device=device)
    angles = -2.0 * math.pi * k / N
    W_re = torch.cos(angles)
    W_im = torch.sin(angles)
    return W_re, W_im


def compute_bitrev_permutation(N: int, device='cuda') -> torch.Tensor:
    """Bit-reversal permutation for Cooley-Tukey DIT."""
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
# TRITON KERNEL - CONFIGURABLE N
# =============================================================================

if HAS_TRITON:
    
    @triton.jit
    def fft_butterfly_stage(
        # Working arrays (in shared memory conceptually)
        y_re_ptr, y_im_ptr,
        # Twiddle table
        W_re_ptr, W_im_ptr,
        # Stage parameters
        stage: tl.constexpr,
        N: tl.constexpr,
        # Thread info
        tid,
    ):
        """
        One stage of Cooley-Tukey DIT FFT.
        
        Each thread handles one butterfly operation.
        """
        stride = 1 << stage
        group_size = stride << 1
        
        # Which butterfly in this stage?
        butterfly_idx = tid
        
        # Compute indices
        group = butterfly_idx // stride
        pos_in_group = butterfly_idx % stride
        
        i = group * group_size + pos_in_group
        j = i + stride
        
        # Twiddle index: k = pos_in_group * (N / group_size)
        tw_idx = pos_in_group * (N // group_size)
        
        # Load twiddle
        w_re = tl.load(W_re_ptr + tw_idx)
        w_im = tl.load(W_im_ptr + tw_idx)
        
        # Load values
        a_re = tl.load(y_re_ptr + i)
        a_im = tl.load(y_im_ptr + i)
        b_re = tl.load(y_re_ptr + j)
        b_im = tl.load(y_im_ptr + j)
        
        # Complex multiply: W * b
        wb_re = w_re * b_re - w_im * b_im
        wb_im = w_re * b_im + w_im * b_re
        
        # Butterfly
        tl.store(y_re_ptr + i, a_re + wb_re)
        tl.store(y_im_ptr + i, a_im + wb_im)
        tl.store(y_re_ptr + j, a_re - wb_re)
        tl.store(y_im_ptr + j, a_im - wb_im)


    @triton.jit
    def trix_fft_general_kernel(
        # Input
        X_re_ptr, X_im_ptr,
        # Output
        Y_re_ptr, Y_im_ptr,
        # Twiddle table
        W_re_ptr, W_im_ptr,
        # Bit-reversal indices
        bitrev_ptr,
        # Dimensions
        N: tl.constexpr,
        LOG_N: tl.constexpr,
        batch_stride: tl.constexpr,
        # Block config
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        General FFT kernel for any power-of-2 N.
        
        Uses shared memory for working buffer.
        Each program handles one FFT in the batch.
        """
        batch_idx = tl.program_id(0)
        base = batch_idx * batch_stride
        
        # Thread ID within the block
        tid = tl.arange(0, BLOCK_SIZE)
        mask = tid < N
        
        # Load input with bit-reversal into registers/local
        # For large N, this goes through shared memory
        indices = tl.load(bitrev_ptr + tid, mask=mask, other=0)
        
        y_re = tl.load(X_re_ptr + base + indices, mask=mask, other=0.0)
        y_im = tl.load(X_im_ptr + base + indices, mask=mask, other=0.0)
        
        # Store to output buffer (we'll use it as working space)
        tl.store(Y_re_ptr + base + tid, y_re, mask=mask)
        tl.store(Y_im_ptr + base + tid, y_im, mask=mask)
        
        # Synchronize
        tl.debug_barrier()
        
        # Process stages
        # Note: We need to reload/store between stages for correctness
        for stage in tl.static_range(LOG_N):
            stride = 1 << stage
            group_size = stride << 1
            
            # Only N/2 butterflies per stage
            num_butterflies = N // 2
            
            # Each thread handles butterflies
            for b in tl.static_range(BLOCK_SIZE // 2):  # Simplified
                if b < num_butterflies:
                    # Compute indices for this butterfly
                    group = b // stride
                    pos_in_group = b % stride
                    
                    i = group * group_size + pos_in_group
                    j = i + stride
                    
                    # Twiddle index
                    tw_idx = pos_in_group * (N // group_size)
                    
                    # Load twiddle
                    w_re = tl.load(W_re_ptr + tw_idx)
                    w_im = tl.load(W_im_ptr + tw_idx)
                    
                    # Load values from working buffer
                    a_re = tl.load(Y_re_ptr + base + i)
                    a_im = tl.load(Y_im_ptr + base + i)
                    b_re = tl.load(Y_re_ptr + base + j)
                    b_im = tl.load(Y_im_ptr + base + j)
                    
                    # Complex multiply: W * b
                    wb_re = w_re * b_re - w_im * b_im
                    wb_im = w_re * b_im + w_im * b_re
                    
                    # Butterfly - store back
                    tl.store(Y_re_ptr + base + i, a_re + wb_re)
                    tl.store(Y_im_ptr + base + i, a_im + wb_im)
                    tl.store(Y_re_ptr + base + j, a_re - wb_re)
                    tl.store(Y_im_ptr + base + j, a_im - wb_im)
            
            # Barrier between stages
            tl.debug_barrier()


    # Specialized kernels for specific sizes (much faster than general)
    @triton.jit
    def trix_fft_n16_kernel(
        X_re_ptr, X_im_ptr,
        Y_re_ptr, Y_im_ptr,
        W_re_ptr, W_im_ptr,
        batch_stride,
        num_batches,
    ):
        """Optimized N=16 FFT (4 stages, fully unrolled)."""
        batch_idx = tl.program_id(0)
        if batch_idx >= num_batches:
            return
        
        base = batch_idx * batch_stride
        
        # Bit-reversal for N=16: [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
        BITREV = tl.constexpr([0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15])
        
        # Load with bit-reversal (unrolled)
        x0_re = tl.load(X_re_ptr + base + 0); x0_im = tl.load(X_im_ptr + base + 0)
        x1_re = tl.load(X_re_ptr + base + 8); x1_im = tl.load(X_im_ptr + base + 8)
        x2_re = tl.load(X_re_ptr + base + 4); x2_im = tl.load(X_im_ptr + base + 4)
        x3_re = tl.load(X_re_ptr + base + 12); x3_im = tl.load(X_im_ptr + base + 12)
        x4_re = tl.load(X_re_ptr + base + 2); x4_im = tl.load(X_im_ptr + base + 2)
        x5_re = tl.load(X_re_ptr + base + 10); x5_im = tl.load(X_im_ptr + base + 10)
        x6_re = tl.load(X_re_ptr + base + 6); x6_im = tl.load(X_im_ptr + base + 6)
        x7_re = tl.load(X_re_ptr + base + 14); x7_im = tl.load(X_im_ptr + base + 14)
        x8_re = tl.load(X_re_ptr + base + 1); x8_im = tl.load(X_im_ptr + base + 1)
        x9_re = tl.load(X_re_ptr + base + 9); x9_im = tl.load(X_im_ptr + base + 9)
        x10_re = tl.load(X_re_ptr + base + 5); x10_im = tl.load(X_im_ptr + base + 5)
        x11_re = tl.load(X_re_ptr + base + 13); x11_im = tl.load(X_im_ptr + base + 13)
        x12_re = tl.load(X_re_ptr + base + 3); x12_im = tl.load(X_im_ptr + base + 3)
        x13_re = tl.load(X_re_ptr + base + 11); x13_im = tl.load(X_im_ptr + base + 11)
        x14_re = tl.load(X_re_ptr + base + 7); x14_im = tl.load(X_im_ptr + base + 7)
        x15_re = tl.load(X_re_ptr + base + 15); x15_im = tl.load(X_im_ptr + base + 15)
        
        # Load twiddles (only need W^0 through W^7 for N=16)
        W0_re = tl.load(W_re_ptr + 0); W0_im = tl.load(W_im_ptr + 0)
        W1_re = tl.load(W_re_ptr + 1); W1_im = tl.load(W_im_ptr + 1)
        W2_re = tl.load(W_re_ptr + 2); W2_im = tl.load(W_im_ptr + 2)
        W3_re = tl.load(W_re_ptr + 3); W3_im = tl.load(W_im_ptr + 3)
        W4_re = tl.load(W_re_ptr + 4); W4_im = tl.load(W_im_ptr + 4)
        W5_re = tl.load(W_re_ptr + 5); W5_im = tl.load(W_im_ptr + 5)
        W6_re = tl.load(W_re_ptr + 6); W6_im = tl.load(W_im_ptr + 6)
        W7_re = tl.load(W_re_ptr + 7); W7_im = tl.load(W_im_ptr + 7)
        
        # ===== STAGE 0: stride=1, all use W^0 =====
        # Butterflies: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
        t_re = x1_re; t_im = x1_im
        y0_re = x0_re + t_re; y0_im = x0_im + t_im
        y1_re = x0_re - t_re; y1_im = x0_im - t_im
        
        t_re = x3_re; t_im = x3_im
        y2_re = x2_re + t_re; y2_im = x2_im + t_im
        y3_re = x2_re - t_re; y3_im = x2_im - t_im
        
        t_re = x5_re; t_im = x5_im
        y4_re = x4_re + t_re; y4_im = x4_im + t_im
        y5_re = x4_re - t_re; y5_im = x4_im - t_im
        
        t_re = x7_re; t_im = x7_im
        y6_re = x6_re + t_re; y6_im = x6_im + t_im
        y7_re = x6_re - t_re; y7_im = x6_im - t_im
        
        t_re = x9_re; t_im = x9_im
        y8_re = x8_re + t_re; y8_im = x8_im + t_im
        y9_re = x8_re - t_re; y9_im = x8_im - t_im
        
        t_re = x11_re; t_im = x11_im
        y10_re = x10_re + t_re; y10_im = x10_im + t_im
        y11_re = x10_re - t_re; y11_im = x10_im - t_im
        
        t_re = x13_re; t_im = x13_im
        y12_re = x12_re + t_re; y12_im = x12_im + t_im
        y13_re = x12_re - t_re; y13_im = x12_im - t_im
        
        t_re = x15_re; t_im = x15_im
        y14_re = x14_re + t_re; y14_im = x14_im + t_im
        y15_re = x14_re - t_re; y15_im = x14_im - t_im
        
        # ===== STAGE 1: stride=2, W^0 and W^4 =====
        # Butterflies: (0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,14), (13,15)
        t_re = y2_re; t_im = y2_im
        z0_re = y0_re + t_re; z0_im = y0_im + t_im
        z2_re = y0_re - t_re; z2_im = y0_im - t_im
        
        t_re = y3_re * W4_re - y3_im * W4_im
        t_im = y3_re * W4_im + y3_im * W4_re
        z1_re = y1_re + t_re; z1_im = y1_im + t_im
        z3_re = y1_re - t_re; z3_im = y1_im - t_im
        
        t_re = y6_re; t_im = y6_im
        z4_re = y4_re + t_re; z4_im = y4_im + t_im
        z6_re = y4_re - t_re; z6_im = y4_im - t_im
        
        t_re = y7_re * W4_re - y7_im * W4_im
        t_im = y7_re * W4_im + y7_im * W4_re
        z5_re = y5_re + t_re; z5_im = y5_im + t_im
        z7_re = y5_re - t_re; z7_im = y5_im - t_im
        
        t_re = y10_re; t_im = y10_im
        z8_re = y8_re + t_re; z8_im = y8_im + t_im
        z10_re = y8_re - t_re; z10_im = y8_im - t_im
        
        t_re = y11_re * W4_re - y11_im * W4_im
        t_im = y11_re * W4_im + y11_im * W4_re
        z9_re = y9_re + t_re; z9_im = y9_im + t_im
        z11_re = y9_re - t_re; z11_im = y9_im - t_im
        
        t_re = y14_re; t_im = y14_im
        z12_re = y12_re + t_re; z12_im = y12_im + t_im
        z14_re = y12_re - t_re; z14_im = y12_im - t_im
        
        t_re = y15_re * W4_re - y15_im * W4_im
        t_im = y15_re * W4_im + y15_im * W4_re
        z13_re = y13_re + t_re; z13_im = y13_im + t_im
        z15_re = y13_re - t_re; z15_im = y13_im - t_im
        
        # ===== STAGE 2: stride=4, W^0, W^2, W^4, W^6 =====
        t_re = z4_re; t_im = z4_im
        w0_re = z0_re + t_re; w0_im = z0_im + t_im
        w4_re = z0_re - t_re; w4_im = z0_im - t_im
        
        t_re = z5_re * W2_re - z5_im * W2_im
        t_im = z5_re * W2_im + z5_im * W2_re
        w1_re = z1_re + t_re; w1_im = z1_im + t_im
        w5_re = z1_re - t_re; w5_im = z1_im - t_im
        
        t_re = z6_re * W4_re - z6_im * W4_im
        t_im = z6_re * W4_im + z6_im * W4_re
        w2_re = z2_re + t_re; w2_im = z2_im + t_im
        w6_re = z2_re - t_re; w6_im = z2_im - t_im
        
        t_re = z7_re * W6_re - z7_im * W6_im
        t_im = z7_re * W6_im + z7_im * W6_re
        w3_re = z3_re + t_re; w3_im = z3_im + t_im
        w7_re = z3_re - t_re; w7_im = z3_im - t_im
        
        t_re = z12_re; t_im = z12_im
        w8_re = z8_re + t_re; w8_im = z8_im + t_im
        w12_re = z8_re - t_re; w12_im = z8_im - t_im
        
        t_re = z13_re * W2_re - z13_im * W2_im
        t_im = z13_re * W2_im + z13_im * W2_re
        w9_re = z9_re + t_re; w9_im = z9_im + t_im
        w13_re = z9_re - t_re; w13_im = z9_im - t_im
        
        t_re = z14_re * W4_re - z14_im * W4_im
        t_im = z14_re * W4_im + z14_im * W4_re
        w10_re = z10_re + t_re; w10_im = z10_im + t_im
        w14_re = z10_re - t_re; w14_im = z10_im - t_im
        
        t_re = z15_re * W6_re - z15_im * W6_im
        t_im = z15_re * W6_im + z15_im * W6_re
        w11_re = z11_re + t_re; w11_im = z11_im + t_im
        w15_re = z11_re - t_re; w15_im = z11_im - t_im
        
        # ===== STAGE 3: stride=8, W^0 through W^7 =====
        t_re = w8_re; t_im = w8_im
        out0_re = w0_re + t_re; out0_im = w0_im + t_im
        out8_re = w0_re - t_re; out8_im = w0_im - t_im
        
        t_re = w9_re * W1_re - w9_im * W1_im
        t_im = w9_re * W1_im + w9_im * W1_re
        out1_re = w1_re + t_re; out1_im = w1_im + t_im
        out9_re = w1_re - t_re; out9_im = w1_im - t_im
        
        t_re = w10_re * W2_re - w10_im * W2_im
        t_im = w10_re * W2_im + w10_im * W2_re
        out2_re = w2_re + t_re; out2_im = w2_im + t_im
        out10_re = w2_re - t_re; out10_im = w2_im - t_im
        
        t_re = w11_re * W3_re - w11_im * W3_im
        t_im = w11_re * W3_im + w11_im * W3_re
        out3_re = w3_re + t_re; out3_im = w3_im + t_im
        out11_re = w3_re - t_re; out11_im = w3_im - t_im
        
        t_re = w12_re * W4_re - w12_im * W4_im
        t_im = w12_re * W4_im + w12_im * W4_re
        out4_re = w4_re + t_re; out4_im = w4_im + t_im
        out12_re = w4_re - t_re; out12_im = w4_im - t_im
        
        t_re = w13_re * W5_re - w13_im * W5_im
        t_im = w13_re * W5_im + w13_im * W5_re
        out5_re = w5_re + t_re; out5_im = w5_im + t_im
        out13_re = w5_re - t_re; out13_im = w5_im - t_im
        
        t_re = w14_re * W6_re - w14_im * W6_im
        t_im = w14_re * W6_im + w14_im * W6_re
        out6_re = w6_re + t_re; out6_im = w6_im + t_im
        out14_re = w6_re - t_re; out14_im = w6_im - t_im
        
        t_re = w15_re * W7_re - w15_im * W7_im
        t_im = w15_re * W7_im + w15_im * W7_re
        out7_re = w7_re + t_re; out7_im = w7_im + t_im
        out15_re = w7_re - t_re; out15_im = w7_im - t_im
        
        # Store output
        tl.store(Y_re_ptr + base + 0, out0_re)
        tl.store(Y_re_ptr + base + 1, out1_re)
        tl.store(Y_re_ptr + base + 2, out2_re)
        tl.store(Y_re_ptr + base + 3, out3_re)
        tl.store(Y_re_ptr + base + 4, out4_re)
        tl.store(Y_re_ptr + base + 5, out5_re)
        tl.store(Y_re_ptr + base + 6, out6_re)
        tl.store(Y_re_ptr + base + 7, out7_re)
        tl.store(Y_re_ptr + base + 8, out8_re)
        tl.store(Y_re_ptr + base + 9, out9_re)
        tl.store(Y_re_ptr + base + 10, out10_re)
        tl.store(Y_re_ptr + base + 11, out11_re)
        tl.store(Y_re_ptr + base + 12, out12_re)
        tl.store(Y_re_ptr + base + 13, out13_re)
        tl.store(Y_re_ptr + base + 14, out14_re)
        tl.store(Y_re_ptr + base + 15, out15_re)
        
        tl.store(Y_im_ptr + base + 0, out0_im)
        tl.store(Y_im_ptr + base + 1, out1_im)
        tl.store(Y_im_ptr + base + 2, out2_im)
        tl.store(Y_im_ptr + base + 3, out3_im)
        tl.store(Y_im_ptr + base + 4, out4_im)
        tl.store(Y_im_ptr + base + 5, out5_im)
        tl.store(Y_im_ptr + base + 6, out6_im)
        tl.store(Y_im_ptr + base + 7, out7_im)
        tl.store(Y_im_ptr + base + 8, out8_im)
        tl.store(Y_im_ptr + base + 9, out9_im)
        tl.store(Y_im_ptr + base + 10, out10_im)
        tl.store(Y_im_ptr + base + 11, out11_im)
        tl.store(Y_im_ptr + base + 12, out12_im)
        tl.store(Y_im_ptr + base + 13, out13_im)
        tl.store(Y_im_ptr + base + 14, out14_im)
        tl.store(Y_im_ptr + base + 15, out15_im)


# =============================================================================
# CODE GENERATOR FOR LARGE N
# =============================================================================

def generate_fft_kernel_code(N: int) -> str:
    """
    Generate Triton kernel code for specific N.
    
    For N=16384, we can't manually write 14 stages.
    Instead, generate the code programmatically.
    """
    LOG_N = int(math.log2(N))
    
    # Generate bit-reversal permutation
    bitrev = []
    for i in range(N):
        rev = 0
        val = i
        for _ in range(LOG_N):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        bitrev.append(rev)
    
    code = f'''
@triton.jit
def trix_fft_n{N}_kernel(
    X_re_ptr, X_im_ptr,
    Y_re_ptr, Y_im_ptr,
    W_re_ptr, W_im_ptr,
    batch_stride,
    num_batches,
    BLOCK_SIZE: tl.constexpr,
):
    """Auto-generated N={N} FFT kernel ({LOG_N} stages)."""
    batch_idx = tl.program_id(0)
    if batch_idx >= num_batches:
        return
    
    base = batch_idx * batch_stride
    
    # Load with bit-reversal using vector operations
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < {N}
    
    # Bit-reversal lookup (precomputed)
    bitrev = tl.constexpr({bitrev})
    rev_offsets = tl.load(bitrev + offsets, mask=mask, other=0)  # This won't work directly
    
    # For large N, load linearly and do bit-reversal via gather
    x_re = tl.load(X_re_ptr + base + offsets, mask=mask, other=0.0)
    x_im = tl.load(X_im_ptr + base + offsets, mask=mask, other=0.0)
    
    # Store to working buffer with bit-reversal
    # ... stages would go here ...
'''
    
    return code


# =============================================================================
# TRITON FFT CLASS (Multi-size support)
# =============================================================================

class TritonFFTLarge(nn.Module):
    """
    Triton FFT supporting multiple sizes up to N=16384.
    
    Uses specialized kernels for small N (8, 16, 32).
    Uses shared-memory approach for large N.
    """
    
    SUPPORTED_SIZES = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
    
    def __init__(self):
        super().__init__()
        self._twiddle_cache = {}
        self._bitrev_cache = {}
    
    def _get_twiddles(self, N: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (N, str(device))
        if key not in self._twiddle_cache:
            self._twiddle_cache[key] = compute_twiddle_table(N, device)
        return self._twiddle_cache[key]
    
    def _get_bitrev(self, N: int, device) -> torch.Tensor:
        key = (N, str(device))
        if key not in self._bitrev_cache:
            self._bitrev_cache[key] = compute_bitrev_permutation(N, device)
        return self._bitrev_cache[key]
    
    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not HAS_TRITON:
            raise RuntimeError("Triton not available")
        
        *batch_dims, N = x_re.shape
        
        if N not in self.SUPPORTED_SIZES:
            raise ValueError(f"N={N} not supported. Use one of {self.SUPPORTED_SIZES}")
        
        device = x_re.device
        batch_size = 1
        for d in batch_dims:
            batch_size *= d
        
        x_re_flat = x_re.reshape(batch_size, N).contiguous()
        x_im_flat = x_im.reshape(batch_size, N).contiguous()
        
        y_re = torch.empty_like(x_re_flat)
        y_im = torch.empty_like(x_im_flat)
        
        W_re, W_im = self._get_twiddles(N, device)
        
        grid = (batch_size,)
        
        # Dispatch to appropriate kernel
        if N == 8:
            from .triton_fft import trix_fft_n8_kernel
            trix_fft_n8_kernel[grid](
                x_re_flat, x_im_flat, y_re, y_im,
                W_re, W_im, N, batch_size
            )
        elif N == 16:
            trix_fft_n16_kernel[grid](
                x_re_flat, x_im_flat, y_re, y_im,
                W_re, W_im, N, batch_size
            )
        else:
            # For larger N, fall back to torch.fft for now
            # Full implementation would use generated/shared-memory kernels
            x_complex = torch.complex(x_re_flat, x_im_flat)
            y_complex = torch.fft.fft(x_complex)
            y_re = y_complex.real.contiguous()
            y_im = y_complex.imag.contiguous()
        
        return y_re.reshape(*batch_dims, N), y_im.reshape(*batch_dims, N)


# =============================================================================
# TESTS
# =============================================================================

def test_triton_fft_large():
    print("="*60)
    print("TRITON FFT LARGE - MULTI-SIZE TEST")
    print("="*60)
    
    if not HAS_TRITON:
        print("Triton not available, skipping")
        return
    
    device = 'cuda'
    fft = TritonFFTLarge()
    
    # Test N=16
    print("\n[Test] N=16")
    x_re = torch.randn(100, 16, device=device)
    x_im = torch.randn(100, 16, device=device)
    
    y_re, y_im = fft(x_re, x_im)
    
    # Compare to torch.fft
    y_torch = torch.fft.fft(torch.complex(x_re, x_im))
    
    error_re = (y_re - y_torch.real).abs().max().item()
    error_im = (y_im - y_torch.imag).abs().max().item()
    
    print(f"  Max error (real): {error_re:.2e}")
    print(f"  Max error (imag): {error_im:.2e}")
    print(f"  Passed: {max(error_re, error_im) < 1e-5}")
    
    # Benchmark
    print("\n[Benchmark] N=16, batch=10000")
    import time
    
    x_re = torch.randn(10000, 16, device=device)
    x_im = torch.randn(10000, 16, device=device)
    
    # Warmup
    for _ in range(10):
        _ = fft(x_re, x_im)
    torch.cuda.synchronize()
    
    # Time
    start = time.time()
    for _ in range(100):
        _ = fft(x_re, x_im)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # Compare to torch
    x_complex = torch.complex(x_re, x_im)
    start = time.time()
    for _ in range(100):
        _ = torch.fft.fft(x_complex)
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    print(f"  Triton: {triton_time:.3f}s ({10000*100/triton_time:,.0f} FFTs/sec)")
    print(f"  Torch:  {torch_time:.3f}s ({10000*100/torch_time:,.0f} FFTs/sec)")
    print(f"  Ratio:  {torch_time/triton_time:.2f}x")
    
    print("\n" + "="*60)
    print("For N=16384, we need:")
    print("  - Code generation for 14 stages, or")
    print("  - Shared-memory loop-based kernel")
    print("  - Current implementation falls back to torch.fft")
    print("="*60)


if __name__ == "__main__":
    test_triton_fft_large()
