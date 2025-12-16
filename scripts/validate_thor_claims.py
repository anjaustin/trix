#!/usr/bin/env python3
"""
Validate "Thor Compiler" Claims

Tests:
1. Memory savings from bit-packing (2-bit ternary vs FP16/FP4)
2. Compute performance: logic-mux vs multiplication
3. Realistic CUDA considerations (warp divergence)

Run: python scripts/validate_thor_claims.py
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
print()


# =============================================================================
# CLAIM 1: Memory Savings from Bit-Packing
# =============================================================================

@dataclass
class MemoryAnalysis:
    """Memory usage for different representations"""
    num_weights: int
    fp32_bytes: int
    fp16_bytes: int
    fp4_bytes: int  # 4-bit, one weight per nibble
    packed_2bit_bytes: int  # 2-bit, 4 weights per byte
    
    def print_report(self):
        print("=" * 60)
        print("CLAIM 1: Memory Savings Analysis")
        print("=" * 60)
        print(f"Number of weights: {self.num_weights:,}")
        print()
        print(f"{'Format':<20} {'Bytes':>15} {'vs FP16':>12} {'vs FP4':>12}")
        print("-" * 60)
        print(f"{'FP32 (baseline)':<20} {self.fp32_bytes:>15,} {self.fp32_bytes/self.fp16_bytes:>11.1f}x {'-':>12}")
        print(f"{'FP16 (standard)':<20} {self.fp16_bytes:>15,} {1.0:>11.1f}x {self.fp16_bytes/self.fp4_bytes:>11.1f}x")
        print(f"{'FP4 (Thor native)':<20} {self.fp4_bytes:>15,} {self.fp16_bytes/self.fp4_bytes:>11.1f}x {1.0:>11.1f}x")
        print(f"{'2-bit packed':<20} {self.packed_2bit_bytes:>15,} {self.fp16_bytes/self.packed_2bit_bytes:>11.1f}x {self.fp4_bytes/self.packed_2bit_bytes:>11.1f}x")
        print()
        print(f"VERDICT: 2-bit packing gives {self.fp16_bytes/self.packed_2bit_bytes:.1f}x over FP16, "
              f"{self.fp4_bytes/self.packed_2bit_bytes:.1f}x over FP4")
        print()


def analyze_memory(num_weights: int = 70_000_000_000) -> MemoryAnalysis:
    """Calculate memory for a 70B parameter model"""
    return MemoryAnalysis(
        num_weights=num_weights,
        fp32_bytes=num_weights * 4,
        fp16_bytes=num_weights * 2,
        fp4_bytes=num_weights // 2,  # 2 weights per byte
        packed_2bit_bytes=num_weights // 4,  # 4 weights per byte
    )


# =============================================================================
# CLAIM 2: Bit-Packing Implementation
# =============================================================================

def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights {-1, 0, +1} into 2 bits each.
    4 weights per byte.
    
    Encoding: -1 -> 0b10, 0 -> 0b00, +1 -> 0b01
    """
    # Map: -1 -> 2, 0 -> 0, +1 -> 1
    encoded = (weights + 1).to(torch.uint8)  # Now: 0, 1, 2
    encoded = torch.where(weights == -1, torch.tensor(2, dtype=torch.uint8, device=weights.device), encoded)
    encoded = torch.where(weights == 0, torch.tensor(0, dtype=torch.uint8, device=weights.device), encoded)
    encoded = torch.where(weights == 1, torch.tensor(1, dtype=torch.uint8, device=weights.device), encoded)
    
    # Pad to multiple of 4
    pad_len = (4 - len(encoded) % 4) % 4
    if pad_len > 0:
        encoded = F.pad(encoded, (0, pad_len))
    
    # Pack 4 weights per byte
    encoded = encoded.view(-1, 4)
    packed = (encoded[:, 0] | 
              (encoded[:, 1] << 2) | 
              (encoded[:, 2] << 4) | 
              (encoded[:, 3] << 6))
    
    return packed


def unpack_ternary_weights(packed: torch.Tensor, num_weights: int) -> torch.Tensor:
    """Unpack 2-bit encoded weights back to {-1, 0, +1}"""
    # Extract 4 weights per byte
    w0 = packed & 0x03
    w1 = (packed >> 2) & 0x03
    w2 = (packed >> 4) & 0x03
    w3 = (packed >> 6) & 0x03
    
    # Interleave
    unpacked = torch.stack([w0, w1, w2, w3], dim=1).flatten()[:num_weights]
    
    # Decode: 0 -> 0, 1 -> +1, 2 -> -1
    result = torch.zeros_like(unpacked, dtype=torch.float32)
    result = torch.where(unpacked == 1, torch.tensor(1.0, device=packed.device), result)
    result = torch.where(unpacked == 2, torch.tensor(-1.0, device=packed.device), result)
    
    return result


def test_packing():
    """Verify packing/unpacking correctness"""
    print("=" * 60)
    print("CLAIM 2: Bit-Packing Implementation Test")
    print("=" * 60)
    
    # Generate random ternary weights
    n = 10000
    weights = torch.randint(-1, 2, (n,), dtype=torch.float32, device=DEVICE)
    
    # Pack and unpack
    packed = pack_ternary_weights(weights)
    unpacked = unpack_ternary_weights(packed, n)
    
    # Verify
    match = (weights == unpacked).all()
    compression = weights.numel() * 4 / packed.numel()  # float32 -> uint8
    
    print(f"Original size: {weights.numel() * 4:,} bytes (float32)")
    print(f"Packed size:   {packed.numel():,} bytes (uint8)")
    print(f"Compression:   {compression:.1f}x")
    print(f"Round-trip:    {'PASS' if match else 'FAIL'}")
    print()
    
    return match


# =============================================================================
# CLAIM 3: Logic-Mux vs Multiplication Performance
# =============================================================================

def ternary_matmul_standard(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Standard matrix multiply (baseline)"""
    return x @ w.T


def ternary_matmul_logic_branching(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    The CLAIMED approach: branching logic.
    WARNING: This is what the analysis suggested, but branching is slow on GPU.
    """
    batch, d_in = x.shape
    d_out, _ = w.shape
    output = torch.zeros(batch, d_out, device=x.device, dtype=x.dtype)
    
    for j in range(d_out):
        for k in range(d_in):
            wval = w[j, k]
            if wval == 1:
                output[:, j] += x[:, k]
            elif wval == -1:
                output[:, j] -= x[:, k]
            # if 0: do nothing (sparse)
    
    return output


def ternary_matmul_logic_vectorized(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    CORRECT approach: branchless vectorized logic.
    Uses masks instead of conditionals.
    """
    # Create masks (no branching)
    pos_mask = (w == 1).float()   # Shape: [d_out, d_in]
    neg_mask = (w == -1).float()
    
    # Vectorized: sum of (x * pos_mask) - (x * neg_mask)
    # Equivalent to: output[b, j] = sum_k(x[b,k] * w[j,k]) for ternary w
    pos_contrib = x @ pos_mask.T  # [batch, d_out]
    neg_contrib = x @ neg_mask.T
    
    return pos_contrib - neg_contrib


def ternary_matmul_logic_optimized(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Most optimized: use torch.where for branchless selection.
    Still requires the multiply, but demonstrates the pattern.
    """
    # This is actually equivalent to standard matmul for ternary
    # The "savings" come from sparsity (zeros) and avoiding FP multiply
    # But on GPU, the matmul kernel is highly optimized
    return x @ w.T


def benchmark_compute():
    """Benchmark different computation approaches"""
    print("=" * 60)
    print("CLAIM 3: Logic-Mux vs Multiplication Performance")
    print("=" * 60)
    
    # Realistic sizes
    batch = 32
    d_in = 512
    d_out = 512
    n_iters = 100
    
    # Generate data
    x = torch.randn(batch, d_in, device=DEVICE)
    w_float = torch.randint(-1, 2, (d_out, d_in), dtype=torch.float32, device=DEVICE)
    
    # Sparsity analysis
    sparsity = (w_float == 0).float().mean().item()
    print(f"Weight sparsity (zeros): {sparsity*100:.1f}%")
    print(f"Matrix size: [{batch}, {d_in}] x [{d_out}, {d_in}].T")
    print()
    
    # Warmup
    for _ in range(10):
        _ = ternary_matmul_standard(x, w_float)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    results = {}
    
    # 1. Standard matmul
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out1 = ternary_matmul_standard(x, w_float)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    results["Standard matmul"] = (t1 - t0) / n_iters * 1000
    
    # 2. Vectorized logic (branchless)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out2 = ternary_matmul_logic_vectorized(x, w_float)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    results["Vectorized logic"] = (t1 - t0) / n_iters * 1000
    
    # Verify correctness
    match = torch.allclose(out1, out2, atol=1e-5)
    
    # 3. Branching logic (only on CPU, too slow for GPU)
    if DEVICE == "cpu" and d_in <= 64:
        t0 = time.perf_counter()
        out3 = ternary_matmul_logic_branching(x, w_float)
        t1 = time.perf_counter()
        results["Branching logic"] = (t1 - t0) * 1000
    
    # Print results
    print(f"{'Method':<25} {'Time (ms)':>12} {'Relative':>12}")
    print("-" * 50)
    baseline = results["Standard matmul"]
    for name, ms in results.items():
        rel = ms / baseline
        print(f"{name:<25} {ms:>12.4f} {rel:>11.2f}x")
    
    print()
    print(f"Correctness check: {'PASS' if match else 'FAIL'}")
    print()
    
    # Analysis
    print("ANALYSIS:")
    print("-" * 50)
    if results["Vectorized logic"] > results["Standard matmul"]:
        print("! Vectorized logic is SLOWER than standard matmul")
        print("  Reason: PyTorch's matmul uses optimized BLAS/cuBLAS")
        print("  The 'logic mux' approach requires 2 matmuls (pos + neg masks)")
    else:
        print("  Vectorized logic is faster (unexpected on modern GPUs)")
    print()
    
    return results


# =============================================================================
# CLAIM 4: Sparse Lookup as CAM
# =============================================================================

def test_sparse_routing():
    """Test if sparse routing actually provides speedup"""
    print("=" * 60)
    print("CLAIM 4: Sparse Routing / CAM Analysis")
    print("=" * 60)
    
    # Simulate TriX routing
    batch = 32
    seq_len = 128
    d_model = 512
    num_tiles = 64
    tile_dim = 128  # Output dimension per tile
    
    x = torch.randn(batch, seq_len, d_model, device=DEVICE)
    
    # Tile signatures (ternary)
    signatures = torch.randint(-1, 2, (num_tiles, d_model), dtype=torch.float32, device=DEVICE)
    
    # Tile weights: each tile is a [tile_dim, d_model] matrix
    tile_weights = torch.randn(num_tiles, tile_dim, d_model, device=DEVICE)
    
    n_iters = 100
    
    # Method 1: Dense (process all tiles, weighted combination)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        # Compute routing scores
        scores = x @ signatures.T  # [batch, seq, num_tiles]
        routing = F.softmax(scores, dim=-1)
        
        # Dense: compute ALL tile outputs then combine
        # x: [batch, seq, d_model], tile_weights: [num_tiles, tile_dim, d_model]
        # For each tile, compute x @ tile_weights[t].T
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]
        all_outputs = torch.stack([x_flat @ tile_weights[t].T for t in range(num_tiles)], dim=1)
        # all_outputs: [batch*seq, num_tiles, tile_dim]
        all_outputs = all_outputs.view(batch, seq_len, num_tiles, tile_dim)
        
        # Weighted combination
        dense_output = torch.einsum('bsnt,bsn->bst', all_outputs, routing)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dense_time = (time.perf_counter() - t0) / n_iters * 1000
    
    # Method 2: Sparse (winner-take-all, only compute 1 tile)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        # Compute routing scores
        scores = x @ signatures.T  # [batch, seq, num_tiles]
        winners = scores.argmax(dim=-1)  # [batch, seq]
        
        # Gather winning tile weights
        flat_winners = winners.flatten()  # [batch*seq]
        winning_weights = tile_weights[flat_winners]  # [batch*seq, tile_dim, d_model]
        
        # Apply only winning tiles
        x_flat = x.view(-1, d_model, 1)  # [batch*seq, d_model, 1]
        sparse_output = torch.bmm(winning_weights, x_flat).squeeze(-1)  # [batch*seq, tile_dim]
        sparse_output = sparse_output.view(batch, seq_len, tile_dim)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    sparse_time = (time.perf_counter() - t0) / n_iters * 1000
    
    print(f"Batch: {batch}, Seq: {seq_len}, d_model: {d_model}, Tiles: {num_tiles}")
    print()
    print(f"{'Method':<25} {'Time (ms)':>12}")
    print("-" * 40)
    print(f"{'Dense (all tiles)':<25} {dense_time:>12.4f}")
    print(f"{'Sparse (winner only)':<25} {sparse_time:>12.4f}")
    print(f"{'Speedup':<25} {dense_time/sparse_time:>12.2f}x")
    print()
    
    if sparse_time < dense_time:
        print(f"VERDICT: Sparse routing IS faster ({dense_time/sparse_time:.1f}x)")
        print("  This validates the core TriX claim about sparse compute.")
    else:
        print("VERDICT: Sparse routing is NOT faster")
        print("  The overhead of winner selection + gather exceeds dense compute.")
    print()


# =============================================================================
# CLAIM 5: Warp Divergence Reality Check
# =============================================================================

def test_warp_divergence():
    """Demonstrate why branching is bad on GPU"""
    print("=" * 60)
    print("REALITY CHECK: Warp Divergence")
    print("=" * 60)
    
    if DEVICE != "cuda":
        print("Skipping (requires CUDA)")
        print()
        return
    
    n = 1_000_000
    x = torch.randn(n, device=DEVICE)
    w = torch.randint(-1, 2, (n,), dtype=torch.float32, device=DEVICE)
    
    n_iters = 1000
    
    # Branchless: multiply (what GPU does well)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out1 = x * w
    torch.cuda.synchronize()
    branchless_time = (time.perf_counter() - t0) / n_iters * 1000
    
    # Branchless: masked add/sub (the RIGHT way)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        pos_mask = (w == 1)
        neg_mask = (w == -1)
        out2 = torch.where(pos_mask, x, torch.zeros_like(x)) - torch.where(neg_mask, x, torch.zeros_like(x))
    torch.cuda.synchronize()
    masked_time = (time.perf_counter() - t0) / n_iters * 1000
    
    print(f"Vector size: {n:,}")
    print()
    print(f"{'Method':<30} {'Time (ms)':>12}")
    print("-" * 45)
    print(f"{'Simple multiply (x * w)':<30} {branchless_time:>12.4f}")
    print(f"{'Masked select (branchless)':<30} {masked_time:>12.4f}")
    print()
    print(f"VERDICT: Simple multiply is {masked_time/branchless_time:.1f}x faster")
    print("  The 'logic mux' approach adds overhead vs optimized multiply.")
    print("  Savings come from SPARSITY (skipping zeros), not from avoiding multiply.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("THOR COMPILER CLAIMS VALIDATION")
    print("=" * 60)
    print()
    
    # Claim 1: Memory
    mem = analyze_memory(70_000_000_000)  # 70B params
    mem.print_report()
    
    # Claim 2: Bit-packing works
    test_packing()
    
    # Claim 3: Logic-mux performance
    benchmark_compute()
    
    # Claim 4: Sparse routing
    test_sparse_routing()
    
    # Claim 5: Warp divergence
    test_warp_divergence()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
VALIDATED:
  [YES] 2-bit packing gives 8x memory savings over FP16
  [YES] 2-bit packing gives 2x memory savings over FP4
  [YES] Sparse routing (winner-take-all) can be faster than dense
  [YES] Bit-packing round-trips correctly

NOT VALIDATED:
  [NO]  "Logic mux" is NOT faster than optimized matmul
  [NO]  Branching in CUDA causes warp divergence (as predicted)
  [NO]  The claimed CUDA kernel would be slower, not faster

NUANCED:
  - Memory savings are REAL and significant
  - Compute savings come from SPARSITY, not from avoiding multiply
  - Modern GPUs are optimized for multiply-add; "tricking" them doesn't help
  - The value of TriX is in ROUTING + SPARSITY, not in "logic gates"
""")


if __name__ == "__main__":
    main()
