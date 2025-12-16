# TriX Optimization Plan for Jetson AGX Thor (Blackwell)

## Executive Summary

**Forget NEON. Use INT8 Tensor Cores.**

Thor's Blackwell GPU delivers:
- **211 TFLOPS FP8**
- **205 TFLOPS INT8** 
- **112 TFLOPS BF16**

TriX ternary weights `{-1, 0, +1}` map directly to INT8. This is the path.

---

# NEON Optimization Plan (DEPRECATED for Thor)

## Current State

### Benchmark Results (CPU, batch=64)

| Size | PyTorch Dense | NEON Packed | Slowdown |
|------|---------------|-------------|----------|
| 512x512 | 0.37 ms | 3.67 ms | 10x slower |
| 1024x1024 | 1.01 ms | 20.7 ms | 20x slower |
| 2048x2048 | 2.07 ms | 89.6 ms | 43x slower |

### Root Causes

1. **Inefficient weight decoding** in inner loop:
```cpp
// CURRENT: Decode weights one-by-one, create float array, then load
for (int i = 0; i < 4; i++) {
    uint8_t code = (packed >> (i * 2)) & 0x03;
    weights[i] = (code == 0x01) ? 1.0f : (code == 0x02) ? -1.0f : 0.0f;
}
float32x4_t w_vec = vld1q_f32(weights);  // Extra load!
```

2. **No parallelism across output dimension**
3. **No memory prefetching**
4. **Single-threaded execution**

---

## Optimization 1: NEON Lookup Table for Weight Decoding

Replace scalar decode with vector table lookup:

```cpp
// OPTIMIZED: Use NEON vtbl to decode 4 weights in parallel
static const int8_t WEIGHT_LUT[4] = {0, 1, -1, 0};  // code -> weight
static const int8x8_t lut = vld1_s8(WEIGHT_LUT_EXPANDED);

// Decode all 4 weights at once using bit manipulation + lookup
uint8x8_t codes = ...;  // Extract 2-bit codes
int8x8_t weights = vtbl1_s8(lut, codes);
```

**Expected speedup: 2-3x** for decoding step

---

## Optimization 2: Process Multiple Outputs in Parallel

Current code processes one output at a time. NEON can process 4 outputs simultaneously:

```cpp
// OPTIMIZED: Compute 4 outputs in parallel
float32x4_t acc0 = vdupq_n_f32(0);
float32x4_t acc1 = vdupq_n_f32(0);
float32x4_t acc2 = vdupq_n_f32(0);
float32x4_t acc3 = vdupq_n_f32(0);

for (int p = 0; p < packed_len; p++) {
    float32x4_t in_vec = vld1q_f32(&input[p * 4]);
    
    // Load weights for 4 output rows
    float32x4_t w0 = decode_weights(packed_w[0][p]);
    float32x4_t w1 = decode_weights(packed_w[1][p]);
    float32x4_t w2 = decode_weights(packed_w[2][p]);
    float32x4_t w3 = decode_weights(packed_w[3][p]);
    
    acc0 = vmlaq_f32(acc0, in_vec, w0);
    acc1 = vmlaq_f32(acc1, in_vec, w1);
    acc2 = vmlaq_f32(acc2, in_vec, w2);
    acc3 = vmlaq_f32(acc3, in_vec, w3);
}
```

**Expected speedup: 2-4x** (better ILP and cache utilization)

---

## Optimization 3: Memory Prefetching

Add prefetch hints for weights and inputs:

```cpp
// Prefetch next cache line of weights
__builtin_prefetch(&packed_w[(o + 4) * packed_in_dim], 0, 3);

// Prefetch next input (for batch processing)
__builtin_prefetch(&input[(b + 1) * in_features], 0, 3);
```

**Expected speedup: 1.2-1.5x** (depends on memory access pattern)

---

## Optimization 4: Multi-threaded Execution

Use OpenMP for parallel batch processing:

```cpp
#pragma omp parallel for
for (int b = 0; b < batch_size; b++) {
    // Process each batch element independently
}
```

**Expected speedup: 4-8x** (Thor has 12 ARM cores)

---

## Optimization 5: Specialized Ternary Operations

Since weights are only {-1, 0, +1}, we can use integer ops:

```cpp
// Instead of: acc += input[i] * weight[i]
// Use conditional add/subtract based on 2-bit code:

// code=01 (+1): acc += input
// code=10 (-1): acc -= input  
// code=00 (0):  skip

// This avoids float multiply entirely!
if (code == 0x01) acc += input[i];
else if (code == 0x02) acc -= input[i];
```

Already implemented but not vectorized. Can use `vbslq_f32` (bit select) for branchless:

```cpp
// NEON branchless ternary multiply
uint32x4_t is_pos = vceqq_u32(codes, vdupq_n_u32(1));
uint32x4_t is_neg = vceqq_u32(codes, vdupq_n_u32(2));
float32x4_t pos_contrib = vbslq_f32(is_pos, in_vec, vdupq_n_f32(0));
float32x4_t neg_contrib = vbslq_f32(is_neg, vnegq_f32(in_vec), vdupq_n_f32(0));
acc = vaddq_f32(acc, vaddq_f32(pos_contrib, neg_contrib));
```

**Expected speedup: 1.5-2x** (eliminates float multiply)

---

## Combined Expected Speedup

| Optimization | Individual | Cumulative |
|--------------|------------|------------|
| Baseline (current) | 1x | 1x |
| LUT decoding | 2-3x | 2-3x |
| Output parallelism | 2-4x | 4-12x |
| Prefetching | 1.2-1.5x | 5-18x |
| Multi-threading | 4-8x | 20-144x |
| Ternary specialization | 1.5-2x | 30-288x |

**Realistic target: 20-50x improvement** (reaching parity with dense PyTorch)

---

## Alternative: CUDA Kernel

However, the bigger opportunity is **CUDA**, not NEON:

| Approach | Throughput | Effort |
|----------|------------|--------|
| Current NEON | 1x | Done |
| Optimized NEON | 20-50x | High |
| CUDA kernel | 100-200x | Medium |

Thor's GPU is vastly more powerful than its CPU. A simple CUDA kernel for sparse ternary matmul would outperform even heavily optimized NEON.

### Recommended CUDA Approach

```cpp
__global__ void trix_forward_cuda(
    const float* __restrict__ input,      // [batch, in_features]
    const uint8_t* __restrict__ packed_w, // [out_features, packed_in]
    const float* __restrict__ scales,     // [out_features]
    const int8_t* __restrict__ gate,      // [batch, num_tiles]
    float* __restrict__ output,           // [batch, out_features]
    int batch_size, int in_features, int out_features, int num_tiles
) {
    // Each thread handles one output element
    int b = blockIdx.x;
    int o = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (o >= out_features) return;
    
    // Check if this tile is active
    int tile = o / (out_features / num_tiles);
    if (gate[b * num_tiles + tile] == 0) {
        output[b * out_features + o] = 0;
        return;
    }
    
    // Compute dot product with packed ternary weights
    float acc = 0;
    int packed_in = (in_features + 3) / 4;
    
    for (int p = 0; p < packed_in; p++) {
        uint8_t packed = packed_w[o * packed_in + p];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = p * 4 + i;
            if (idx < in_features) {
                uint8_t code = (packed >> (i * 2)) & 0x03;
                if (code == 0x01) acc += input[b * in_features + idx];
                else if (code == 0x02) acc -= input[b * in_features + idx];
            }
        }
    }
    
    output[b * out_features + o] = acc * scales[o];
}
```

---

## Recommendation

**For Thor specifically:**

1. **Short-term:** Implement CUDA kernel (1-2 days)
   - 100x faster than current NEON
   - Works with existing training pipeline

2. **Medium-term:** Optimize NEON for edge deployment
   - For devices without GPU (Jetson Nano, Raspberry Pi)
   - Lower priority for Thor

3. **Long-term:** Integrate with cuSPARSE or custom tensor cores
   - Maximum performance for production

---

## Next Steps

1. [ ] Implement basic CUDA kernel
2. [ ] Benchmark CUDA vs dense PyTorch
3. [ ] If CUDA is competitive, integrate into SparseLookupFFN
4. [ ] Optimize NEON as secondary target

---

# INT8 Tensor Core Strategy (RECOMMENDED for Thor)

## Why INT8?

| Precision | TFLOPS | TriX Fit |
|-----------|--------|----------|
| FP32 | 57 | Baseline |
| FP16 | 106 | Good |
| BF16 | 112 | Good |
| **INT8** | **205** | **Perfect** - ternary {-1,0,+1} |
| FP8 | 211 | Overkill |

TriX weights are `{-1, 0, +1}`. INT8 represents this exactly with 200+ TFLOPS throughput.

## Implementation Plan

### Phase 1: INT8 Ternary Linear (Drop-in Replacement)

```python
class TernaryLinearINT8(nn.Module):
    """
    Ternary linear layer using INT8 tensor cores.
    
    Weights stored as INT8 {-1, 0, +1}.
    Uses torch._int_mm for tensor core acceleration.
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        # Trainable float weights (quantized during forward)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features))
        
        # Cached INT8 weights for inference
        self.register_buffer('weight_int8', None)
    
    def quantize_weights(self):
        """Quantize to ternary INT8."""
        w = self.weight.data
        w_ternary = torch.zeros_like(w, dtype=torch.int8)
        w_ternary[w > 0.5] = 1
        w_ternary[w < -0.5] = -1
        self.weight_int8 = w_ternary
    
    def forward(self, x):
        if self.training:
            # STE quantization for training
            w = ste_ternary(self.weight)
            return F.linear(x, w) * self.scale
        else:
            # INT8 tensor core inference
            if self.weight_int8 is None:
                self.quantize_weights()
            
            # Quantize input to INT8
            x_scale = x.abs().max() / 127
            x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
            
            # INT8 matmul (205 TFLOPS!)
            out_int32 = torch._int_mm(x_int8, self.weight_int8.t())
            
            # Dequantize
            return out_int32.float() * x_scale * self.scale
```

### Phase 2: Sparse INT8 Routing

The real win is combining INT8 with sparse routing:

```python
def sparse_int8_forward(x, weights_int8, tile_indices, scales):
    """
    Sparse forward with INT8 tensor cores.
    
    Only computes the selected tile per token.
    64 tiles = 64x compute reduction.
    """
    batch, seq, d_model = x.shape
    num_tiles = weights_int8.shape[0] // d_model
    tile_size = d_model
    
    # Gather weights for selected tiles
    # tile_indices: [batch, seq] -> which tile each token uses
    selected_weights = weights_int8[tile_indices * tile_size : (tile_indices + 1) * tile_size]
    
    # Batched INT8 matmul for each unique tile
    # (This is where custom CUDA kernel would help)
    outputs = []
    for tile_id in tile_indices.unique():
        mask = tile_indices == tile_id
        x_tile = x[mask]  # Tokens routed to this tile
        w_tile = weights_int8[tile_id * tile_size:(tile_id + 1) * tile_size]
        
        x_int8 = quantize_to_int8(x_tile)
        out = torch._int_mm(x_int8, w_tile.t())
        outputs.append((mask, out))
    
    # Scatter back
    return scatter_outputs(outputs, x.shape)
```

### Phase 3: Fused CUDA Kernel (Maximum Performance)

For peak performance, fuse routing + sparse matmul:

```cpp
// CUDA kernel: sparse ternary matmul with INT8 tensor cores
__global__ void sparse_ternary_matmul_int8(
    const int8_t* __restrict__ x,        // [batch*seq, d_model]
    const int8_t* __restrict__ weights,  // [num_tiles, tile_size, d_model]
    const int32_t* __restrict__ routes,  // [batch*seq] tile index per token
    const float* __restrict__ scales,    // [num_tiles, tile_size]
    float* __restrict__ output           // [batch*seq, d_model]
) {
    // Use wmma (Warp Matrix Multiply Accumulate) for INT8
    // Each warp handles 16x16 tile of output
    // Only compute the routed tile
}
```

## Expected Performance

| Approach | Tokens/sec | vs Current |
|----------|------------|------------|
| Current PyTorch (FP32) | 40,000 | 1x |
| INT8 Dense | 150,000 | 3.7x |
| INT8 Sparse (64 tiles) | 1,000,000+ | 25x+ |
| Fused CUDA Kernel | 2,000,000+ | 50x+ |

## Training Strategy

For training on Thor:

1. **Use BF16 AMP** for forward/backward (112 TFLOPS)
2. **STE quantization** in forward pass
3. **INT8 inference** for validation (205 TFLOPS)

```python
# Training loop
model.train()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits, aux = model(x)  # BF16 compute
    loss = criterion(logits, targets) + aux
loss.backward()

# Validation
model.eval()
model.quantize_to_int8()  # Convert weights
with torch.no_grad():
    logits = model(x)  # INT8 tensor cores
```

## Memory Analysis

| Component | FP32 | INT8 | Savings |
|-----------|------|------|---------|
| Weights (25M params) | 100 MB | 25 MB | 4x |
| Activations | Depends on batch | Same | - |
| Total model | ~100 MB | ~25 MB | 4x |

With Thor's 128GB, memory isn't the bottleneck. INT8 is about **compute throughput**.

## Next Steps

1. [ ] Implement `TernaryLinearINT8` module
2. [ ] Integrate with SparseLookupFFN
3. [ ] Benchmark INT8 vs FP32 training throughput
4. [ ] Implement sparse routing kernel if needed
5. [ ] Profile and optimize memory access patterns

## NEON: When to Use

NEON optimization is only relevant for:
- Jetson Nano (no tensor cores)
- Raspberry Pi 5
- Other CPU-only edge devices

For Thor, **always use INT8 tensor cores**.
