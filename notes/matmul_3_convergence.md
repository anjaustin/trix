# MatMul Exploration - Phase 3: Convergence

**What crystallizes. The plan.**

---

## The Core Insight

**We already have the engine. We already proved it works.**

FFT and MatMul are both instances of:
```
Permutation × Block-Diagonal × Permutation × Block-Diagonal × ...
```

FFT: We implemented this with twiddle opcodes. 0.00 error.
MatMul: Same structure, different block contents.

The generalization isn't a leap. It's a cartridge swap.

---

## What Crystallizes

### Insight 1: Routing IS the Linear Algebra

A permutation matrix P is just routing:
```python
# Matrix view: y = P @ x
# Routing view: y[i] = x[perm[i]]
```

We've been doing this. The partner selection in FFT. The tile routing in Hollywood Squares. It's all permutation matrices in disguise.

### Insight 2: Block-Diagonal IS Tiles

A block-diagonal matrix applies independent operations to chunks:
```python
# Matrix view: y = (B₁ ⊕ B₂ ⊕ ...) @ x
# Tile view: y[chunk_i] = tile_i(x[chunk_i])
```

The Hollywood Squares topology IS block-diagonal computation.

### Insight 3: The Ternary Path

For exact computation:
- Constrain block values to {-1, 0, +1}
- Like our FP4 atoms
- Trades expressiveness for exactness

For approximate computation:
- Allow richer block values (FP4, INT4)
- Standard quantization
- Good enough for inference

**Start exact. Relax if needed.**

---

## The Concrete Plan

### Phase 1: Butterfly Layer (Today)

Build a single butterfly layer:

```python
class ButterflyLayer:
    """
    One stage of butterfly computation.
    
    For N inputs:
    - Pair elements at distance stride
    - Apply 2×2 block to each pair
    - Routing is structural (like FFT partner selection)
    """
    
    def __init__(self, N, stride, blocks):
        self.N = N
        self.stride = stride
        self.blocks = blocks  # N/2 blocks of 2×2
    
    def forward(self, x):
        y = x.clone()
        for i in range(self.N // 2):
            # Routing: which elements to pair
            idx_a = ...  # structural, like FFT
            idx_b = ...
            
            # Block operation
            a, b = x[idx_a], x[idx_b]
            block = self.blocks[i]
            y[idx_a] = block[0,0]*a + block[0,1]*b
            y[idx_b] = block[1,0]*a + block[1,1]*b
        
        return y
```

### Phase 2: Verify on Known Matrices

Test cases where we know the answer:

| Matrix | Expected Error |
|--------|---------------|
| Identity | 0.00 |
| Permutation | 0.00 |
| DFT | 0.00 (matches our FFT) |
| Hadamard | 0.00 (matches our WHT) |
| Random | Measure |

### Phase 3: Ternary Blocks

Constrain blocks to ternary:

```python
TERNARY_BLOCKS = {
    'identity': [[1, 0], [0, 1]],
    'swap': [[0, 1], [1, 0]],
    'add': [[1, 1], [1, -1]],  # Hadamard-like
    'sub': [[1, -1], [1, 1]],
    # ... enumerate useful 2×2 ternary matrices
}
```

How many 2×2 ternary matrices are there?
- 3^4 = 81 total
- Many are equivalent up to scaling
- Probably ~20-30 useful ones

This is like our twiddle opcode table, but for MatMul.

### Phase 4: Monarch Layer

Generalize to arbitrary block sizes:

```python
class MonarchLayer:
    def __init__(self, N, block_size):
        self.N = N
        self.block_size = block_size
        self.num_blocks = N // block_size
        self.blocks = [...]  # num_blocks blocks of block_size × block_size
    
    def forward(self, x):
        x = x.reshape(-1, self.num_blocks, self.block_size)
        for i, block in enumerate(self.blocks):
            x[:, i, :] = x[:, i, :] @ block
        return x.reshape(-1, self.N)
```

### Phase 5: TriX Butterfly MLP

Replace transformer FFN:

```python
class TriXButterflyMLP:
    def __init__(self, d_model, num_stages=None):
        if num_stages is None:
            num_stages = int(log2(d_model))
        
        self.stages = [
            ButterflyLayer(d_model, 2**i, blocks_i)
            for i in range(num_stages)
        ]
    
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
            x = F.gelu(x)  # or other activation
        return x
```

---

## Success Criteria

### Minimum Viable

1. Butterfly layer runs
2. Identity matrix: 0.00 error
3. DFT matrix: matches our FFT implementation

### Good

4. Hadamard matrix: matches our WHT
5. Random matrix: measure approximation quality
6. Ternary blocks: identify useful set

### Great

7. Monarch layer with variable block size
8. TriX Butterfly MLP prototype
9. Compare efficiency vs dense MLP

### Transformative

10. Train small transformer with butterfly MLPs
11. Measure accuracy/efficiency tradeoff
12. Demonstrate MatMul-free inference

---

## The One-Line Summary

> **MatMul is FFT with different blocks.**

Same routing. Same structure. Different opcodes.

We load a new cartridge into the ray-gun.

---

## Next Action

Start coding `ButterflyLayer`. Test on identity. Then DFT. Then measure random.

The path is clear.

---

*End of convergence phase.*

*Ready to build.*
