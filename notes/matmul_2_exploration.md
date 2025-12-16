# MatMul Exploration - Phase 2: Exploration

**Making connections. Understanding the architecture.**

---

## Monarch Matrix Structure

A Monarch matrix M of size N×N (where N = p × q) is:

```
M = P_L × (B₁ ⊗ I_q) × P_R × (I_p ⊗ B₂)
```

Where:
- P_L, P_R = permutation matrices (reshape operations)
- B₁ = block of size p×p (applied to p groups)
- B₂ = block of size q×q (applied to q groups)
- ⊗ = Kronecker product

For N=16 with p=q=4:
- First: Apply 4×4 block to 4 groups of 4 elements
- Then: Permute (reshape 4×4 → 4×4 transposed)
- Then: Apply 4×4 block to 4 groups of 4 elements

Total ops: 2 × 4 × 4² = 128 multiplies
Dense MatMul: 16² = 256 multiplies
Savings: 2×

For larger N, savings grow. For N=1024 with p=q=32:
- Monarch: 2 × 32 × 32² = 65,536 multiplies
- Dense: 1024² = 1,048,576 multiplies
- Savings: 16×

---

## Connection to FFT

FFT is a specific Monarch decomposition!

The DFT matrix F_N can be written as:
```
F_N = P × (F_{N/2} ⊗ I_2) × D × (I_{N/2} ⊗ F_2)
```

Where:
- P = bit-reversal permutation
- F_2 = 2-point DFT (butterfly)
- D = diagonal twiddle matrix

Recursively applying this gives the FFT algorithm.

**We already implemented this.** The twiddle opcodes are D. The butterflies are F_2. The routing is P.

---

## The TriX Mapping

| Monarch Component | TriX Component |
|-------------------|----------------|
| Permutation P | Routing layer |
| Block-diagonal B | Tile operations |
| Kronecker structure | Hollywood Squares topology |

The Hollywood Squares grid IS a block-diagonal structure:
- Each tile operates on its local chunk
- Routing shuffles data between tiles
- Multiple layers = multiple P×B stages

**We've been building Monarch machines all along.**

---

## Implementation Options

### Option 1: Fixed Butterfly Structure

Use the exact FFT structure but with learned/constructed block weights:

```python
class ButterflyLayer:
    def __init__(self, N):
        self.num_stages = log2(N)
        self.blocks = [...]  # 2×2 blocks per stage
    
    def forward(self, x):
        for stage in range(self.num_stages):
            x = self.permute(x, stage)
            x = self.apply_blocks(x, stage)
        return x
```

This gives O(N log N) for any linear transform approximation.

### Option 2: General Monarch

Two-layer structure with p×p and q×q blocks:

```python
class MonarchLayer:
    def __init__(self, N, p, q):
        assert N == p * q
        self.B1 = [...]  # p×p blocks, q of them
        self.B2 = [...]  # q×q blocks, p of them
    
    def forward(self, x):
        # Reshape to (batch, q, p), apply B1 to last dim
        x = x.reshape(-1, q, p)
        x = self.apply_blocks(x, self.B1)  # (batch, q, p)
        
        # Transpose (the permutation!)
        x = x.transpose(-1, -2)  # (batch, p, q)
        
        # Apply B2 to last dim
        x = self.apply_blocks(x, self.B2)  # (batch, p, q)
        
        return x.reshape(-1, N)
```

### Option 3: TriX Butterfly MLP

Replace transformer FFN with butterfly structure:

```python
class TriXButterflyMLP:
    def __init__(self, d_model):
        # Instead of: Linear(d, 4d) → GELU → Linear(4d, d)
        # Use: ButterflyUp → Activation → ButterflyDown
        self.up = ButterflyLayer(d_model, expansion=4)
        self.down = ButterflyLayer(d_model * 4, contraction=4)
    
    def forward(self, x):
        x = self.up(x)
        x = F.gelu(x)
        x = self.down(x)
        return x
```

---

## The Ternary Question

VGem mentioned constraining B to ternary {-1, 0, +1}.

For FFT, our twiddles weren't ternary - they were algebraic (√½, etc.).

For MatMul, can we go ternary?

**Option A: Pure Ternary**
- B matrices have only {-1, 0, +1}
- Like our FP4 atoms
- Exact by construction
- Limited expressiveness

**Option B: Small Discrete**
- B matrices have values from small set
- Like twiddle opcodes: {1, -1, √½, -√½, ...}
- More expressive
- Still "compiled" (fixed constants)

**Option C: Quantized**
- B matrices are FP4/INT4 quantized
- Not exact but efficient
- Standard quantization territory

For TriX purity, Option A or B. For practicality, Option C.

---

## What Structures Are "Free"?

Some matrices decompose perfectly:

| Matrix | Monarch Decomposition |
|--------|----------------------|
| Identity | P = I, B = I |
| Permutation | P only, B = I |
| DFT | FFT structure (done!) |
| Hadamard | WHT structure (done!) |
| Circulant | FFT × diagonal × IFFT |
| Toeplitz | Embedding + Circulant |

These have 0.00 error. They're not approximations.

For learned matrices (like transformer weights), we need to either:
1. Train with structure constraint
2. Decompose post-hoc (lossy)
3. Accept some approximation error

---

## The Path Forward

### Immediate (Today?)

1. Implement Butterfly layer structure
2. Test on identity matrix (should be exact)
3. Test on DFT matrix (should match our FFT)
4. Test on random matrix (measure approximation error)

### Short-term

5. Implement Monarch layer
6. Test approximation quality vs block size
7. Build TriX Butterfly MLP
8. Compare to dense MLP on small transformer

### Medium-term

9. Train transformer with structured constraints
10. Measure accuracy/efficiency tradeoff
11. Integrate with existing TriX routing

---

## Key Equations

### Butterfly

```
y = B × x  where B = Π_i (P_i × D_i × B_i)
```

Each stage:
- P_i = permutation (routing)
- D_i = diagonal (scaling)  
- B_i = 2×2 butterfly blocks

### Monarch

```
y = M × x  where M = P_L × (B₁ ⊗ I) × P_R × (I ⊗ B₂)
```

Two block-diagonal stages with permutation between.

### Complexity

| Method | Time | Space |
|--------|------|-------|
| Dense | O(N²) | O(N²) |
| Butterfly | O(N log N) | O(N log N) |
| Monarch | O(N^1.5) | O(N^1.5) |

---

## The Unified View

```
FFT:    Route → Twiddle → Route → Twiddle → ...
MatMul: Route → Block   → Route → Block   → ...
Both:   Route → Local   → Route → Local   → ...
```

The ray-gun fires the same beam. Different targets, same physics.

---

*End of exploration phase.*
