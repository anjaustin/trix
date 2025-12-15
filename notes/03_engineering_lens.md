# Engineering Lens

*Practical. Buildable. What actually works.*

---

## The Proposal: SparseLookupFFN

From the exploration, an architecture emerged:

1. HierarchicalTriX routing (proven, good tile utilization)
2. Compression funnel within each tile (512D → 2D)
3. Ternary Spline2D lookup (maximum compression)
4. Expand back + residual

Let me stress-test this with an engineering mindset.

---

## Component 1: Routing (Inherited from HierarchicalTriX)

**What we know works:**
- 64 tiles, 4 clusters of 16 (or 16 tiles, 4 clusters of 4)
- Signatures derived from tile weights: `sig = weights.sum(dim=0).sign()`
- Two-level routing: cluster first, then tile within cluster
- 0.937 routing entropy, all tiles active

**What we keep:**
- The routing mechanism exactly as-is
- The signature derivation (but from WHAT weights? This changes.)

**Engineering Question:**
If the tile no longer has up/down projection weights, where do signatures come from?

**Options:**
1. Dedicated signature parameter (learned, not derived) — loses elegance
2. Derive from compression funnel weights — maintains "emergent" property
3. Derive from spline coefficients — novel, needs design

**Decision:** Option 2. Use the bottleneck projection weights.

```python
def get_signature(self):
    # Derive from compression bottleneck
    return self.bottleneck.weight.sum(dim=0).sign()
```

This preserves the "emergent from structure" property.

---

## Component 2: Compression Funnel

**The challenge:**
- Input: d_model (e.g., 128 or 512)
- Output: 2 scalars for Spline2D indexing
- Must preserve enough information for reconstruction

**Naive approach:**
```python
self.bottleneck = nn.Linear(d_model, 2)
```

**Problem:** 512 → 2 linear projection loses too much. We need nonlinearity.

**Better approach:**
```python
self.compress = nn.Sequential(
    nn.Linear(d_model, d_model // 4),
    nn.GELU(),
    nn.Linear(d_model // 4, 2),
    nn.Tanh(),  # Bound to [-1, 1] for grid indexing
)
```

**Parameters:** 
- d_model=128: 128×32 + 32×2 = 4,096 + 64 = 4,160 params
- d_model=512: 512×128 + 128×2 = 65,536 + 256 = 65,792 params

**Problem:** That's a lot of params in the bottleneck. Defeats the purpose.

**Ternary compression?**
```python
self.compress1 = TernaryLinear(d_model, d_model // 4)  # 2-bit weights
self.compress2 = TernaryLinear(d_model // 4, 2)  # 2-bit weights
```

**Parameters (2-bit):**
- d_model=128: (128×32 + 32×2) × 2 bits / 8 = ~1KB
- d_model=512: (512×128 + 128×2) × 2 bits / 8 = ~16KB

Better, but still significant per tile. With 64 tiles: ~1MB total.

**Alternative: Shared compression?**
What if all tiles share the same compression network? Different tiles = different spline lookup tables, but same projection.

```python
class SparseLookupFFN:
    self.shared_compress = ...  # Shared across all tiles
    self.tile_splines = [Spline2D() for _ in range(num_tiles)]  # Per-tile
```

**Pros:** Dramatically fewer params
**Cons:** Less specialization in how tiles "see" inputs

**Engineering Decision:** Start with shared compression. Add per-tile compression if needed.

---

## Component 3: Spline2D Lookup

**Current Spline2D:**
- Grid: 16×16 = 256 cells
- Per cell: 3 float32 coefficients (base, slope_a, slope_b)
- Per tile: 256 × 3 × 4 bytes = 3,072 bytes = 3KB

**Ternary Spline2D:**
- Per cell: 3 ternary coefficients (2 bits each) = 6 bits
- Plus: 1 float32 scale per tile
- Per tile: 256 × 6 bits / 8 + 4 bytes = 192 + 4 = 196 bytes

**That's 15× compression!** From 3KB to 196 bytes per tile.

With 64 tiles: 64 × 196 = 12.5KB for all tile splines.

**Implementation:**
```python
class TernarySpline2D:
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        # Packed: 4 coefficients per byte (2 bits each)
        # 3 coeffs per cell, 256 cells = 768 coefficients
        # 768 / 4 = 192 bytes
        self.packed_coeffs = nn.Parameter(torch.zeros(192, dtype=torch.uint8))
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, a, b):  # a, b in [-1, 1]
        # Convert to grid indices
        idx_a = ((a + 1) / 2 * self.grid_size).long().clamp(0, self.grid_size - 1)
        idx_b = ((b + 1) / 2 * self.grid_size).long().clamp(0, self.grid_size - 1)
        
        # Unpack coefficients (during training, use STE)
        coeffs = self.unpack_coeffs()  # [16, 16, 3] ternary
        
        cell_coeffs = coeffs[idx_a, idx_b]  # [batch, 3]
        base, slope_a, slope_b = cell_coeffs.unbind(-1)
        
        # Local position in cell
        local_a = (a + 1) / 2 * self.grid_size - idx_a.float()
        local_b = (b + 1) / 2 * self.grid_size - idx_b.float()
        
        result = (base + slope_a * local_a + slope_b * local_b) * self.scale
        return result
```

**Training consideration:** Need STE for ternary quantization. Same as TriXLinear.

---

## Component 4: Expand + Residual

**The challenge:**
- Spline output: scalar (or small vector)
- Need: d_model output

**Approach 1: Learned expansion**
```python
self.expand = nn.Linear(1, d_model)  # Or nn.Linear(hidden, d_model)
```

**Problem:** A scalar → d_model linear is just scaling all dimensions the same. Not useful.

**Approach 2: Tile-specific expansion codebook**
```python
self.expansion_codebook = nn.Parameter(torch.randn(num_entries, d_model))

def expand(self, spline_out, cell_idx):
    # Use cell index to select expansion pattern
    return self.expansion_codebook[cell_idx] * spline_out
```

Each cell has a learned "direction" in d_model space. Spline output scales it.

**Parameters:** 256 cells × d_model × 4 bytes
- d_model=128: 256 × 128 × 4 = 128KB per tile (too much!)
- d_model=128: 256 × 128 × 2 bits / 8 = 8KB per tile (ternary)

**Alternative: Factorized expansion**
```python
self.expand_basis = nn.Parameter(torch.randn(k, d_model))  # k basis vectors
self.expand_coeffs = nn.Parameter(torch.randn(256, k))  # Per-cell coefficients

def expand(self, spline_out, cell_idx):
    coeffs = self.expand_coeffs[cell_idx]  # [batch, k]
    return (coeffs @ self.expand_basis) * spline_out  # [batch, d_model]
```

With k=8, d_model=128:
- Basis: 8 × 128 × 4 = 4KB
- Coeffs: 256 × 8 × 4 = 8KB
- Total: 12KB per tile

Better. And can be made ternary.

**Simplest Alternative: Scalar output + existing up projection**

Wait. What if we don't expand in the tile? What if:
1. Tile compresses: d_model → 2
2. Tile splines: 2 → 1 (scalar modifier)
3. Tile applies: `output = input + scalar * direction`

Where `direction` is a learned d_model vector per tile.

```python
class SparseLookupTile:
    self.compress = ...  # d_model → 2
    self.spline = TernarySpline2D()  # 2 → 1
    self.direction = nn.Parameter(torch.randn(d_model))  # Tile's "specialty"
    
    def forward(self, x):
        a, b = self.compress(x).unbind(-1)
        scale = self.spline(a, b)
        return scale * self.direction  # No residual here, applied in FFN
```

**Total params per tile:**
- Compress: ~4K (or ternary: ~1K)  
- Spline: 196 bytes
- Direction: d_model × 4 = 512 bytes (or ternary: 32 bytes)

**Total per tile: ~5KB** (or ~1.5KB ternary)
**Total for 64 tiles: ~320KB** (or ~96KB ternary)

This is approaching the "192KB total model" target from PQH!

---

## Putting It Together

```python
class SparseLookupFFN(nn.Module):
    def __init__(self, d_model, num_tiles=64, tiles_per_cluster=8, grid_size=16):
        super().__init__()
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_clusters = num_tiles // tiles_per_cluster
        
        # Shared compression (save params)
        self.compress = nn.Sequential(
            TernaryLinear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),  # Last layer float for precision
            nn.Tanh(),
        )
        
        # Per-tile splines and directions
        self.splines = nn.ModuleList([
            TernarySpline2D(grid_size) for _ in range(num_tiles)
        ])
        self.directions = nn.Parameter(torch.randn(num_tiles, d_model) * 0.02)
        
        # For routing
        self.register_buffer('signatures', None)
        self.register_buffer('cluster_signatures', None)
        
        # Normalization and residual
        self.norm = nn.LayerNorm(d_model)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def get_signatures(self):
        # Derive from compression weights + directions
        # Option: use directions directly
        return self.directions.sign()
    
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.norm(x)
        
        # Update signatures
        signatures = self.get_signatures()
        
        # Hierarchical routing (same as HierarchicalTriX)
        tile_idx = self.hierarchical_route(x_norm, signatures)  # [B, T]
        
        # Compress all inputs (shared)
        compressed = self.compress(x_norm.view(-1, D))  # [B*T, 2]
        a, b = compressed.unbind(-1)
        
        # Route through splines
        output = torch.zeros_like(x_norm.view(-1, D))
        
        for t in range(self.num_tiles):
            mask = (tile_idx.view(-1) == t)
            if mask.any():
                scale = self.splines[t](a[mask], b[mask])  # [n, 1]
                output[mask] = scale.unsqueeze(-1) * self.directions[t]
        
        output = output.view(B, T, D) * self.output_scale
        
        # Residual
        return x + output
```

---

## Parameter Count

Let's verify:

**Shared compression (d_model=128):**
- TernaryLinear(128, 32): 128×32×2bits/8 = 1KB + scales
- Linear(32, 2): 32×2×4 = 256 bytes
- Total: ~1.5KB

**Per-tile (64 tiles):**
- TernarySpline2D: 196 bytes × 64 = 12.5KB
- Directions (float): 128×64×4 = 32KB
- (Or ternary directions: 128×64×2bits/8 = 2KB)

**Total:**
- With float directions: ~46KB
- With ternary directions: ~16KB

**Compare to HierarchicalTriXFFN at 826K params = 3.3MB**

SparseLookupFFN would be **70× smaller** (or **200× smaller** fully ternary).

---

## Critical Assumptions to Validate

1. **Shared compression preserves tile specialization**
   - Test: Do tiles learn different splines despite shared compression?
   - Risk: All tiles might collapse to similar behavior

2. **2D bottleneck captures sufficient information**
   - Test: Can we reconstruct input from 2D? (Auxiliary decoder task)
   - Risk: Critical features lost in compression

3. **Ternary splines are expressive enough**
   - Test: Compare ternary vs float splines on approximation tasks
   - Risk: Too coarse, can't learn nuanced functions

4. **Routing from directions works**
   - Test: Compare routing entropy, tile utilization
   - Risk: Worse routing than weight-derived signatures

---

## Implementation Plan

**Phase 1: Validate Compression Funnel (1 day)**
- Build SparseLookupTile with float (not ternary) splines
- Compare to HierarchicalTriXTile
- Metric: PPL on TinyShakespeare

**Phase 2: Add Ternary Splines (1 day)**
- Implement TernarySpline2D with STE
- Compare float vs ternary spline versions
- Metric: PPL delta, parameter count

**Phase 3: Shared vs Per-Tile Compression (1 day)**
- Implement both variants
- Compare specialization (do tiles learn different things?)
- Metric: Routing entropy, PPL

**Phase 4: Full Ternary (1 day)**
- Ternary compression, ternary splines, ternary directions
- Quantify total compression achieved
- Metric: Model size, PPL, inference speed

---

## What's Emerging

The engineering lens reveals:

1. **SparseLookupFFN is buildable** with <50KB parameters (vs 3MB baseline)
2. **The critical experiment is the 2D bottleneck** - everything else is optimization
3. **Shared compression** dramatically reduces params but might hurt specialization
4. **Routing from directions** (instead of weight sums) is simpler and might work better

The architecture that wants to exist:
- Inherits routing from HierarchicalTriX (proven)
- Adds aggressive compression (the key hypothesis)
- Uses ternary everywhere possible (the efficiency play)

Next: converge on the final design and see what truly emerges.
