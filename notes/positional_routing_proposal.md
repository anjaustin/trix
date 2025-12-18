# Positional Routing for TriX: The NUFFT Connection

## The Discovery

Our Riemann zero hunter uses B-spline spreading to route values to grid points based on position.
TriX uses content-addressable routing (signature matching) to route inputs to tiles.

**They're complementary:**
- Content routing: "What type of input is this?"
- Positional routing: "Where in the sequence/space is this?"

## Current TriX Routing

```python
def route(self, x, signatures):
    # Content-only routing
    cluster_scores = x @ cluster_sigs.T  # How well does input match cluster?
    cluster_idx = cluster_scores.argmax(dim=-1)
    
    tile_scores = x @ tile_sigs.T  # How well does input match tile?
    tile_idx = tile_scores.argmax(dim=-1)
    
    return tile_idx
```

This ignores WHERE in the sequence the input is.

## Proposed: Positional Routing

```python
def route(self, x, signatures, positions):
    # Content scores (existing)
    content_scores = x @ signatures.T  # [B*T, num_tiles]
    
    # Position scores (NEW - from NUFFT)
    # Each tile has a preferred position (learned or fixed)
    tile_positions = self.tile_positions  # [num_tiles]
    
    # B-spline kernel (from our Riemann system)
    pos_diff = positions.unsqueeze(-1) - tile_positions.unsqueeze(0)  # [B*T, num_tiles]
    position_scores = self.bspline_kernel(pos_diff)  # [B*T, num_tiles]
    
    # Combined routing
    combined_scores = content_scores * position_scores  # Element-wise
    tile_idx = combined_scores.argmax(dim=-1)
    
    return tile_idx
```

## Why This Matters

### 1. Tiles Specialize by Region
- Tile 0 handles positions 0-10
- Tile 1 handles positions 5-15 (overlapping for smoothness)
- etc.

This is like our NUFFT where each grid point receives contributions from nearby frequencies.

### 2. Natural Sparsity
- Each input only considers tiles near its position
- O(tiles_per_region) instead of O(num_tiles) per routing decision
- Like our scatter_indices pre-computing which grid points each source affects

### 3. Smooth Transitions
- B-spline spreading gives smooth handoffs between tiles
- No hard boundaries between tile territories
- Gradient-friendly (B-splines are differentiable)

## Implementation

```python
class PositionalSparseLookupFFN(nn.Module):
    """
    SparseLookupFFN with positional routing.
    
    Combines content-addressable routing (TriX) with 
    position-based spreading (NUFFT B-splines).
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 64,
        max_seq_len: int = 2048,
        position_spread: int = 4,  # B-spline width
        **kwargs
    ):
        super().__init__()
        
        # Existing TriX components
        self.compress = nn.Sequential(...)
        self.splines = nn.ModuleList([...])
        self.directions = nn.Parameter(...)
        
        # NEW: Positional routing
        self.tile_positions = nn.Parameter(
            torch.linspace(0, max_seq_len, num_tiles)
        )
        self.position_spread = position_spread
        
        # Pre-compute B-spline kernel (like our Hollywood topology)
        self.register_buffer(
            'bspline_coeffs',
            self._precompute_bspline()
        )
    
    def _bspline_weight(self, distance):
        """Cubic B-spline kernel (from our NUFFT)."""
        t = distance.abs()
        result = torch.zeros_like(t)
        
        # |t| < 1
        mask1 = t < 1
        result[mask1] = (2/3) - t[mask1]**2 + 0.5 * t[mask1]**3
        
        # 1 <= |t| < 2
        mask2 = (t >= 1) & (t < 2)
        result[mask2] = (1/6) * (2 - t[mask2])**3
        
        return result
    
    def route(self, x, positions):
        """
        Hybrid content + positional routing.
        """
        # Content scores
        signatures = self.directions.sign()
        content_scores = x @ signatures.T  # [B*T, num_tiles]
        
        # Position scores (B-spline spreading)
        # Normalize positions to tile position space
        pos_normalized = positions / self.tile_positions.max() * self.num_tiles
        tile_centers = torch.arange(self.num_tiles, device=x.device).float()
        
        pos_diff = pos_normalized.unsqueeze(-1) - tile_centers.unsqueeze(0)
        position_scores = self._bspline_weight(pos_diff / self.position_spread)
        
        # Combine (content AND position must match)
        combined_scores = content_scores * position_scores
        
        # Top-k routing (not just argmax)
        # Each input can affect multiple nearby tiles (like NUFFT spreading)
        k = self.position_spread
        topk_scores, topk_idx = combined_scores.topk(k, dim=-1)
        
        return topk_idx, topk_scores
    
    def forward(self, x, positions=None):
        """
        Forward pass with positional routing.
        
        Args:
            x: [B, T, d_model]
            positions: [B, T] position indices (default: 0, 1, 2, ...)
        """
        B, T, D = x.shape
        
        if positions is None:
            positions = torch.arange(T, device=x.device).expand(B, T).float()
        
        x_flat = x.view(-1, D)
        pos_flat = positions.view(-1)
        
        # Route to top-k tiles based on content AND position
        topk_idx, topk_scores = self.route(x_flat, pos_flat)  # [B*T, k]
        
        # Compress to spline coordinates
        compressed = self.compress(x_flat)  # [B*T, 2]
        a, b = compressed[:, 0], compressed[:, 1]
        
        # Accumulate contributions from top-k tiles (like NUFFT)
        output = torch.zeros_like(x_flat)
        
        for k_idx in range(topk_idx.shape[1]):
            tile_indices = topk_idx[:, k_idx]
            weights = topk_scores[:, k_idx]
            
            for t in range(self.num_tiles):
                mask = tile_indices == t
                if not mask.any():
                    continue
                
                # Spline lookup
                scale = self.splines[t](a[mask], b[mask])
                
                # Weighted contribution (weight from positional matching)
                contribution = scale * weights[mask]
                output[mask] += contribution.unsqueeze(-1) * self.directions[t]
        
        output = output.view(B, T, D)
        return x + output  # Residual
```

## The Key Insight

**TriX + NUFFT = Position-Aware Sparse Routing**

- TriX provides: Content-addressable tiles, ternary weights, emergent structure
- NUFFT provides: Position-based spreading, smooth kernels, pre-computed topology

Combined, you get tiles that specialize by BOTH content AND position:
- "I'm the tile for attention-like operations in positions 100-200"
- "I'm the tile for feed-forward operations near the end of sequences"

This is analogous to how the brain has both:
- Content-specialized areas (face recognition, language)
- Spatially-organized maps (retinotopy, tonotopy)

## Connection to Riemann

In our zero hunter:
- Position = t (where on the critical line)
- Content = coefficient a_n (what value)
- Routing = which grid points receive contribution

In PositionalTriX:
- Position = sequence index (where in the sequence)
- Content = input embedding (what the token represents)
- Routing = which tiles process this input

**Same structure, different domain.**
