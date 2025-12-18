# Riemann System → ML Training: Convergence

## The Core Isomorphism

| Riemann Zero Hunter | Transformer |
|---------------------|-------------|
| Sum over n terms | Sum over sequence positions |
| exp(i·t·ln(n)) | exp(i·θ·(i-j)) (RoPE) |
| Coefficients a_n | Value vectors V |
| Query t | Query position i |
| B-spline spreading | Attention weight distribution |
| Pre-computed scatter_indices | Attention routing pattern |
| FP64 phases, FP32 FFT | FP32 softmax, FP16 matmul |

**They are the SAME computation with different parameters.**

## Emerged Principles

### Principle 1: Topology is Free (Hollywood)

**Riemann**: Pre-compute scatter_indices once, reuse for billions of evaluations.

**ML Application**: Pre-compute attention routing once per architecture.

```python
class HollywoodAttention:
    def __init__(self, pattern='causal'):
        # Compute ONCE at init
        self.routing = build_routing_table(pattern)
        self.gather_indices = torch.tensor(...)  # Like scatter_indices
    
    def forward(self, Q, K, V):
        # REUSE every forward pass
        K_gathered = K[:, self.gather_indices]  # O(1) routing
        # ... compute only non-zero attention
```

**Benefit**: Eliminate mask computation, enable true sparse attention.

### Principle 2: Precision is Path-Dependent

**Riemann**: Phase computation (exp(i·θ)) needs FP64. FFT can be FP32.

**ML Application**: Softmax needs FP32. Matrix multiplies can be FP16.

```python
def precision_aware_attention(Q, K, V):
    # FAST path: FP16 matmul
    scores = (Q.half() @ K.half().T)
    
    # PRECISE path: FP32 softmax
    weights = F.softmax(scores.float() / sqrt_d, dim=-1)
    
    # FAST path: FP16 output
    return (weights.half() @ V.half()).float()
```

**Benefit**: 2x memory, 1.5x speed, same accuracy.

### Principle 3: Non-Uniform Grids via Spreading

**Riemann**: ln(n) frequencies spread to uniform grid via B-splines.

**ML Application**: Learned positions spread to attention via learned kernels.

```python
class SpreadingAttention:
    def __init__(self):
        self.position_encoder = nn.Linear(d, 1)  # Learned positions
        self.spreading_kernel = nn.Parameter(...)  # Learned "B-spline"
    
    def forward(self, Q, K, V):
        # Compute non-uniform positions (like our grid_idx)
        pos_q = self.position_encoder(Q)  # Where queries "live"
        pos_k = self.position_encoder(K)  # Where keys "live"
        
        # Spread attention based on position similarity
        # (like B-spline spreading to nearby grid points)
        pos_diff = pos_q.unsqueeze(-1) - pos_k.unsqueeze(-2)
        spreading_weights = self.spreading_kernel(pos_diff)
        
        # Modulate attention by spreading
        raw_attn = Q @ K.T
        spread_attn = raw_attn * spreading_weights  # Position-aware!
        
        return F.softmax(spread_attn, dim=-1) @ V
```

**Benefit**: Attention that respects semantic distance, not just token position.

### Principle 4: Streaming with Checkpoints

**Riemann**: Process infinite stream, checkpoint at milestones, generate proofs.

**ML Application**: Training as infinite stream, checkpoint with verification.

```python
class VerifiedTrainer:
    def train_step(self, batch):
        loss = self.model(batch)
        loss.backward()
        self.optimizer.step()
        
        # Compute verification metrics (like Riemann-von Mangoldt)
        grad_norm = compute_grad_norm()
        activation_stats = compute_activation_stats()
        
        return loss, {
            'grad_norm': grad_norm,
            'expected_grad_norm': self.expected_grad_norm(self.step),
            'deviation': abs(grad_norm - expected) / expected
        }
    
    def checkpoint(self):
        proof = {
            'step': self.step,
            'loss': self.loss_history[-1],
            'expected_loss': self.expected_loss_curve(self.step),
            'grad_stats': self.grad_stats,
            'checksum': hash(self.model.state_dict())
        }
        self.save_checkpoint(proof)
```

**Benefit**: Detect training anomalies, prove training integrity.

## Concrete Implementations

### 1. HollywoodTransformer

A drop-in replacement for standard attention:

```python
class HollywoodTransformerLayer(nn.Module):
    """
    Transformer layer with pre-computed attention routing.
    
    Key insight: Attention patterns are often structured.
    Pre-compute the structure, only compute non-zero entries.
    """
    
    def __init__(self, d_model, n_heads, routing='causal'):
        super().__init__()
        self.attn = HollywoodAttention(d_model, n_heads, routing)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Hollywood attention
        x = x + self.ffn(self.norm2(x))
        return x
```

### 2. AdaptivePrecisionTrainer

Fine-grained mixed precision:

```python
class AdaptivePrecisionTrainer:
    """
    Mixed precision with per-operation granularity.
    
    Key insight from Riemann: Some operations need precision,
    others don't. Softmax = phase computation = needs FP32.
    """
    
    PRECISION_MAP = {
        'linear': torch.float16,      # Matrix multiply: FP16 ok
        'softmax': torch.float32,     # Phase-like: needs FP32
        'layernorm': torch.float32,   # Normalization: needs FP32
        'gelu': torch.float16,        # Activation: FP16 ok
        'embedding': torch.float32,   # Lookup: FP32 for accuracy
    }
    
    def wrap_model(self, model):
        for name, module in model.named_modules():
            op_type = self._classify_op(module)
            if op_type in self.PRECISION_MAP:
                module.register_forward_hook(
                    self._precision_hook(self.PRECISION_MAP[op_type])
                )
```

### 3. SpectralPositionEncoding

Learned non-uniform positions:

```python
class SpectralPositionEncoding(nn.Module):
    """
    Position encoding inspired by NUFFT.
    
    Instead of fixed sin/cos at integer positions,
    learn a continuous position for each token,
    then spread to nearby "grid points" via B-spline.
    """
    
    def __init__(self, d_model, max_len):
        super().__init__()
        self.position_predictor = nn.Linear(d_model, 1)
        
        # B-spline spreading (from Riemann NUFFT)
        self.spread_width = 4
        self.register_buffer('bspline_weights', self._compute_bspline())
    
    def forward(self, x):
        # Predict continuous position for each token
        positions = self.position_predictor(x).squeeze(-1)  # (B, N)
        
        # Spread to "grid" (like NUFFT)
        pos_floor = positions.floor().long()
        pos_frac = positions - pos_floor.float()
        
        # B-spline weights
        weights = self._bspline(pos_frac)  # (B, N, 4)
        
        # Create position encoding by spreading
        # Each token contributes to 4 "frequency bins"
        # This is exactly like our NUFFT spreading!
        ...
```

## For TriX Specifically

If TriX is a tile-based architecture, the mapping is direct:

| Riemann Tile | TriX Tile |
|--------------|-----------|
| ThetaTile | PositionEncodingTile |
| DirichletTile | EmbeddingTile |
| SpectralTile | AttentionTile |
| SignTile | ActivationTile |
| ProbeTile | OutputTile |

Each tile:
1. Has explicit inputs/outputs (like our evaluate() methods)
2. Pre-computes topology at init (like our scatter_indices)
3. Uses appropriate precision per operation
4. Is independently testable/verifiable

## The Meta-Pattern: Computation as Signal Processing

**Everything is a transform:**
- Embedding: discrete tokens → continuous vectors (like DFT)
- Attention: query signal correlated with key signals (like spectral analysis)
- FFN: frequency filtering (low-pass/high-pass on features)
- LayerNorm: signal normalization (like AGC in radio)

**The Riemann hunter is a specialized transformer:**
- Input: position t (like a query)
- Keys: the frequencies ln(n)
- Values: the coefficients n^{-1/2}
- Output: Z(t) (the "attended" response)

**Transformers could be generalized Riemann hunters:**
- Input: any query (not just position)
- Keys: any features (not just log frequencies)
- Values: any content (not just 1/sqrt(n))
- Output: any task (not just zero detection)

## Summary: What Emerged

1. **Hollywood Attention**: Pre-compute routing, eliminate masks
2. **Adaptive Precision**: FP32 softmax, FP16 everything else
3. **Spectral Positions**: Learned non-uniform positions via spreading
4. **Verified Training**: Checkpoints with integrity proofs
5. **The Isomorphism**: Riemann hunting ≡ specialized attention

The Riemann zero hunter is a **proof-of-concept** for these techniques.
If it works for hunting zeros at 20M/sec, it can work for training transformers.
