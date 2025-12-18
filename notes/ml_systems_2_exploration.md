# Riemann System → ML Training: Exploration

## Testing the Core Ideas

### Idea 1: Attention as Frequency Transform

Standard attention:
```
attn(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
```

Fourier view:
```
Q @ K.T = convolution in position space
        = multiplication in frequency space (by convolution theorem)
```

If Q and K are "signals", then Q @ K.T computes their cross-correlation.
FFT can compute this in O(N log N).

Let me verify this mathematically:

```
(Q @ K.T)[i,j] = Σ_d Q[i,d] * K[j,d]

This is the dot product of row i of Q with row j of K.

For each row i of Q:
  scores[i] = Q[i] @ K.T  (shape: N)
  
This is Q[i] correlated with each row of K.

If we think of K as N signals of length d,
and Q[i] as a filter of length d,
then scores[i] is the correlation of Q[i] with each K[j].
```

The issue: this isn't a convolution in the token dimension.
It's a dot product in the embedding dimension.

Hmm, let me think differently...

### Idea 2: Linearized Attention via Random Features

There's existing work on this: Random Feature Attention (Performer).

```
softmax(Q @ K.T) ≈ φ(Q) @ φ(K).T

where φ is a random feature map.
```

This gives O(N) attention!

But the connection to our NUFFT is different.

### Idea 3: The REAL Connection - Position Encodings

In Riemann: exp(i * t * ln(n))
- t is the "query position" (what t are we evaluating at?)
- ln(n) is the "key frequency" (how does term n oscillate?)

In Transformers with rotary embeddings (RoPE):
```
Q_rot[i] = Q[i] * exp(i * θ * i)  # rotate by position i
K_rot[j] = K[j] * exp(i * θ * j)  # rotate by position j

Q_rot @ K_rot.T has factor exp(i * θ * (i - j))
```

This IS a frequency! The relative position (i - j) determines the phase!

RoPE attention is computing:
```
Σ_j V[j] * exp(i * θ * (i - j)) * base_attention(Q, K)
```

This is EXACTLY like our Riemann sum, but with:
- Uniform positions (i, j are integers)
- The exp factor modulates attention by position

### Idea 4: Non-Uniform Positions

What if positions aren't 0, 1, 2, 3, ... but learned?

```
pos[i] = neural_net(token[i])  # Learned position embedding

Q_rot[i] = Q[i] * exp(i * θ * pos[i])
K_rot[j] = K[j] * exp(i * θ * pos[j])
```

Now the positions are NON-UNIFORM.

To compute attention efficiently, we need... NUFFT!

```
# Spread K to uniform grid
K_uniform = NUFFT_spread(K_rot, pos_K)

# FFT
K_freq = FFT(K_uniform)

# For each query, compute response
for i in range(N):
    # Query at position pos[i]
    # This is like evaluating at a specific frequency
    response[i] = NUFFT_eval_at(K_freq, pos[i])
```

This could give O(N log N) attention with learned positions!

### Idea 5: Hollywood Routing for Sparse Attention

Many attention patterns are sparse:
- Causal: attend only to past (50% sparse)
- Local: attend to window (>90% sparse for long sequences)
- Block-sparse: attend to specific blocks

Currently: mask out unwanted positions (still compute them!)

Hollywood: pre-compute the routing map

```python
# Pre-compute for causal attention
def build_causal_routing(N):
    # For position i, attend to positions 0..i
    attend_from = []
    for i in range(N):
        attend_from.append(list(range(i + 1)))
    return attend_from

# Use routing (like our scatter_indices)
def hollywood_attention(Q, K, V, routing):
    N, d = Q.shape
    output = torch.zeros_like(Q)
    
    for i in range(N):
        indices = routing[i]  # Pre-computed!
        K_local = K[indices]  # Gather
        V_local = V[indices]
        
        # Small local attention
        scores = Q[i] @ K_local.T / math.sqrt(d)
        weights = F.softmax(scores, dim=-1)
        output[i] = weights @ V_local
    
    return output
```

This avoids computing masked positions entirely!

### Idea 6: Adaptive Precision Softmax

Our discovery: phase computation needs FP64.

In attention, softmax is the "phase-like" operation:
- Converts raw scores to probabilities
- Involves exp() which is sensitive to input scale
- Normalization can cause numerical issues

```python
def adaptive_precision_attention(Q, K, V):
    # Bulk matmul in FP16 (fast)
    scores = torch.matmul(Q.half(), K.half().T)  # O(N²) but fast
    
    # Softmax in FP32 (precise)
    scores_fp32 = scores.float() / math.sqrt(Q.shape[-1])
    weights = F.softmax(scores_fp32, dim=-1)
    
    # Output in FP16 (fast)
    output = torch.matmul(weights.half(), V.half())
    
    return output.float()
```

This is finer-grained than standard AMP!

### Testing: Precision Impact on Softmax

Let me think about where precision matters most:

1. **Q @ K.T**: Dot products, robust to noise
2. **Scaling by sqrt(d)**: Simple division, robust
3. **Softmax**: exp() and normalization, SENSITIVE!
4. **@ V**: Weighted sum, moderately robust

The softmax is where precision matters most because:
- exp(x) grows/shrinks exponentially with x
- Small errors in x → big errors in exp(x)
- Normalization divides by sum, propagates errors

Our Riemann insight: exp(i*θ) needs precise θ.
Attention insight: exp(score) needs precise score.

### The Synthesis: Spectral Hollywood Attention

Combining all insights:

```python
class SpectralHollywoodAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, sparsity_pattern='causal'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Standard projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Learned positions (like our grid_idx)
        self.position_net = nn.Linear(d_model, n_heads)
        
        # Pre-computed routing (Hollywood topology)
        self.routing = self._build_routing(max_seq_len, sparsity_pattern)
        
        # NUFFT spreading weights (pre-computed like our B-spline weights)
        self.spreading_weights = self._build_spreading_weights(max_seq_len)
    
    def _build_routing(self, N, pattern):
        """Pre-compute attention routing."""
        if pattern == 'causal':
            return [list(range(i + 1)) for i in range(N)]
        elif pattern == 'local':
            window = 128
            return [list(range(max(0, i - window), i + 1)) for i in range(N)]
        # ... other patterns
    
    def _build_spreading_weights(self, N):
        """Pre-compute NUFFT-style spreading weights."""
        # Like our B-spline weights
        pass
    
    def forward(self, x):
        B, N, D = x.shape
        
        # Project (FP16 for speed)
        Q = self.W_q(x).half()
        K = self.W_k(x).half()
        V = self.W_v(x).half()
        
        # Compute learned positions (like our grid_idx)
        positions = self.position_net(x)  # (B, N, n_heads)
        
        # Apply positional modulation (like exp(i*t*ln(n)))
        # This is where RoPE-style rotation happens
        Q_rot = self._apply_position(Q, positions)
        K_rot = self._apply_position(K, positions)
        
        # Hollywood sparse attention with adaptive precision
        output = self._sparse_attention(Q_rot, K_rot, V)
        
        return self.W_o(output.float())
    
    def _sparse_attention(self, Q, K, V):
        """Sparse attention using pre-computed routing."""
        B, N, D = Q.shape
        output = torch.zeros_like(Q)
        
        for i in range(N):
            indices = self.routing[i]
            
            # Gather (like our NUFFT gather)
            K_local = K[:, indices]
            V_local = V[:, indices]
            
            # Compute attention (FP32 for softmax precision!)
            scores = torch.matmul(Q[:, i:i+1].float(), K_local.float().transpose(-1, -2))
            scores = scores / math.sqrt(self.d_head)
            weights = F.softmax(scores, dim=-1)
            
            # Apply (back to FP16)
            output[:, i] = torch.matmul(weights.half(), V_local).squeeze(1)
        
        return output
```

### What This Enables

1. **O(N * avg_routing_size)** instead of O(N²)
   - For causal: O(N²/2) → still O(N²) but 2x faster
   - For local window 128: O(N * 128) → O(N) effectively

2. **Adaptive precision**: FP16 compute, FP32 softmax
   - 2x memory savings
   - Faster matmuls
   - Stable training

3. **Pre-computed topology**: No masking overhead
   - Routing computed once, reused every forward pass
   - Like our scatter_indices

4. **Learned positions**: Tokens find their natural "frequencies"
   - Non-uniform attention based on semantic position
   - Like our NUFFT with ln(n) frequencies

### The Deeper Pattern

What we discovered in Riemann:
- Computation has STRUCTURE (the sum over n)
- Structure can be PRE-COMPUTED (scatter indices, weights)
- Precision matters for PHASES (exp terms)
- Non-uniform → uniform via SPREADING (B-splines)

What this means for transformers:
- Attention has STRUCTURE (sparsity patterns)
- Structure can be PRE-COMPUTED (routing tables)
- Precision matters for SOFTMAX (the "phase" of attention)
- Non-uniform positions → uniform grid via LEARNED EMBEDDINGS

The Riemann hunter IS a transformer, just for a different task!
