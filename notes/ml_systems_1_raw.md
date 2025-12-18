# Riemann System → ML Training: Raw Exploration

## What We Built

Let me inventory the pieces:

1. **Hollywood Topology** - Pre-computed routing (indices, weights)
2. **Adaptive Precision** - FP64 for critical paths, FP32 for bulk compute
3. **NUFFT** - Non-uniform → uniform frequency transform via B-spline spreading
4. **Streaming Evaluation** - Process infinite sequence, checkpoint, resume
5. **Sparse-to-Dense** - Scatter/gather patterns with pre-computed maps

## Stream of Consciousness

What IS a transformer doing?

Attention: Q @ K.T @ V
- Q, K, V are projections of input
- Q @ K.T is "which tokens attend to which"
- Softmax makes it a probability distribution
- Multiply by V to get weighted values

The Q @ K.T step is O(N²) where N = sequence length.
This is the bottleneck for long sequences.

What if attention IS a frequency transform?

Think about it:
- Q encodes "what am I looking for"
- K encodes "what do I contain"
- The dot product measures similarity

In Fourier terms:
- Q is like a probe frequency
- K is like a signal
- Dot product extracts that frequency component

Wait... this is EXACTLY what we do in Riemann-Siegel!

S(t) = Σ_n a_n * exp(i * t * ln(n))

- t is the probe (query)
- ln(n) is the frequency (key)
- a_n is the coefficient (value)
- The sum extracts the response at frequency t

## The Connection

Attention: Σ_j softmax(Q_i · K_j) * V_j
Riemann:  Σ_n weight(t, n) * a_n

Both are weighted sums where the weight depends on a "similarity" function!

In attention: similarity = dot product + softmax
In Riemann: similarity = exp(i * t * ln(n)) (phase alignment)

## What If We Replace Attention with FFT?

Standard attention is O(N²) - every query attends to every key.

FFT-based attention:
1. Project Q, K, V to frequency domain
2. Multiply in frequency domain (convolution becomes multiplication)
3. Project back

This is O(N log N)!

But wait... the queries and keys have NON-UNIFORM positions (like our ln(n) frequencies).

We need... NUFFT for attention!

## NUFFT Attention

The insight: Token positions in attention are like non-uniform sample points.

Standard attention assumes uniform positions (token 0, 1, 2, ...).
But semantic "distance" between tokens is NOT uniform.
- "The cat sat" - "cat" and "sat" are semantically close
- "The very very very big cat" - "the" and "cat" are far apart syntactically but related

What if we encode semantic position as a non-uniform frequency?

Token i at semantic position p_i (learned or computed)
Query Q_i probes at position p_i
Keys K_j respond at positions p_j

The attention weight is then: exp(i * p_i * p_j) or similar

This IS our NUFFT spreading pattern!

## Hollywood Squares for Attention

The attention pattern is often SPARSE:
- Causal masking: only attend to past
- Local attention: only attend to nearby
- Sparse patterns: attend to specific positions

These patterns are FIXED for a given architecture.

Hollywood says: pre-compute the routing!

Instead of computing the full N×N attention matrix:
1. Pre-compute which (i,j) pairs have non-zero attention
2. Store as sparse topology
3. Only compute those entries

This is like our pre-computed scatter_indices!

## Adaptive Precision for Training

Our discovery: FP64 for phases, FP32 for bulk compute.

In training:
- Gradients need precision (small updates accumulate)
- Forward pass can be lower precision
- Loss computation needs precision

Mixed precision training already does this!
But we learned something more specific:

The CRITICAL path is phase/angle computation.
In attention, the critical path is the softmax normalization.

What if:
- Compute Q @ K.T in FP16/BF16 (bulk matrix multiply)
- Compute softmax in FP32 (precision critical)
- Compute output in FP16/BF16

This is finer-grained than standard mixed precision.

## The Checkpoint Pattern

Our hunt checkpoints at milestones with proofs.

Training does this too: save model every N steps.

But our system has something extra: VERIFICATION.

We compute expected zero count (Riemann-von Mangoldt) and compare.

For training:
- Expected loss curve (theoretical or from similar runs)
- Gradient norm bounds
- Activation statistics

Checkpoint + verify = TRUSTWORTHY training.

## What About TriX Specifically?

TriX is "tile-based transformers" (I'm inferring from context).

The Riemann system is deeply tile-based:
- ThetaTile: Compute theta function
- DirichletTile: Compute coefficients
- SpectralTile: FFT
- SignTile: Sign change detection
- ProbeTile: Full evaluation

Each tile is:
- Self-contained
- Composable
- Independently verifiable

This maps to TriX architecture:
- Each tile = one attention head or FFT block
- Tiles compose via routing (Hollywood)
- Each tile has explicit inputs/outputs

## Emergence: Spectral Attention Tile

What if we create a new TriX tile that does "spectral attention"?

```
SpectralAttentionTile:
  Input: Q (queries), K (keys), V (values)
  
  1. Embed Q, K into frequency space
     q_freq = NUFFT_spread(Q, learned_positions)
     k_freq = NUFFT_spread(K, learned_positions)
  
  2. Multiply in frequency domain
     attn_freq = q_freq * conj(k_freq)  # O(N) not O(N²)!
  
  3. Transform back
     attn_weights = NUFFT_gather(attn_freq)
  
  4. Apply to values
     output = attn_weights @ V
  
  Complexity: O(N log N) vs O(N²)
```

## Emergence: Hollywood Attention Routing

What if attention patterns are pre-compiled like our scatter_indices?

```
HollywoodAttentionTile:
  Setup (once per architecture):
    1. Analyze attention patterns from training
    2. Find sparse structure (which i,j pairs matter)
    3. Compile routing table: attend_from[i] = [j1, j2, ...]
    4. Pre-compute relative position encodings
  
  Forward (every batch):
    1. Load routing table (zero cost - just topology)
    2. Gather K values: K_local[i] = K[attend_from[i]]
    3. Compute local attention (small dense matmul)
    4. Scatter results back
  
  This is like our NUFFT spreading but for attention!
```

## Emergence: Adaptive Precision Attention

What if different parts of attention use different precision?

```
AdaptivePrecisionAttention:
  # Bulk compute in FP16
  QK = torch.matmul(Q.half(), K.T.half())  # O(N²) but fast
  
  # Precision-critical softmax in FP32
  QK_fp32 = QK.float()
  attn_weights = F.softmax(QK_fp32, dim=-1)
  
  # Output in FP16
  output = torch.matmul(attn_weights.half(), V.half())
```

This is exactly what we learned from Riemann:
- Phase computation (softmax) needs precision
- Bulk multiply can be low precision

## The Meta-Insight

The Riemann system taught us:

1. **Routing is topology, not computation** - Pre-compute once, reuse forever
2. **Precision is path-dependent** - Some paths need FP64, others don't
3. **Sparse structure is everywhere** - Don't compute zeros
4. **Frequency domain is powerful** - Convolutions become multiplications
5. **Non-uniform grids are natural** - NUFFT bridges the gap

These ALL apply to transformers!

## What Emerges

A new architecture: **Spectral Transformer**

- Replace dense attention with spectral (NUFFT-based) attention
- Pre-compute attention routing (Hollywood topology)
- Adaptive precision per operation
- O(N log N) complexity instead of O(N²)
- Checkpoint + verify training integrity

This could enable:
- 10x longer context (N log N vs N²)
- 2-4x training speed (adaptive precision)
- Provably correct training (verification at checkpoints)
