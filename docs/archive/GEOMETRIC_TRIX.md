# Geometric TriX: SparseLookupFFN v3/v4

> **⚠️ EXPERIMENTAL** - These modules explore positional and spatiotemporal routing. The stable API is `SparseLookupFFNv2`.

## Overview

Geometric TriX extends the SparseLookupFFN with **positional awareness** and **state tracking**, moving from content-only routing to geometric routing through a manifold.

## The Geometric Insight

### Content-Only Routing (v2)

```
Input → Content Signature → Nearest Tile → Output
```

- Routes based on WHAT the input contains
- No awareness of WHERE in sequence
- Routing pattern: "sparkle" (random per token)

### Positional Routing (v3)

```
Input → Content × Position → Tile with Continuity → Output
```

- Routes based on WHAT and WHERE
- B-spline spreading enforces C² continuity
- Routing pattern: "wave" (smooth transitions)

### SpatioTemporal Routing (v4)

```
Input → Content × Position × State → Tile → Output + State Update
```

- Routes based on WHAT, WHERE, and HISTORY
- State carries forward through sequence
- Enables temporal binding (bracket matching, counting)

## SparseLookupFFNv3

### Key Features

1. **Positional Routing**: Content scores modulated by position
2. **Gauge Transforms**: Rotate vectors between tile coordinate systems
3. **Vortex Expert**: Handles "turbulent" residuals
4. **Topology Tracker**: Detects persistent routing cycles (Hodge)

### Usage

```python
from trix.nn import SparseLookupFFNv3

ffn = SparseLookupFFNv3(
    d_model=128,
    num_tiles=16,
    enable_positional=True,
    enable_gauge=True,
    enable_vortex=True,
)

output, info, aux = ffn(x)  # x: [batch, seq, d_model]
```

### Diagonality Score

Measures routing pattern geometry:
- 0.0 = "sparkle" (random, no structure)
- 1.0 = "wave" (perfect diagonal flow)

```python
# Measured results
content_only_diagonality = 0.000  # No structure
positional_diagonality = 0.968   # Strong wave pattern
```

## SparseLookupFFNv4

### Key Features

1. **Full 3-Way Routing**: Content × Position × State
2. **State Encoder**: Learns state representation
3. **State Modulation**: State affects routing scores
4. **Tile-State Tracking**: Which tiles see which states

### Usage

```python
from trix.nn import SparseLookupFFNv4

ffn = SparseLookupFFNv4(
    d_model=128,
    num_tiles=16,
    d_state=32,
    state_influence=0.3,
)

# Initialize state
state = ffn.init_state(batch_size=4)

# Forward with state
output, info, aux, new_state = ffn(x, state=state)
```

### 29D Hypercube

The routing space forms a hypercube:
- Content: 10 dimensions (1024 content clusters)
- Position: 10 dimensions (1024 position bins)
- State: 9 dimensions (512 state configurations)

Total: 29 dimensions = 537M possible routes

## Millennium Problem Connections

### Navier-Stokes (Reynolds Decomposition)

```
Flow = Smooth Mean + Turbulent Fluctuation
Routing = Base Tile + Vortex Expert
```

The vortex expert handles residuals that don't fit smooth routing.

### Yang-Mills (Gauge Transforms)

```
Parallel Transport = Rotate vector when moving between tiles
Gauge Transform = Learn tile-to-tile rotation matrices
```

Vectors maintain meaning when transitioning between tiles.

### Hodge Conjecture (Topology)

```
Persistent Cycles = Routing patterns that repeat
Topology Tracker = Detect and optimize cycles
```

Identifies which routing paths can be merged.

## Files

| File | Purpose |
|------|---------|
| `src/trix/nn/sparse_lookup_v3.py` | Positional + Gauge + Vortex |
| `src/trix/nn/sparse_lookup_v4.py` | Full SpatioTemporal |
| `src/trix/nn/xor_routing.py` | XOR-based signature matching |
| `experiments/positional_routing/` | Validation experiments |

## Benchmark Results

### TinyShakespeare Character-LM

| Model | Accuracy | Tile Purity | Diagonality |
|-------|----------|-------------|-------------|
| v2 (Content-only) | 58.2% | 45% | 0.00 |
| v3 (Geometric) | 57.6% | 66% | 0.125 |

Notes:
- Similar accuracy (character-level doesn't need geometry)
- Higher tile purity (more consistent routing)
- Weak diagonality (character tasks don't benefit from position)

### 6502 Operations

| Model | ADC Accuracy | Overall |
|-------|--------------|---------|
| v2 (Content-only) | 4.8% | 87.6% |
| v3 + 3 layers | 30.6% | 88.2% |
| v4 (SpatioTemporal) | 12% | 85% |

Notes:
- Geometric helps ADC (needs state awareness)
- SpatioTemporal over-fragments (too many dimensions)
- Best solution: Atomic Composition (100%)

## When to Use

### Use v2 (Stable)
- Production deployments
- Tasks without strong positional structure
- When simplicity matters

### Use v3 (Experimental)
- Research on geometric routing
- Tasks with spatial structure
- When tile purity matters more than accuracy

### Use v4 (Experimental)
- Research on state-dependent routing
- Temporal tasks (counting, tracking)
- When state history affects computation

## The Lesson

> "Geometry regularizes. But not all tasks need it."

Positional routing adds structure. Structure can help (smoother loss landscape) or hurt (over-constraining).

**Match the geometry to the problem.**
