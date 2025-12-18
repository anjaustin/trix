# Hollywood Squares FFT - Layered Parallelism

*Extension to the Triton FFT spec: Hollywood Squares OS integration*

---

## The Insight

The Triton FFT kernel handles **within-FFT parallelism** (butterflies).

Hollywood Squares OS handles **across-FFT parallelism** (orchestration).

Together, they form a complete system:

```
Hollywood Squares = Topology + Message Passing + Supervision
Triton FFT        = Compiled TriX on GPU silicon

Hollywood Squares FFT = Orchestrated GPU compute with verified composition
```

---

## The Three Layers

### Layer 1: Triton FFT Kernel (Done)

The atomic compute unit. One kernel = one FFT (or one batch of FFTs).

```python
@triton.jit
def trix_fft_kernel(...):
    # All stages fused
    # Twiddles in registers/shared memory
    # Butterflies parallelized across threads
```

**What it handles:**
- Stage-level parallelism (unrolled loop)
- Butterfly-level parallelism (threads)
- Twiddle lookup (precomputed table)

**What it doesn't handle:**
- Multi-kernel orchestration
- Precision cascading
- Fault recovery
- Multi-GPU distribution

### Layer 2: Fabric Kernel (The Orchestrator)

The Hollywood Squares master that manages FFT tiles.

```python
class FFTFabricKernel:
    """
    Master node for Hollywood Squares FFT.
    
    Responsibilities:
    - Twiddle table management (broadcast once, use everywhere)
    - Batch dispatch (round-robin or load-balanced)
    - Result collection and anomaly detection
    - Fault supervision (restart failed tiles)
    """
    
    def __init__(self, num_tiles: int, devices: List[str]):
        self.tiles = [FFTTile(i, dev) for i, dev in enumerate(devices)]
        self.twiddle_tables = {}  # Shared across all tiles
        self.anomaly_queue = Queue()
    
    def broadcast_twiddles(self, N: int):
        """Compute twiddles once, broadcast to all tiles."""
        W_re, W_im = compute_twiddle_table(N, 'cuda')
        for tile in self.tiles:
            tile.set_twiddles(W_re, W_im)
    
    def dispatch_batch(self, x_re: Tensor, x_im: Tensor) -> Tensor:
        """
        Distribute work across tiles.
        
        Hollywood Squares topology: each tile processes a slice.
        """
        batch_size = x_re.shape[0]
        chunk_size = batch_size // len(self.tiles)
        
        futures = []
        for i, tile in enumerate(self.tiles):
            start = i * chunk_size
            end = start + chunk_size if i < len(self.tiles) - 1 else batch_size
            futures.append(tile.process_async(x_re[start:end], x_im[start:end]))
        
        # Collect results
        results = [f.result() for f in futures]
        return torch.cat(results, dim=0)
```

**What it handles:**
- Work distribution
- Twiddle sharing (compute once, use many)
- Result aggregation
- Fault detection

### Layer 3: Precision Cascade (Screening/Verification)

The Hollywood Squares pattern from Riemann: fast screening, selective refinement.

```python
class PrecisionCascadeFFT:
    """
    Two-pass FFT with precision escalation.
    
    Pass 1: FP16 screening (10x faster, some error)
    Pass 2: FP32/FP64 refinement (only for flagged regions)
    
    For Riemann Probe:
    - FP16 detects sign changes (potential zeros)
    - FP32 confirms exact zero location
    - FP64 verifies anomalies (off critical line?)
    """
    
    def __init__(self):
        self.fp16_fabric = FFTFabricKernel(dtype=torch.float16)
        self.fp32_fabric = FFTFabricKernel(dtype=torch.float32)
        self.fp64_verifier = HighPrecisionVerifier()
    
    def process(self, t_values: Tensor) -> List[Zero]:
        # Pass 1: FP16 screening (fast)
        Z_fp16 = self.fp16_fabric.evaluate_Z(t_values)
        candidates = detect_sign_changes(Z_fp16)
        
        # Pass 2: FP32 refinement (accurate)
        refined = []
        for c in candidates:
            Z_fp32 = self.fp32_fabric.evaluate_Z(c.t_neighborhood)
            zero = bisect_to_zero(Z_fp32)
            if zero.suspicious:
                # Pass 3: FP64 verification (paranoid)
                self.fp64_verifier.verify(zero)
            refined.append(zero)
        
        return refined
```

**What it handles:**
- 10x speedup from FP16 screening
- Precision only where needed
- Anomaly escalation

---

## The Complete Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RIEMANN PROBE HOLLYWOOD                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   PRECISION CASCADE                          │   │
│  │                                                               │   │
│  │   t_values ──► [FP16 Screen] ──► [FP32 Refine] ──► zeros    │   │
│  │                     │                  │                      │   │
│  │                     └──► anomalies ────┴──► [FP64 Verify]    │   │
│  │                                                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   FABRIC KERNEL                              │   │
│  │                                                               │   │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│  │   │ Twiddle │  │  Batch  │  │ Result  │  │  Fault  │        │   │
│  │   │ Bcast   │  │ Dispatch│  │ Collect │  │ Recover │        │   │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘        │   │
│  │                                                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│           ┌──────────────┼──────────────┐                          │
│           ▼              ▼              ▼                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │
│  │  FFT Tile 0 │ │  FFT Tile 1 │ │  FFT Tile N │                   │
│  │  ┌───────┐  │ │  ┌───────┐  │ │  ┌───────┐  │                   │
│  │  │Triton │  │ │  │Triton │  │ │  │Triton │  │                   │
│  │  │Kernel │  │ │  │Kernel │  │ │  │Kernel │  │                   │
│  │  └───────┘  │ │  └───────┘  │ │  └───────┘  │                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
│       GPU 0           GPU 1           GPU N                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why This Matters

### 1. Separation of Concerns

- **Triton kernel**: Optimal GPU utilization for single FFT
- **Fabric kernel**: Optimal work distribution across resources
- **Precision cascade**: Optimal compute allocation by importance

Each layer does one thing well.

### 2. Inherited Correctness

From Hollywood Squares theorem:
> Deterministic message passing + bounded local semantics + enforced observability ⇒ global convergence with inherited correctness

- Triton FFT is verified (matches torch.fft to 0.00)
- Fabric kernel is deterministic (same input → same dispatch)
- Message passing is typed (input/output contracts)

Therefore: the composed system is correct.

### 3. Scaling Properties

| Resource | How it scales |
|----------|---------------|
| More SMs | More butterflies in parallel (within-kernel) |
| More GPUs | More tiles in fabric (across-kernel) |
| More precision | Cascade adds layers (vertical) |
| More t-range | Batch dispatch handles it (horizontal) |

The architecture scales in all dimensions.

---

## Implementation Plan

### Phase 1: Single-GPU Fabric (Current)
- TritonFFT kernel ✅
- Basic batch dispatch (implicit in kernel)
- Single precision

### Phase 2: Precision Cascade
- FP16 screening kernel
- FP32 refinement kernel
- Anomaly queue between them

### Phase 3: Multi-GPU Fabric
- FFTFabricKernel class
- Twiddle broadcast
- Cross-device batch dispatch
- Result collection

### Phase 4: Full Hollywood Squares
- Supervision trees
- Fault recovery
- Trace replay
- Observable execution

---

## The Punchline

> "Hollywood Squares is not about parallelism. It's about **topology**."

The FFT stages are not "parallelized" — they're **wired**. The precision levels are not "selected" — they're **routed**. The GPUs are not "distributed" — they're **composed**.

Topology is algorithm. The wiring determines the behavior.

When we add Hollywood Squares to Triton FFT, we're not adding an optimization. We're revealing the natural architecture that was always there.

---

*Ready for implementation when approved.*
