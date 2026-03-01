# SPEC: Riemann Probe - 100% TriX Implementation

**Status:** SPECIFICATION  
**Version:** 1.0  
**Date:** 2025-12-17

---

## Overview

The Riemann Probe verifies the Riemann Hypothesis by computing zeros of the Riemann zeta function on the critical line Re(s) = 0.5.

**This spec defines the 100% TriX implementation.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  RIEMANN PROBE - PURE TRIX                                          │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  THETA   │ → │ DIRICHLET│ → │ SPECTRAL │ → │   SIGN   │ → ZEROS │
│  │  TILE    │   │   TILE   │   │   TILE   │   │   TILE   │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│       │              │              │              │                │
│       ▼              ▼              ▼              ▼                │
│  TemporalTile   TemporalTile   TriX FFT     TemporalTile           │
│  (θ function)   (n^{-it})      (0.00 error) (sign change)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. ThetaTile

**Purpose:** Compute Riemann-Siegel theta function θ(t)

**Formula:**
```
θ(t) = (t/2)·log(t/2π) - t/2 - π/8 + O(1/t)
```

**TriX Implementation:**
- Uses `TemporalTileLayer` for sequential computation
- Microcode: LOG, MUL, SUB, DIV
- Input: t values (batch)
- Output: θ(t) values

**Source:** `trix.nn.TemporalTileLayer`

---

### 2. DirichletTile

**Purpose:** Generate Dirichlet series coefficients n^{-1/2-it}

**Formula:**
```
a_n(t) = n^{-1/2} · e^{-it·log(n)}
       = n^{-1/2} · (cos(t·log(n)) - i·sin(t·log(n)))
```

**TriX Implementation:**
- Uses `TemporalTileLayer` for coefficient generation
- Microcode: LOG, COS, SIN, MUL
- Input: t value, N (number of terms)
- Output: Complex coefficients [N x 2] (real, imag)

**Source:** `trix.nn.TemporalTileLayer`

---

### 3. SpectralTile

**Purpose:** Evaluate Z(t) using FFT

**Formula (Riemann-Siegel):**
```
Z(t) = 2·Re(e^{iθ(t)} · Σ_{n≤N} n^{-1/2-it})
```

**TriX Implementation:**
- Uses `pure_trix_fft` from `experiments/fft_atoms/`
- 100% accuracy (verified)
- Scales N=8 to N=64 per stage, compose for larger

**Key Files:**
- `experiments/fft_atoms/pure_trix_fft_discrete.py` - Real FFT
- `experiments/fft_atoms/pure_trix_fft_twiddle_v2.py` - Complex FFT
- `experiments/fft_atoms/pure_trix_fft_nscale_v2.py` - N-scaling

**Source:** `experiments/fft_atoms/`

---

### 4. SignChangeTile

**Purpose:** Detect sign changes in Z(t) to locate zeros

**Method:**
```
If Z(t_i) · Z(t_{i+1}) < 0, zero exists in [t_i, t_{i+1}]
```

**TriX Implementation:**
- Uses `TemporalTileLayer` for parallel sign detection
- Microcode: SIGN, MUL, CMP
- Input: Z values (batch)
- Output: Zero locations (refined via bisection)

**Source:** `trix.nn.TemporalTileLayer`

---

## Data Flow

```
Input: t_start, t_end, resolution

1. THETA TILE
   t_values[M] → θ(t_values)[M]

2. DIRICHLET TILE  
   For each t in t_values:
     Generate a_n(t) for n = 1..N
   Output: coefficients[M x N x 2]

3. SPECTRAL TILE (TriX FFT)
   Apply FFT to sum Dirichlet series
   Output: Z_values[M]

4. SIGN TILE
   Detect sign changes in Z_values
   Output: zero_candidates[K]

Output: List of zeros on critical line
```

---

## TriX Primitives Used

| Component | TriX Primitive | Location |
|-----------|----------------|----------|
| ThetaTile | `TemporalTileLayer` | `trix.nn.temporal_tiles` |
| DirichletTile | `TemporalTileLayer` | `trix.nn.temporal_tiles` |
| SpectralTile | `pure_trix_fft` | `experiments/fft_atoms/` |
| SignChangeTile | `TemporalTileLayer` | `trix.nn.temporal_tiles` |

---

## What Exists vs What Needs Building

### EXISTS (verified working):
- [x] `TemporalTileLayer` - temporal computation
- [x] `pure_trix_fft` - 100% accurate FFT (N=8 to N=64)
- [x] FFT composition - scales to arbitrary N
- [x] FFT/IFFT round-trip verified

### NEEDS BUILDING:
- [ ] `ThetaTile` - wrap theta computation in TemporalTileLayer
- [ ] `DirichletTile` - wrap coefficient generation
- [ ] `SpectralTile` - integrate pure_trix_fft for Z(t)
- [ ] `SignChangeTile` - wrap sign detection
- [ ] `RiemannProbeTriX` - pipeline orchestrator
- [ ] Large-N FFT composition (N > 64)

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| FFT Accuracy | 0.00 error | Verified in fft_atoms |
| Z(t) Accuracy | 1e-10 | Sufficient for zero detection |
| Throughput | 10^6 zeros/sec | With GPU parallelism |
| VRAM | < 32 GB | For 10^9 zeros |

---

## File Structure (To Be Created)

```
experiments/number_theory/
├── trix_riemann/
│   ├── __init__.py
│   ├── theta_tile.py      # ThetaTile implementation
│   ├── dirichlet_tile.py  # DirichletTile implementation
│   ├── spectral_tile.py   # SpectralTile (wraps pure_trix_fft)
│   ├── sign_tile.py       # SignChangeTile implementation
│   ├── probe.py           # RiemannProbeTriX orchestrator
│   └── benchmark.py       # Performance verification
└── trix_riemann_test.py   # Integration tests
```

---

## Dependencies

**Internal (TriX):**
```python
from trix.nn import TemporalTileLayer, TemporalTileStack

# FFT atoms (to be imported)
from experiments.fft_atoms.pure_trix_fft_discrete import TriXFFTReal
from experiments.fft_atoms.pure_trix_fft_twiddle_v2 import TriXFFTComplex
```

**External:**
```python
import torch  # Tensor backend only, NOT for FFT
import math   # Constants only
```

**NOT ALLOWED:**
```python
# These defeat the purpose of TriX:
torch.fft.*        # Use TriX FFT instead
numpy.fft.*        # Use TriX FFT instead
scipy.fft.*        # Use TriX FFT instead
mpmath.*           # For verification only, not production
```

---

## Verification

Each component must pass:

1. **Unit Test:** Output matches mpmath to 1e-10
2. **Integration Test:** Finds all 10 known zeros
3. **Scale Test:** 10^6 zeros with 0 anomalies
4. **Performance Test:** Meets throughput target

---

## Implementation Order

1. **Phase 1:** ThetaTile + unit tests
2. **Phase 2:** DirichletTile + unit tests  
3. **Phase 3:** SpectralTile (integrate FFT atoms) + unit tests
4. **Phase 4:** SignChangeTile + unit tests
5. **Phase 5:** RiemannProbeTriX pipeline + integration tests
6. **Phase 6:** Billion zero test + performance verification

---

## Success Criteria

The implementation is complete when:

1. `python trix_riemann_test.py` passes all tests
2. All 10 known zeros verified with TriX
3. 10^9 zeros scanned with 0 anomalies
4. No `torch.fft`, `numpy.fft`, or external FFT calls
5. Throughput ≥ 10^5 zeros/sec (single GPU)

---

## References

- `experiments/fft_atoms/README.md` - TriX FFT documentation
- `experiments/fft_atoms/pure_trix_fft_discrete.py` - Reference implementation
- `src/trix/nn/temporal_tiles.py` - TemporalTileLayer source
- Edwards, H.M. "Riemann's Zeta Function" - Mathematical background
- Odlyzko, A. "The 10^20-th zero of the Riemann zeta function" - Algorithm

---

**This is the spec. Implementation follows this exactly.**
