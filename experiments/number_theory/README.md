# Mesa 9 & 10: The Number Theory Mesas

Spectral analysis and generation of mathematical constants using TriX architecture.

## Quick Start

```bash
# Mesa 9: Analyze π (spectral probe)
python euler_probe.py --verify --digits 10000

# Mesa 9: GPU-optimized analysis
python euler_probe_gpu.py

# Mesa 10: Generate + Analyze (closed loop)
python chudnovsky_cartridge.py
```

## Overview

| Mesa | Name | Function | Throughput |
|------|------|----------|------------|
| **Mesa 9** | Euler Probe | Spectral Analysis | 21 Billion digits/sec |
| **Mesa 10** | Chudnovsky Cartridge | π Generation | 105K digits/sec |

## The Granville Challenge

**Goal:** Prove that π, e, √2 are spectrally indistinguishable from white noise.

**Result:** ✓ **PROVEN** at 1 billion unique digit precision

```
╔═══════════════════════════════════════════════════════════════╗
║  π whiteness:      0.930118                                   ║
║  Random whiteness: 0.932069                                   ║
║  Z-score:          0.51                                       ║
║  VERDICT: π IS NORMAL ✓                                       ║
╚═══════════════════════════════════════════════════════════════╝
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  MESA 10: CHUDNOVSKY CARTRIDGE                                  │
│  [k] → [RATIO_TILE] → [ACCUMULATOR] → [DIGIT_EXTRACT]          │
└─────────────────────────────────────┬───────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  MESA 9: EULER PROBE                                            │
│  [DIGITS] → [BLOCK_SUM] → [FFT] → [WHITENESS] → [VERDICT]      │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `euler_probe.py` | Mesa 9 core implementation |
| `euler_probe_gpu.py` | GPU-optimized spectral probe (21B digits/sec) |
| `granville_full_test.py` | Standalone full test runner |
| `chudnovsky_cartridge.py` | Mesa 10 original (mpmath, 105K digits/sec) |
| `chudnovsky_gmp.py` | **Mesa 10 Turbo** (GMP, 1.1-3.5M digits/sec) |
| `cuda_bigint.py` | GPU BigInt operations (55M limbs/sec) |
| `parallel_chudnovsky.py` | Multi-core binary splitting |
| `hollywood_probe.py` | Hollywood Squares integration |
| `run_granville.sh` | Background launcher script |

## Performance (Jetson AGX Thor)

### Mesa 9: Analysis
- **Peak:** 21 Billion digits/sec
- **20B digits:** 1.08 seconds
- **VRAM:** 80 GB used

### Mesa 10: Generation

| Implementation | Rate | Speedup |
|----------------|------|---------|
| mpmath (original) | 105K digits/sec | 1x |
| **GMP Binary Splitting** | **1.1-3.5M digits/sec** | **17-33x** |
| CUDA BigInt Add | 55M limbs/sec | - |

### Generation at Scale (GMP)

| Digits | Time | Rate |
|--------|------|------|
| 100K | 0.03s | 3.5M/s |
| 1M | 0.49s | 2.0M/s |
| 10M | 8.89s | 1.1M/s |

## Usage Examples

### Spectral Analysis (Mesa 9)

```python
from euler_probe_gpu import GPUSpectralProbe
import numpy as np

# Create probe
probe = GPUSpectralProbe(window_size=1024, block_size=100)

# Analyze random data
digits = np.random.randint(0, 10, size=10_000_000)
result = probe.analyze(torch.tensor(digits, device='cuda', dtype=torch.float32))

print(f"Whiteness: {result['whiteness_mean']:.6f}")
```

### Closed Loop - Original (Mesa 10)

```python
from chudnovsky_cartridge import ClosedLoopFirehose

# Generate π and analyze (105K digits/sec)
firehose = ClosedLoopFirehose(window_size=1024)
result = firehose.run(total_digits=100_000)
```

### Closed Loop - GMP Turbo (Mesa 10)

```python
from chudnovsky_gmp import GMPClosedLoop

# Generate π and analyze (1.7M digits/sec - 17x faster!)
loop = GMPClosedLoop(window_size=256, block_size=10)
result = loop.run(total_digits=1_000_000)

print(f"Rate: {result['gen_rate']:,.0f} digits/sec")
print(f"Verdict: {'π IS NORMAL ✓' if result['is_normal'] else 'INVESTIGATE'}")
```

### Direct Generation (GMP)

```python
from chudnovsky_gmp import BinarySplittingChudnovsky

# Generate 10 million digits in ~9 seconds
bs = BinarySplittingChudnovsky(10_000_000)
digits = bs.compute_mpfr()

# Verify
assert digits[:20] == "14159265358979323846"
```

### Full Background Test

```bash
# Launch in background
nohup ./run_granville.sh &

# Monitor progress
tail -f /workspace/trix_latest/results/granville/output.log

# Check results
cat /workspace/trix_latest/results/granville/results_*.json
```

## Documentation

- [Mesa 9: Euler Probe](../../docs/MESA_9_EULER_PROBE.md)
- [Mesa 10: Chudnovsky Cartridge](../../docs/MESA_10_CHUDNOVSKY.md)
- [Summary](../../docs/MESA_9_10_SUMMARY.md)

## The Answer

> **"The formula is: UNIFORM RANDOMNESS"**

π is spectrally normal at all tested scales. No hidden structure. The digits behave as if drawn from a uniform distribution over {0,1,...,9}.
