# FFT Atoms - Pure TriX FFT

A complete FFT implementation using TriX architecture.  
**100% accuracy. Forward + Inverse. Scales N=8 to N=64.**

---

## Quick Start (30 seconds)

```bash
# From repo root:
cd experiments/fft_atoms
python run_all_fft.py
```

That's it. If it prints `ALL TESTS PASSED`, you're done.

---

## What You'll See

```
============================================================
TRIX FFT - COMPLETE TEST SUITE
============================================================

[1/5] Real FFT (discrete ops)...
  ✓ 100/100 passed

[2/5] Complex FFT (twiddle factors)...
  ✓ 100/100 passed

[3/5] N-Scaling (8→64)...
  ✓ N=8:  100/100
  ✓ N=16: 100/100
  ✓ N=32: 100/100
  ✓ N=64: 100/100

[4/5] FFT/IFFT Round-trip...
  ✓ IFFT(FFT(x)) == x (max error: 1.2e-06)

[5/5] Full integration...
  ✓ Complete FFT subsystem operational

============================================================
ALL TESTS PASSED
============================================================
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy

That's the full list. No CUDA required (but it's faster with it).

### Install Dependencies

```bash
pip install torch numpy
```

Or from repo root:

```bash
pip install -e .
```

---

## What's In Here

| File | What It Does |
|------|--------------|
| `run_all_fft.py` | **Start here.** Runs everything. |
| `pure_trix_fft_discrete.py` | Real FFT with discrete operations |
| `pure_trix_fft_twiddle_v2.py` | Complex FFT with twiddle factors |
| `pure_trix_fft_nscale_v2.py` | Tests scaling from N=8 to N=64 |
| `pure_trix_fft_ifft.py` | Forward/Inverse round-trip test |

---

## How It Works (One Paragraph)

FFT is decomposed into atoms: ADDRESS (which elements pair), BUTTERFLY (add/subtract), TWIDDLE (complex rotation). Each atom uses **fixed microcode** (exact arithmetic) and **learned routing** (which operation when). The routing can be learned by a neural network OR computed algorithmically - both give 100% accuracy. The key insight: don't learn arithmetic, learn WHEN to use each operation.

---

## Run Individual Tests

```bash
# Real FFT
python pure_trix_fft_discrete.py

# Complex FFT with twiddles
python pure_trix_fft_twiddle_v2.py

# N-scaling test
python pure_trix_fft_nscale_v2.py

# Round-trip test
python pure_trix_fft_ifft.py
```

---

## Troubleshooting

### "No module named torch"
```bash
pip install torch
```

### "No module named numpy"
```bash
pip install numpy
```

### "No module named trix"
```bash
# From repo root:
pip install -e .
```

### Tests pass but I want to understand how it works
Read `pure_trix_fft_discrete.py` first. It's the simplest complete example.  
Then read `pure_trix_fft_twiddle_v2.py` for complex numbers.

### I want the deep dive
See `docs/FFT_ATOMS_HYBRID.md` in the repo root.

---

## Results Summary

| Component | Accuracy |
|-----------|----------|
| ADDRESS (structural) | 100% |
| BUTTERFLY (discrete ops) | 100% |
| TWIDDLE (complex rotation) | 100% |
| N=8 FFT | 100% |
| N=16 FFT | 100% |
| N=32 FFT | 100% |
| N=64 FFT | 100% |
| IFFT round-trip | 100% (error < 1e-5) |

---

## License

MIT. Do whatever you want with it.

---

## Questions?

Open an issue or read the journal: `notes/journal_2024_12_16_fft_session.md`
