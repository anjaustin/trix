# FP4 Roadmap

**Goal:** TriX efficiently using FP4

**Status:** In Progress

---

## Path 1: Complete the Plumbing

Wire FP4 atoms into the compiler. End-to-end FP4 compilation.

- [x] Integrate FP4AtomLibrary into TriXCompiler (`use_fp4=True`)
- [x] Update Composer to use FP4 threshold circuits
- [x] Update Emitter to pack weights as actual 4-bit values
- [x] Create FP4 weight packing/unpacking utilities (`fp4_pack.py`)
- [x] Build inference path that loads and executes FP4 circuits
- [x] Verify 8-bit adder works end-to-end in FP4
- [x] Measure actual memory savings (1.7x for small atoms, ~7x theoretical)
- [ ] Document integration

**Status:** COMPLETE ✓

### Results

| Circuit | Float32 | FP4 | Compression |
|---------|---------|-----|-------------|
| Full Adder | 100 B | 58 B | 1.7x |
| 8-bit Adder | 100 B | 58 B | 1.7x |

Note: Compression limited by header overhead for small circuits.
For larger circuits with more atom types, compression approaches 7-8x.

---

## Path 2: Compile the FFT

Apply FP4 compilation to the existing FFT implementation.

- [x] Identify FFT atoms (routing: partner, is_upper, twiddle_index)
- [x] Implement FFT routing as FP4 threshold circuits
- [x] Compile N=8 real FFT - **100% exact**
- [x] Scale to N=16, 32 - **100% exact**
- [x] Add twiddle index routing for complex FFT
- [ ] Debug complex FFT twiddle application
- [x] Document results

**Status:** MOSTLY COMPLETE ✓

### Results

| Component | N=8 | N=16 | N=32 |
|-----------|-----|------|------|
| IS_UPPER | 100% (2 layers) | 100% (2 layers) | 100% (2 layers) |
| PARTNER | 100% (2 layers/bit) | 100% (2 layers/bit) | 100% (2 layers/bit) |
| TWIDDLE | 100% (1-2 layers/bit) | - | - |
| Real FFT | **100% exact** | **100% exact** | **100% exact** |
| Complex FFT | Bug in twiddle application | - | - |

### Key Insight

FFT has two types of operations:
1. **Structural routing** (PARTNER, IS_UPPER, TWIDDLE_INDEX) → FP4 threshold circuits
2. **Arithmetic** (a+b, a-b, complex multiply) → Fixed microcode

We compile the ROUTING to FP4. The arithmetic stays exact.

### Files

- `experiments/fft_atoms/fft_compiler.py` - FFT compilation to FP4

---

## Path 3: The Tile Observatory

Research path - discover atoms from trained tiles.

- [ ] Train TriX language model
- [ ] Instrument tile activations
- [ ] Cluster tile behaviors
- [ ] Extract candidate atoms
- [ ] Test if candidates can be made exact
- [ ] If yes, compile to FP4

**Status:** RESEARCH / NOT STARTED

---

## Completed

- [x] TriX Compiler architecture
- [x] Float atoms (trained, 100%)
- [x] FP4 atoms (constructed, 100%)
- [x] Minterm generator
- [x] Validation infrastructure
- [x] FP4 packing utilities (`fp4_pack.py`)
- [x] FP4 Emitter and Loader
- [x] Compiler integration (`use_fp4=True`)
- [x] End-to-end FP4 pipeline tested

---

## Log

| Date | Milestone |
|------|-----------|
| 2025-12-16 | Compiler v0.7.0 complete |
| 2025-12-16 | FP4 atoms verified (10/10, 100%) |
| 2025-12-16 | FP4 plumbing complete |
| 2025-12-16 | Full Adder + 8-bit Adder working in FP4 |
