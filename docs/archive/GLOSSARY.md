# TriX Glossary

**Precise definitions for all terminology used in the TriX system.**

---

## A

### Atom
The smallest unit of computation in TriX. A tiny neural network (typically 1-2 layers) that computes a specific boolean function exactly. Examples: AND, OR, XOR, SUM, CARRY.

### Atomic Decomposition
The principle that complex computations can be broken into primitive atoms. "Routing learns WHEN, atoms compute WHAT."

---

## B

### Bit-Reversal Permutation
A permutation where index i maps to the bit-reversal of i. Used in FFT to reorder outputs. Example: for N=8, index 3 (011) maps to index 6 (110).

### Block-Diagonal Matrix
A matrix with non-zero entries only in square blocks along the diagonal. Butterfly networks use 2×2 block-diagonal structure.

### Block Opcode
A named 2×2 matrix used as a building block in butterfly networks. Examples: I (identity), H+ (Hadamard), SWAP.

### Butterfly Network
A computational structure with log₂(N) stages, where each stage pairs elements and applies 2×2 operations. Used for FFT, WHT, and structured MatMul.

### Butterfly Operation
A 2×2 linear transform applied to a pair of elements:
```
[y_i]   [a b] [x_i]
[y_j] = [c d] [x_j]
```

---

## C

### Carry Atom
FP4 atom that computes the carry bit of a full adder: outputs 1 if at least 2 of 3 inputs are 1.

### Circuit
A composition of atoms with defined wiring. Inputs flow through atoms according to the topology.

### Circuit Specification (CircuitSpec)
A declarative description of a circuit's structure: inputs, outputs, atom instances, and connections.

### Compiled Routing
Routing logic implemented as FP4 neural circuits rather than conditional code. Enables hardware deployment.

### Composition
Connecting outputs of one atom to inputs of another. The fundamental way to build complex circuits from simple atoms.

### Cooley-Tukey Algorithm
The divide-and-conquer algorithm for FFT. Splits N-point DFT into smaller DFTs, combines with twiddle factors.

---

## D

### DFT (Discrete Fourier Transform)
Transform that converts a signal from time domain to frequency domain. Complexity O(N²) naive, O(N log N) via FFT.

### DIT (Decimation In Time)
FFT variant where input is bit-reversed and output is in natural order. TriX uses DIT for DFT compilation.

---

## E

### E2M1 Format
4-bit floating point format with 1 sign bit, 2 exponent bits, 1 mantissa bit. Range: ±6 with gaps.

### Emit
The compiler phase that outputs the final representation (JSON topology + FP4 weights).

### Exhaustive Verification
Testing all possible inputs to prove correctness. Feasible for small circuits (e.g., 2^16 = 65,536 for 8-bit adder).

---

## F

### FFT (Fast Fourier Transform)
O(N log N) algorithm for computing DFT. Uses butterfly structure with twiddle factors.

### FP4
4-bit representation for neural network weights. TriX uses custom lookup tables for atoms.

### FP4 Atom
An atom whose weights are stored in 4-bit format. Constructed (not trained) to compute exactly.

### Full Adder
Circuit that adds three bits (a, b, carry_in) producing sum and carry_out. Composed from SUM + CARRY atoms.

---

## G

### Gate
Synonym for atom in digital logic context. AND gate, OR gate, etc.

---

## H

### Hadamard Matrix
Orthogonal matrix with entries ±1. For N=2: [[1,1],[1,-1]]. Self-inverse up to scaling.

### Hadamard-like Block
A 2×2 ternary matrix that is orthogonal. There are exactly 12 such matrices.

### Hollywood Squares Topology
Grid arrangement of tiles where each tile handles specific functionality. Named for the TV show layout.

---

## I

### Isomorphic Transformer
Transformer architecture where attention → spectral mixing (WHT/FFT) and FFN → butterfly MLP. No O(N²) operations.

### is_upper Circuit
FP4 circuit that determines if a position is in the "upper" half of a butterfly pair (for sign selection).

---

## L

### Layer (Butterfly)
One stage of a butterfly network. Pairs elements with stride 2^stage and applies block operations.

### Lookup Table (LUT)
Table mapping indices to values. FP4 atoms use LUTs to store 16 possible 4-bit weights.

---

## M

### Manifest
JSON file listing all outputs from compilation with checksums for integrity verification.

### Mesa
Emergent global coherence from local interactions. When tiles coordinate without central control.

### Microcode
Fixed operation code within atoms. Atoms execute microcode; routing selects which atoms to invoke.

### Monarch Matrix
Generalized butterfly structure: (B₁ ⊗ I_q) × P × (I_p ⊗ B₂). Enables O(N√N) matrix operations.

### MUX Atom
Multiplexer atom: outputs first input if selector is 0, second input if selector is 1.

---

## N

### NF4 Format
Normal Float 4-bit format with non-uniform quantization levels. Used in QLoRA.

### Neural CPU
Vision of neural networks as compiled machines rather than trained approximators.

---

## O

### Opcode
Operation code specifying which computation to perform. Twiddle opcodes select complex multipliers; block opcodes select 2×2 matrices.

---

## P

### Packing
Converting float weights to 4-bit representation for storage efficiency.

### Partner Circuit
FP4 circuit that computes the XOR partner position in a butterfly stage.

### Permutation Matrix
Matrix with exactly one 1 in each row and column. Reorders elements without scaling.

---

## R

### Ripple Carry Adder
N-bit adder composed of N full adders in sequence. Carry "ripples" from LSB to MSB.

### Routing
Logic that determines data flow through a circuit. In TriX, routing is compiled to FP4 circuits.

---

## S

### Spectral Mixer
Component that mixes information across sequence positions using WHT/FFT. Replaces attention in Isomorphic Transformer.

### Stage
One level of a butterfly network. N-point transform has log₂(N) stages.

### Stride
Distance between paired elements in a butterfly stage. Stage s has stride 2^s.

### SUM Atom
FP4 atom that computes the sum bit of a full adder: XOR of 3 inputs (parity).

---

## T

### Ternary
Values in {-1, 0, +1}. Ternary weights enable efficient hardware implementation.

### Threshold Circuit
Neural network that computes boolean functions using threshold activation. TriX atoms are threshold circuits.

### Tile
A specialized unit in Hollywood Squares topology. Each tile handles one aspect of computation.

### Topology
The graph structure of a circuit: which atoms connect to which.

### TriX
The overall system for compiled neural computation. Name suggests "tricks" of the trade.

### TriX Compiler
System that takes circuit specifications and produces verified, optimized neural circuits.

### Twiddle Factor
Complex exponential W_N^k = e^{-2πik/N} used in FFT. Rotates complex numbers in the frequency domain.

### Twiddle Opcode
Named twiddle factor with algebraic value. For N=8: W0=1, W1=(√2-i√2)/2, W2=-i, etc.

---

## U

### Unpacking
Converting 4-bit packed representation back to float weights for execution.

---

## V

### Verification
Process of proving circuit correctness. Exhaustive verification tests all inputs.

---

## W

### WHT (Walsh-Hadamard Transform)
Transform using Hadamard matrix. All entries ±1, no complex numbers. Related to XOR operations.

### Wiring
Connections between atoms in a circuit. Specifies which outputs feed which inputs.

---

## X

### XOR Pairing
Pattern where position i pairs with position i XOR stride. Fundamental to butterfly structure.

---

## Numerical Reference

| Symbol | Meaning |
|--------|---------|
| N | Transform size (must be power of 2) |
| log₂(N) | Number of butterfly stages |
| O(N log N) | FFT/WHT complexity |
| O(N²) | Naive DFT/attention complexity |
| 2^16 | Number of 8-bit adder test cases (65,536) |
| 81 | Number of ternary 2×2 matrices (3^4) |
| 12 | Number of orthogonal ternary 2×2 matrices |

---

*Terms defined. Ambiguity eliminated.*
