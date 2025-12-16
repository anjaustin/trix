# Glossary

**Terms used in TriX, defined precisely.**

---

## Core Concepts

### Atom
The smallest unit of computation in TriX. An atom implements a single boolean or arithmetic function (e.g., AND, XOR, SUM). Atoms are verified to compute exactly - 100% accuracy on all inputs.

*Example:* The SUM atom computes the parity of three inputs: `SUM(a, b, c) = a XOR b XOR c`

### Tile
A specialized compute unit that implements one or more atoms. In the TriX architecture, tiles are arranged in a grid and can be routed to form larger circuits.

### Routing
The pattern of connections between tiles. Routing determines which tile outputs feed into which tile inputs. In compiled transforms, routing is structural - determined by the algorithm, not learned.

### Microcode
Fixed operations that don't change based on input. In TriX, arithmetic operations (add, subtract, multiply by constant) are microcode. They execute the same way every time.

*Contrast with:* Routing, which varies based on position/stage.

---

## FP4 Concepts

### FP4 (4-bit Floating Point)
A compact representation using only 4 bits per weight. In TriX, we use a custom encoding that exactly represents the values we need.

### Threshold Circuit
A neural network layer where each neuron computes a weighted sum and applies a threshold (step function). With carefully chosen weights, threshold circuits can compute boolean functions exactly.

*Formula:* `y = 1 if (Σ w_i x_i + b) > 0 else 0`

### Construction (vs Training)
Building a circuit by directly setting weights to known-correct values, rather than learning weights through optimization. Constructed circuits are exact by design.

*Example:* An AND gate can be constructed as `y = step(x1 + x2 - 1.5)` - no training needed.

---

## Transform Concepts

### Walsh-Hadamard Transform (WHT)
A linear transform using only addition and subtraction. The WHT of a vector x is computed by repeatedly pairing elements and computing sums/differences.

*Key property:* Self-inverse. `WHT(WHT(x)) = N * x`

*Structure:* Partner selection via XOR. `partner(stage, pos) = pos XOR 2^stage`

### Discrete Fourier Transform (DFT)
A transform that converts signals from time domain to frequency domain. Requires complex multiplication by "twiddle factors" - roots of unity.

*Key property:* Decomposes signal into frequency components.

### Fast Fourier Transform (FFT)
An efficient algorithm for computing DFT in O(N log N) time instead of O(N²). The Cooley-Tukey algorithm is the most common.

### Twiddle Factor
A complex number used in FFT computation. For N-point DFT, twiddle factors are powers of `W_N = e^(-2πi/N)`.

*Example for N=8:* `W_8^0 = 1`, `W_8^1 = (1-i)/√2`, `W_8^2 = -i`, etc.

### Twiddle Opcode
A fixed operation that multiplies by a specific twiddle factor. Instead of computing twiddles at runtime (via cos/sin), we predefine opcodes for each needed twiddle.

*Key insight:* Twiddles are not computed. They are selected.

### Butterfly
The basic operation in FFT/WHT. Takes two inputs (a, b) and produces two outputs (a+b, a-b) or (a + W*b, a - W*b) for complex FFT.

---

## Verification Concepts

### Exhaustive Verification
Testing a function on ALL possible inputs. For boolean functions with few inputs, this proves correctness absolutely.

*Example:* A 2-input AND gate has 4 possible inputs. Testing all 4 proves it correct.

### Truth Table
A complete listing of a function's output for every possible input. Used to define and verify boolean functions.

### Exactness
A computation is exact if it produces the mathematically correct result with no approximation error. TriX atoms are exact by construction.

*Contrast with:* Approximate computation, which accepts some error for efficiency.

---

## Architecture Concepts

### Compilation
Transforming a high-level specification into a fixed circuit that executes without runtime decisions. In TriX, compilation means all routing is resolved, all weights are set, nothing is computed at runtime.

### Structural Routing
Routing that is determined by the algorithm structure, not by input values. For FFT, the partner selection pattern is structural - it depends only on (stage, position), not on data.

### Hollywood Squares
The TriX topology where tiles are arranged in a grid, like the game show. Each tile can receive from neighbors and route to the next layer.

---

## Mathematical Notation

### ⊕ (XOR)
Exclusive OR. `a ⊕ b = 1` iff exactly one of a, b is 1.

### W_N
Primitive N-th root of unity. `W_N = e^(-2πi/N)`

### W_N^k
k-th power of W_N. `W_N^k = e^(-2πik/N) = cos(2πk/N) - i·sin(2πk/N)`

### H_N
Hadamard matrix of size N×N. The WHT of vector x is `H_N @ x`.

### σ (sigma)
Activation function. In threshold circuits, `σ(x) = 1 if x > 0 else 0` (step function).

---

## Common Abbreviations

| Abbrev | Meaning |
|--------|---------|
| DFT | Discrete Fourier Transform |
| FFT | Fast Fourier Transform |
| WHT | Walsh-Hadamard Transform |
| DIT | Decimation In Time (FFT variant) |
| FP4 | 4-bit Floating Point |
| ROM | Read-Only Memory |

---

## Related Terms

### Cooley-Tukey Algorithm
The most common FFT algorithm. Uses divide-and-conquer with bit-reversal and twiddle factors.

### Bit Reversal
Reordering array elements by reversing the binary representation of their indices. Required for Cooley-Tukey DIT FFT.

*Example:* For N=8, index 1 (001) becomes 4 (100).

### Radix-2
An FFT that divides the problem in half at each stage. Requires N to be a power of 2.

---

*When in doubt, check this glossary.*
