# Pure TriX FFT: Neural Control Planes for Algorithmic Execution

**Draft Abstract**

---

## Abstract

We present a complete Fast Fourier Transform implementation using the TriX architecture, demonstrating that neural routing can learn algorithmic structure while maintaining exact arithmetic execution. Unlike approaches that train neural networks to approximate FFT through gradient descent on input-output examples, our method decomposes the algorithm into atomic components—addressing, butterfly operations, and twiddle factor selection—and learns the *control flow* while preserving *exact computation*.

The architecture follows a consistent pattern across all components: fixed microcode (exact arithmetic operations) combined with learned or algorithmic routing (control logic). For the butterfly operation, discrete coefficient pairs (1,1) and (1,-1) serve as opcodes; routing learns which operation to apply. For twiddle factors, the N roots of unity are fixed microcode; routing learns the structural mapping from (stage, position) to twiddle index.

Key results on the complete FFT register:

- **Structural learning (ADDRESS):** 100% accuracy learning partner selection via i XOR 2^stage
- **Discrete operations (BUTTERFLY):** 100% accuracy with emergent tile specialization (91-95% purity)
- **Complex rotation (TWIDDLE):** 100% accuracy on structural routing in 10 training epochs
- **N-scaling:** 100% accuracy on N=8, 16, 32, 64 with identical architecture
- **Round-trip closure:** IFFT(FFT(x)) = x with max error ~10^-6

The critical insight emerged from failure: initial attempts to learn butterfly arithmetic directly achieved 0% accuracy, revealing that neural networks cannot reliably extrapolate arithmetic beyond training ranges. The solution—learning *when* to use each operation rather than learning the operations themselves—yielded 100% accuracy with perfect generalization.

This work suggests a broader principle: neural architectures can serve as *control planes* for mathematical execution, learning algorithmic structure while delegating exact computation to fixed primitives. The resulting system is not an approximation of FFT but an exact implementation with learned control flow—more analogous to a CPU executing microcode than a neural network fitting a function.

The complete implementation requires ~25K parameters for the routing network, trains in under 100 epochs, and executes with O(N log N) complexity. No attention mechanisms, no learned arithmetic, no hybrid external organs—pure TriX throughout.

---

## Keywords

sparse routing, algorithmic learning, FFT, discrete operations, neural control plane, exact computation, TriX architecture

---

## Notes for Revision

- Need to formalize "fixed microcode + learned routing" pattern
- Should compare parameter count / training cost to transformer-based FFT learning
- Could strengthen the "control plane" framing with references to DSP/CPU literature
- The 6502 parallel (opcodes as microcode, learned sequencing) deserves expansion
- Failure analysis (0% on learned coefficients) is a key contribution - emphasize

---

*Draft v1 - 2024-12-16*
