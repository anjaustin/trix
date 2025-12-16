# MatMul Exploration - Phase 1: Raw

**Stream of consciousness. What am I thinking?**

---

VGem just handed me the key. And it's the same key we already used.

"You have already built the engine. You just need to load a different cartridge."

FFT wasn't special. FFT was an *instance*. The general pattern is:

```
Permutation × Block-Diagonal × Permutation × Block-Diagonal × ...
```

For FFT, the permutations were bit-reversal and stage-wise partner selection. The block-diagonal operations were butterflies with twiddles.

For MatMul, the permutations are... permutations. The block-diagonal operations are small dense blocks (tiles).

It's the same machine. Different program.

---

Monarch Matrices. I've heard of these but never connected them to what we're doing.

The claim: Any dense matrix can be approximated as a product of sparse block-diagonal matrices interleaved with permutations.

```
W ≈ P₁ B₁ P₂ B₂ ... Pₖ Bₖ
```

Where:
- P = permutation (routing, O(N) to apply)
- B = block-diagonal (tiles, O(N) to apply)

Total: O(N × k) instead of O(N²)

If k = O(log N), you get O(N log N). Same as FFT. Not a coincidence.

---

The TriX translation hits hard:

- **Block-Diagonal** = Tiles processing chunks independently
- **Permutation** = Routing shuffling between tiles

We've been building this all along. The Hollywood Squares topology. The signature-based routing. The tile specialization.

FFT was proof of concept. MatMul is the general case.

---

VGem's warning is important: "Generic MatMuls are unstructured and messy."

Random matrices won't decompose cleanly. But structured matrices will:
- Identity (trivial)
- Shift/rotation (permutation only)
- DFT matrix (we just did this!)
- Hadamard (we just did this!)
- Specific learned projections

The question: what structures matter for transformers?

Attention: Q, K, V projections. Output projection.
MLP: Up projection, down projection.

If these can be structured (or constrained to be structured during training), we get efficient inference.

---

The "MatMul-Free Transformer" possibility is wild.

- Sequence mixing: FFT/spectral (done)
- Channel mixing: Structured MatMul/Monarch (next)

No O(N²) attention. No O(N²) dense layers. Everything is O(N log N) routing + local ops.

Is this real? Can it work?

The FFT result suggests yes. Spectral transforms can replace attention (FNet showed this). Structured matrices can replace dense layers (Monarch showed this).

We're just doing it with the TriX machinery: compiled routing + fixed microcode.

---

What I need to understand:

1. How to decompose a matrix into P₁ B₁ P₂ B₂ form
2. What permutation patterns work well
3. How to constrain B to ternary (or small discrete values)
4. What accuracy is achievable for real transformer weights

Let me explore.

---

*End of raw phase.*
