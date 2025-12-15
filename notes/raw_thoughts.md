# Raw Thoughts on TriX

## First Impressions

This is a research project exploring 2-bit ternary neural networks with a focus on sparse computation. The core idea is elegant: use ternary weights ({-1, 0, +1}) not just for compression, but as a mechanism for emergent routing. The tagline "Don't learn what you can read" captures this nicely.

But digging deeper, this is actually part of a larger vision called **PQH (Pseudo-Quasi-Holographic) Architecture** - a hierarchical system where routing replaces dense computation entirely.

## What TriX Actually Does

The project implements a drop-in replacement for transformer FFN layers. Instead of dense matrix multiplications, it:

1. Packs weights into 2 bits (4 weights per byte) - achieving 16x memory compression vs FP32
2. Uses weight signatures (sum of weights per tile, then sign) as "addresses"
3. Routes inputs to tiles based on content similarity (input @ signatures.T)
4. Only computes the winning tile - sparse execution

This is essentially content-addressable memory where the addresses emerge from the weight structure itself. No learned router parameters.

## Architecture Overview

The codebase has three main abstraction levels:

- **HierarchicalTriXFFN** - The recommended approach. Two-level hierarchical routing (clusters -> tiles). O(sqrt(n)) routing complexity.
- **SparseTriXFFN** - Simpler 4-tile version. Proven to work.
- **TriXFFN** - Classic emergent routing reference implementation.

Supporting infrastructure:
- `kernel/` - C++ with ARM NEON optimization for packed 2-bit inference
- `qat/` - Quantization-aware training utilities
- `nn/` - All the PyTorch modules including some interesting spline-based layers

## What Surprised Me

1. **The documentation depth** - 28 markdown files in `docs/` covering everything from raw exploration notes to formal abstracts. This feels like genuine research documentation, not just API docs.

2. **The spline layers are central, not peripheral** - There's a whole subsystem (`spline.py`, `spline2d.py`, `spline_adc.py`, `additive_kan.py`) implementing KAN-style (Kolmogorov-Arnold Network) components. These aren't a side experiment - they're the **deployment target**. The project includes "Spline-6502": a 3KB neural processor that achieves 100% accuracy on 6502 CPU operations (ADC, shifts, logic, etc.). The spline code can export binary lookup tables for actual 6502 hardware. This connects to FLYNNCONCEIVABLE, an external project that implements a neural 6502 CPU.

3. **The test coverage** - 146 tests across 8 files. Tests cover numerical accuracy, gradient flow, routing stability, pack/unpack roundtrips. This is more thorough than many research repos.

4. **Jetson-first development** - The project explicitly targets Jetson AGX Thor. ARM NEON optimization is first-class, not an afterthought.

## Technical Observations

### The Routing Mechanism

```python
signature = weights.sum(dim=0).sign()
scores = input @ signatures.T
winner = scores.argmax()
```

This is clever. The signature is essentially "what features does this tile care about?" and routing becomes "which tile's preferences best match this input?" Zero learned parameters for routing.

### Memory Layout

Weights packed as 2 bits: `00` = 0, `01` = +1, `10` = -1. Four weights per byte. The C++ kernel handles the bit manipulation and NEON vectorization.

### Auxiliary Losses

The FFN returns `(output, routing_info, aux_losses)`. The aux_losses include load balancing terms to prevent routing collapse (all inputs going to one tile).

## Concerns / Questions

1. **Validation claims** - README shows TinyShakespeare results but the "Reproduce" section is empty. No exact commands, configs, or seeds provided.

2. **Hardware portability** - Only tested on Jetson. The README warns other platforms are "untested." CPU fallback exists in the kernel but unclear if it's been validated.

3. **The spline/TriX integration gap** - The spline infrastructure is well-developed and the TriX routing is well-developed, but integrating them (replacing TriX tile MLPs with spline evaluation) is listed as "pending" in the docs. This is the key missing piece for the full PQH vision.

4. **Training stability** - Ternary quantization during training can be tricky. The QAT module exists but I haven't dug into how well it handles gradient estimation through the sign() function.

5. **Scale** - Results are on TinyShakespeare char-LM. How does this scale to larger models/datasets?

## Opportunities I See

1. **Benchmarking** - Add actual benchmark scripts with timing comparisons (dense vs sparse, packed vs unpacked)

2. **Reproducibility** - Fill in the README's "Reproduce" section with exact commands

3. **Documentation cleanup** - The docs/ folder has raw exploration notes mixed with formal docs. Could benefit from organization.

4. **Cross-platform CI** - GitHub Actions to test on x86 (scalar fallback) and ARM

5. **Complete the Spline-TriX integration** - The roadmap in PQH_EXECUTIVE_SUMMARY.md is clear: replace tile MLPs with spline evaluation. This would validate the full vision.

6. **Quantization path** - The docs ask: how do we get from float32 spline coefficients to 2-bit? Each cell currently uses 32-bit floats but the target is 6 bits per cell (3 coefficients x 2 bits).

## The Bigger Picture (PQH Vision)

Reading the docs more carefully, the full vision is:

```
Traditional NN:     Input -> Dense Matrix Multiply -> Output
                    O(n^2) computation, all weights touched

PQH Architecture:   Input -> Route -> Load -> Compute -> Output
                    O(1) routing, O(1) computation, sparse weights
```

The hierarchy is:
- **Level 0**: Full model (pseudo-holographic: any input addressable)
- **Level 1**: Clusters (sqrt(n) of them)
- **Level 2**: Tiles (each a 3KB Spline-6502)
- **Level 3**: Spline cells (3 coefficients each)

The quote that captures it: *"The neural network doesn't compute the answer. It routes to where the answer lives."*

This is ambitious. If it works at scale, it's a fundamentally different approach to neural computation - closer to content-addressable memory than traditional matrix multiplication.

## Bottom Line

This is more ambitious than a typical research repo. It's not just "2-bit quantization" - it's a complete architectural philosophy where:

1. **Routing replaces computation** - The intelligence is in knowing where to look, not in dense matrix ops
2. **Splines replace MLPs** - Piecewise linear functions that can run on 1975-class hardware
3. **Hierarchical structure** - O(sqrt(n)) routing instead of O(n) dense computation

The implementation is clean, well-tested (146 passing tests), and clearly the result of iterative exploration. The validation on TinyShakespeare shows 13.4% perplexity improvement with the hierarchical approach.

**What's proven:**
- TriX hierarchical routing works (validated)
- Spline-6502 achieves 100% accuracy on CPU operations (exhaustively tested)
- The components exist and work independently

**What's still pending:**
- Integration of splines into TriX tiles
- Reproducibility commands in README
- Scaling beyond TinyShakespeare
- Cross-platform validation

Worth following closely. The core thesis - "routing IS the intelligence" - is a genuinely different way of thinking about neural networks.

---

## Benchmark Results (Dec 14, 2024)

Ran head-to-head comparison on TinyShakespeare char-level LM:
- 4 layers, 128 d_model, 16 tiles (4 clusters of 4)
- 20k training samples, 10 epochs
- Jetson AGX Thor (CUDA)

| Model | Params | Val PPL | Routing H | Active Tiles |
|-------|--------|---------|-----------|--------------|
| HierarchicalTriXFFN | 826,304 | **11.79** | 0.937 | 64/64 |
| HybridKANFFN | 882,112 | 11.80 | 0.850 | 51/64 |

**Result: Essentially a tie** (0.1% difference).

### Observations

1. **HierarchicalTriXFFN** achieves better tile utilization (all 64 tiles active) and higher routing entropy (more balanced distribution).

2. **HybridKANFFN** uses 7% more parameters but achieves same performance. Its spline-based nonlinearity doesn't hurt, but doesn't help either at this scale.

3. Both models show healthy routing - no collapse to single tiles.

4. HybridKANFFN's routing entropy (0.850) is lower, suggesting some tiles are underutilized. This might become a problem at larger scale.

### Interpretation

Nova's Path 1 (HybridKAN as evolutionary step) is **viable but not clearly better**. The spline-ish inductive bias neither helps nor hurts at this scale.

Next questions:
- Does the difference grow with model size?
- Does HybridKAN handle quantization better (spline coeffs might quantize more cleanly than MLP weights)?
- Can we test Nova's Path 3 idea: bottleneck -> 2 discrete scalars -> Spline2D?

### Code

Benchmark script: `scripts/benchmark_ffn.py`
Results: `results/benchmark_results.json`
