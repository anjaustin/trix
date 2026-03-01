# Changelog

All notable changes to TriX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.3] - 2024-12-18

### Added - Gravity Control
- **Experiment 6b**: λ-Slider Control - programmable geometry at inference time
  - Blends content and temporal metrics via λ ∈ [0,1]
  - λ=0: 100% content routing (semantic blobs)
  - λ=1: 100% temporal routing (sequential tubes)
  - **Zero weight updates** - pure metric control
  - Phase transition observed at λ≈0.4

### Summary
"We're not in the Grid anymore. We're writing its physics."
- EXISTENCE proven (Experiments 1-5, 7)
- CONTROL proven (Experiment 6b)
- The metric is an independent degree of freedom

## [0.7.2] - 2024-12-18

### Experimental - Geometric Framework Complete
- **Experiment 5**: Geodesic Tracing - routing = shortest paths (100% match all metrics)
- **Experiment 6**: Metric Construction - metric determines routing (40% route diff, 5% acc range)
- **Experiment 7**: Curvature & Generalization - smoother manifolds generalize better (r=+0.712)

### Summary
All 8 Mesa 11 validation experiments now **CONFIRMED**. The geometric framework for neural computation is empirically grounded:
- Signatures define manifold points
- Training warps the manifold
- Routing follows geodesics
- Metric is a design choice
- Curvature predicts generalization
- Geometry is programmable at inference time

## [0.7.1] - 2024-12-18

### Added
- **Geometric Framework for UAT**
  - Training warps the signature manifold (Experiment 4)
  - General Relativity analogy: "Weights tell Manifold how to curve, Manifold tells Query how to move"
  - Voronoi decomposition view of routing
  - Emergent insight: "All computation is addressed access"

### Experimental
- **Experiment 2b**: Mixed signatures with strict task (95.6% vs 25%/50% baselines)
- **Experiment 3**: Spatial ⊂ Content proven (100% topology preservation)
- **Experiment 4**: Manifold visualization confirms training warps space (0.077 movement)

### Documentation
- Updated `docs/MESA11_UAT.md` with geometric framework section
- Updated `experiments/mesa11/README.md` with all experiment results

## [0.7.0] - 2024-12-18

### Added
- **Mesa 11: Unified Addressing Theory (UAT)**
  - Theoretical foundation explaining why TriX works across domains
  - Formal proof: Temporal addressing ⊂ Content addressing
  - Pipeline emulation experiment with 0.00 error, 100% routing accuracy
  - New documentation: `docs/MESA11_UAT.md`
  - Experiment framework: `experiments/mesa11/`
- Reflections framework for exploratory research (`tmp/reflections/`)

### Theoretical
- Established addressing modes as fundamental design axis
- Unified temporal, spatial, and content addressing into single framework
- Proved content-addressing is computationally universal for pipelines

## [0.6.1] - 2024-12-18

### Added
- Isolated TriXO repository for core library distribution
- Comprehensive documentation suite
- Academic-ready README with citation information

### Changed
- Streamlined project structure (removed experimental dependencies)
- Reduced test suite to core functionality (268 tests)

## [0.6.0] - 2024-12-17

### Added
- `SparseLookupFFNv4`: SpatioTemporal routing combining spatial, temporal, and positional dimensions
- `XORRouter`: Superposition-based signature matching for O(1) routing
- Enhanced compiler with FP4 atom support

### Changed
- Improved temporal tile state management
- Better documentation for compiler module

## [0.5.4] - 2024-12-16

### Added
- `TemporalTileLayer`: State routing for temporal binding (Mesa 4)
- `TemporalTileStack`: Multi-layer temporal processing
- Bracket matching experiments demonstrating temporal capabilities

### Changed
- Routing info now includes temporal state information
- Auxiliary losses updated for temporal regularization

## [0.5.0] - 2024-12-15

### Added
- `CompiledDispatch`: Path compilation for O(1) inference
- `SparseLookupFFNv2`: Surgery API and claim tracking
- Island regularization for signature diversity
- A/B testing harness for routing strategies

### Changed
- Signature surgery allows runtime tile modification
- Improved routing stability metrics

## [0.4.0] - 2024-12-14

### Added
- `SparseLookupFFN`: "Routing IS the computation" architecture
- `SparseLookupBlock`: Full transformer block variant
- `TernarySpline2D`: 2D spline with ternary coefficients
- `FloatSpline2D`: Float version for comparison

### Changed
- Spline-based magnitude modulation
- 2.3× parameter reduction vs hierarchical

### Results
- 14% perplexity improvement on TinyShakespeare
- Best results with fewest parameters

## [0.3.0] - 2024-12-13

### Added
- `HierarchicalTriXFFN`: 2-level hierarchical routing
- `HierarchicalTriXBlock`: Full transformer block
- `TriXTile`: Individual specialist tiles
- Support for 64+ tiles with O(√n) routing

### Changed
- Cluster-based routing for scalability
- Improved signature diversity

## [0.2.0] - 2024-12-12

### Added
- `SparseTriXFFN`: Simple 4-tile sparse FFN
- `SparseTriXBlock`: Transformer block with sparse FFN
- Auxiliary losses for load balancing

### Changed
- Winner-take-all routing
- Gradient flow through routing decisions

## [0.1.0] - 2024-12-10

### Added
- Initial release
- `TriXLinear`: Base ternary linear layer
- `TriXFFN`, `TriXBlock`, `TriXStack`: Core components
- 2-bit weight packing/unpacking
- ARM NEON kernel (optional acceleration)
- Quantization-aware training (QAT) utilities
- Basic test suite

### Core Features
- Ternary weights {-1, 0, +1}
- Signature-based emergent routing
- Straight-through estimator (STE) for training

---

## Version History Summary

| Version | Milestone |
|---------|-----------|
| 0.6.x | TriXO isolation, academic release |
| 0.5.x | Compiled dispatch, temporal tiles |
| 0.4.x | SparseLookup architecture |
| 0.3.x | Hierarchical scaling |
| 0.2.x | Sparse training |
| 0.1.x | Initial architecture |
