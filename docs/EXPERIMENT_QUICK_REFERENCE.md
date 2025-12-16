# Semantic Geometry Experiments - Quick Reference

## Running the Experiments

```bash
cd /workspace/trix_latest

# 1. Original thesis demo (single run)
python experiments/geometry_thesis.py

# 2. Convergence test (10 seeds, ~2 min)
python experiments/convergence_test.py

# 3. Fuzzy boundary tests (~1 min)
python experiments/fuzzy_boundary_test.py

# 4. Full test suite (verify nothing broke)
python -m pytest tests/ -v
```

## Expected Results Summary

| Experiment | Key Metric | Expected Value |
|------------|------------|----------------|
| geometry_thesis.py | Test accuracy | ~99% |
| geometry_thesis.py | Routing purity | >95% |
| convergence_test.py | Containment (10 seeds) | 78% ± 8% |
| convergence_test.py | Purity (10 seeds) | 97% ± 2% |
| fuzzy_boundary_test.py | Shared dims purity | ~98% (no drop) |
| fuzzy_boundary_test.py | 40% mix purity | ~84% (graceful) |

## File Map

```
experiments/
├── geometry_thesis.py       # Minimal proof of concept
├── convergence_test.py      # Multi-seed validation  
└── fuzzy_boundary_test.py   # Robustness tests

notes/
├── emergence_session_01_raw.md         # Raw exploration
├── emergence_session_02_reflection.md  # Node analysis
└── emergence_session_03_convergence.md # Synthesis

docs/
├── SEMANTIC_GEOMETRY_THESIS.md  # Full documentation
└── EXPERIMENT_QUICK_REFERENCE.md # This file
```

## Key Code Snippets

### Minimal TriX Layer
```python
class MinimalTriXLayer(nn.Module):
    def forward(self, x):
        sigs = self.signatures          # [n_tiles, d_model]
        scores = x @ sigs.T             # [batch, n_tiles]
        tile_indices = scores.argmax(-1) # Hard routing
        # Each tile has own output
        logits = self.tile_outputs[tile_idx](x)
        return logits, tile_indices
```

### Ternary Quantization (STE)
```python
def _quantize_ternary(self, x):
    with torch.no_grad():
        q = torch.zeros_like(x)
        q[x > 0.3] = 1.0
        q[x < -0.3] = -1.0
    return x + (q - x).detach()  # Straight-through estimator
```

### Routing Loss (Key Insight)
```python
# Supervise the OBJECTIVE, not the MECHANISM
routing_loss = F.nll_loss(
    F.log_softmax(x @ signatures.T, dim=-1),
    target_class  # Class i should route to tile i
)
```

## Pass/Fail Criteria

### Convergence Test
- ✓ PASS: Containment >70%, std <15%
- ✓ PASS: Purity >90%, std <10%
- ✓ PASS: Recovery >60%

### Fuzzy Boundary Test
- ✓ PASS: Shared dims purity >90%
- ✓ PASS: 20% mix purity >70%
- ✓ PASS: 40% mix purity >50%
- ✗ FAIL: Any purity <50% (collapse)

## Completed Experiments

### Signature Surgery ✓
```bash
python experiments/signature_surgery.py
```
**Result:** Hand-designed signature claimed 100% of target class.
System enhanced design with discriminative negatives.

## Next Experiments (Not Yet Implemented)

### Natural Data Pilot
- MNIST: Test if tiles specialize by visual feature
- Char-text: Test if tiles specialize by character class

### Multi-Layer Composition
- Stack 2-3 TriX layers
- Analyze routing patterns across depth
- Test coarse→fine hierarchy

---

*Last updated: December 15, 2024*
