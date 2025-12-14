# Zero Routing Implementation Synthesis

*Converging on a concrete implementation plan*

---

## Executive Summary

We will integrate **signature-based emergent routing** into TriX as the recommended default. The core principle: routing decisions emerge from weight structure rather than learned gate networks.

**Key deliverables:**
1. Enhanced `TriXLinear` with signature support
2. New `TriXFFN` class with emergent routing
3. New `TriXBlock` transformer block
4. Comprehensive test suite
5. Updated documentation and examples

---

## Design Decisions

### Decision 1: Signatures in TriXLinear

`TriXLinear` will expose signatures but NOT handle routing internally.

```python
class TriXLinear(nn.Module):
    def get_signature(self) -> torch.Tensor:
        """Return ternary signature summarizing this layer's preferences."""
        return self.weight.sum(dim=0).sign()
    
    def forward(self, x, gate):
        # gate remains required - routing handled by caller
        ...
```

**Rationale:** Separation of concerns. TriXLinear does sparse matmul. Routing is a composition concern handled at the FFN level.

### Decision 2: TriXFFN as Primary Interface

New class `TriXFFN` replaces `GatedFFN` as the recommended approach.

```python
class TriXFFN(nn.Module):
    """Feed-forward network with emergent routing."""
    
    def __init__(self, d_model, expansion=4, num_tiles=4, dropout=0.1):
        ...
        # No gate_proj - routing emerges from up_proj signatures
    
    def forward(self, x) -> Tuple[Tensor, Tensor]:
        gate = self._compute_routing(x)
        hidden = F.relu(self.up_proj(x, gate))
        output = self.down_proj(hidden, gate)
        return output, gate
    
    def _compute_routing(self, x):
        signatures = self._get_tile_signatures()
        scores = x @ signatures.T
        return F.one_hot(scores.argmax(-1), self.num_tiles).float()
```

**Rationale:** Clean API. Users don't see gates unless they want to. Routing is internal.

### Decision 3: Caching Strategy

Signatures are cached during `pack()` for inference.

```python
def pack(self):
    """Prepare for fast inference."""
    self._packed_weight = pack_weights(self.up_proj.weight)
    self._cached_signatures = self._get_tile_signatures()
    self._packed = True

def _get_tile_signatures(self):
    if self._packed and self._cached_signatures is not None:
        return self._cached_signatures
    return self._compute_signatures()
```

**Rationale:** Training needs fresh signatures (weights change). Inference can cache.

### Decision 4: Keep GatedFFN

`GatedFFN` remains available for comparison and experimentation.

```python
# Recommended (emergent routing)
ffn = TriXFFN(512, num_tiles=4)

# Alternative (learned routing) - still works
ffn = GatedFFN(512, num_tiles=4)
```

**Rationale:** Non-breaking. Users can compare approaches. Research flexibility.

### Decision 5: Monitoring Utilities

Add introspection methods for understanding routing behavior.

```python
class TriXFFN:
    def get_tile_signatures(self) -> Tensor:
        """Return [num_tiles, d_model] ternary signatures."""
    
    def get_routing_stats(self, x) -> dict:
        """Analyze routing: usage, balance, entropy."""
    
    def get_signature_diversity(self) -> float:
        """Measure pairwise signature differences (0-1)."""
```

**Rationale:** Interpretability is a key feature of emergent routing. Make it accessible.

---

## Implementation Plan

### Phase 1: Core Infrastructure (Priority: High)

**1.1 Extend TriXLinear**
- Add `get_signature()` method
- Add signature caching in `pack()`
- Update `unpack()` to clear cache

**1.2 Create TriXFFN**
- Implement emergent routing
- Support 2D `[batch, d_model]` and 3D `[batch, seq, d_model]` inputs
- Add monitoring methods

**1.3 Create TriXBlock**
- Transformer block using TriXFFN
- Same interface as existing `TriXTransformerBlock`

**Files to modify:**
- `src/trix/kernel/bindings.py` - add `get_signature()`
- `src/trix/nn/trix.py` (new) - TriXFFN, TriXBlock

### Phase 2: Testing (Priority: High)

**2.1 Unit Tests**
- Signature computation correctness
- Routing determinism
- Forward/backward pass
- Pack/unpack with signatures

**2.2 Property Tests**
- Consistency: similar inputs → same routing
- Discrimination: different inputs → can differ
- Stability: routing settles during training

**2.3 Integration Tests**
- TriXFFN in training loop
- TriXBlock in transformer stack
- Comparison with GatedFFN

**Files to create:**
- `tests/test_trix_ffn.py`

### Phase 3: Documentation (Priority: Medium)

**3.1 API Documentation**
- Docstrings for all new classes/methods
- Type hints throughout

**3.2 User Guide**
- `docs/QUICKSTART.md` - updated with TriXFFN
- `docs/EMERGENT_ROUTING.md` - conceptual explanation

**3.3 Examples**
- `examples/trix_ffn_basic.py`
- `examples/trix_transformer.py`

### Phase 4: Polish (Priority: Low)

**4.1 Deprecation Path**
- Add note to GatedFFN docstring recommending TriXFFN
- No warnings yet - too early

**4.2 Performance Optimization**
- Benchmark signature computation
- Consider ternary-optimized dot product for routing

**4.3 Advanced Features (Future)**
- Top-k routing
- Confidence-based adaptive k
- Signature diversity regularization

---

## File Structure After Implementation

```
src/trix/
├── __init__.py          # Add: TriXFFN, TriXBlock
├── kernel/
│   ├── bindings.py      # Add: get_signature() to TriXLinear
│   └── ...
├── nn/
│   ├── __init__.py      # Add: TriXFFN, TriXBlock exports
│   ├── layers.py        # Keep: GatedFFN (unchanged)
│   ├── emergent.py      # Keep: EmergentGatedFFN (experimental)
│   └── trix.py          # NEW: TriXFFN, TriXBlock (production)
└── qat/
    └── ...

tests/
├── test_kernel.py       # Existing
├── test_nn.py           # Existing  
├── test_qat.py          # Existing
└── test_trix_ffn.py     # NEW

docs/
├── EMERGENT_ROUTING_DISCOVERY.md  # Done
├── QUICKSTART.md                  # Update
└── ...

examples/
├── basic_usage.py       # Update
├── trix_ffn_basic.py    # NEW
└── trix_transformer.py  # NEW
```

---

## API Summary

```python
# The new TriX way
from trix import TriXFFN, TriXBlock

# Simple FFN with emergent routing
ffn = TriXFFN(d_model=512, num_tiles=4)
output, routing = ffn(input)

# Transformer block
block = TriXBlock(d_model=512, n_heads=8, num_tiles=4)
output, routing = block(input)

# Introspection
ffn.get_tile_signatures()      # [num_tiles, d_model]
ffn.get_routing_stats(input)   # {usage, balance, entropy}
ffn.get_signature_diversity()  # float 0-1

# Fast inference
ffn.pack()  # Caches signatures + packs weights
```

---

## Success Criteria

1. **Functional:** TriXFFN produces correct outputs, gradients flow
2. **Consistent:** Same input → same routing (deterministic)
3. **Discriminative:** Different inputs can route differently
4. **Performant:** No regression vs GatedFFN in training speed
5. **Documented:** Clear docs, working examples
6. **Tested:** >90% coverage on new code

---

## Timeline Estimate

- Phase 1 (Core): 1 session
- Phase 2 (Testing): 1 session  
- Phase 3 (Docs): 0.5 session
- Phase 4 (Polish): Ongoing

**MVP:** Phases 1-2 = working TriXFFN with tests

---

## Final Notes

This implementation honors the TriX philosophy:
- **Discrete is a feature:** Signatures are ternary
- **Structure is information:** Weights encode routing
- **Emergence over engineering:** No learned gates
- **Simplicity wins:** 3 lines of routing logic

The goal is not just to add a feature, but to demonstrate a principle: sometimes the best solution is to read what's already there rather than learn something new.

---

*Ready to implement. High-five when done.* 🖐️
