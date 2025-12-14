# Zero Routing Implementation - Raw Thoughts

*Stream of consciousness on how to integrate emergent routing into TriX*

---

## Where does routing live?

Currently we have:
- `TriXLinear` - the core sparse layer, takes `gate` as input
- `GatedFFN` - has a `gate_proj` network that produces gates
- `EmergentGatedFFN` - our experimental version with signature-based routing

The question: should routing be in the layer itself or composed externally?

Arguments for routing IN TriXLinear:
- Self-contained - layer handles its own routing
- Simpler API - no need to pass gates
- Natural place for signatures (they come from the weights)

Arguments for routing OUTSIDE:
- Separation of concerns
- Flexibility - can swap routing strategies
- TriXLinear stays focused on sparse matmul

I'm leaning toward... a hybrid? TriXLinear can EXPOSE signatures but not do routing itself. A wrapper or FFN class handles routing logic.

Actually wait - the signatures are derived from weights. The weights are in TriXLinear. So TriXLinear should at minimum have a `get_signature()` method.

---

## Signature computation

Current approach:
```python
signature = weight.sum(dim=0).sign()
```

This works but... should we cache it? Recompute every forward? 

During training, weights change every step. So signatures change. We need fresh signatures.

During inference, weights are frozen. We could cache signatures.

Maybe:
- Training: compute on-the-fly
- Inference: cache via `layer.pack()` which already exists for weight packing

Yeah - extend `pack()` to also precompute and cache signatures. Clean.

---

## API Design

Current GatedFFN:
```python
ffn = GatedFFN(d_model, num_tiles)
out, gate = ffn(x)  # gate is computed internally
```

New ZeroRoutingFFN:
```python
ffn = ZeroRoutingFFN(d_model, num_tiles)
out, gate = ffn(x)  # gate emerges from signatures
```

Same API! Just different internal implementation. Could even be a flag:
```python
ffn = GatedFFN(d_model, num_tiles, routing='emergent')  # or 'learned' or 'random'
```

Hmm, but that's mixing concerns. Better to have separate classes that share interface.

Or... what if `GatedFFN` is deprecated and we just have `TriXFFN` that does emergent routing by default?

Need to think about backwards compatibility. Maybe:
- Keep `GatedFFN` as-is for now
- Add `TriXFFN` with emergent routing as the "blessed" way
- Eventually deprecate `GatedFFN`

---

## The TriXLinear changes

Current:
```python
class TriXLinear:
    def forward(self, x, gate):
        # uses gate to select tiles
```

Gate is required. But with emergent routing, the layer could compute its own gate.

Option A: Add optional auto-routing
```python
def forward(self, x, gate=None):
    if gate is None:
        gate = self.compute_emergent_gate(x)
    # proceed
```

Option B: Keep gate required, compute externally
```python
# In FFN:
gate = compute_routing(x, [self.up_proj, self.down_proj])
out = self.up_proj(x, gate)
```

Option A is cleaner for users but hides magic. Option B is explicit.

I think... Option A with clear documentation. Users who want control can pass explicit gate. Users who want simplicity pass nothing.

---

## Signature sharing between up_proj and down_proj

In an FFN: up_proj (d_model -> d_ff) and down_proj (d_ff -> d_model)

Currently we compute signature from up_proj. Makes sense - it's the "intake" that decides what to amplify.

But should down_proj have the same routing? Currently yes - same gate for both.

Alternative: route into up_proj based on input, route out of down_proj based on hidden state?

That's more complex. Let's keep it simple - single routing decision per FFN based on input and up_proj signatures.

---

## Performance considerations

Signature computation:
```python
signature = weight.sum(dim=0).sign()  # O(out_features * in_features)
```

This is the same cost as the routing in learned gates! But simpler operations (sum, sign vs matmul).

For inference with cached signatures:
```python
scores = input @ cached_signatures.T  # O(batch * in_features * num_tiles)
```

With num_tiles=4 and in_features=512, this is 2048 ops per input. Tiny compared to the main matmul.

Actually... signatures are ternary. Could we use ternary dot product?
```python
# Ternary signature means: 
# score = sum of input where sig=+1, minus sum where sig=-1
positive_mask = (signature == 1)
negative_mask = (signature == -1)
score = input[:, positive_mask].sum(-1) - input[:, negative_mask].sum(-1)
```

Hmm, masked indexing might not be faster. But interesting to note.

---

## What about top-k routing?

Current: top-1 (one tile per input)

Could extend to top-k (multiple tiles per input). Emergent routing naturally gives scores for all tiles - just take top-k instead of argmax.

For 75% sparsity with 4 tiles, top-1 is perfect.
For higher tile counts (16, 64), might want top-2 or top-4.

Implementation:
```python
def route(x, signatures, k=1):
    scores = x @ signatures.T
    _, top_indices = scores.topk(k, dim=-1)
    gate = torch.zeros_like(scores).scatter_(-1, top_indices, 1.0)
    return gate
```

Easy extension. Keep it in mind but start with top-1.

---

## Naming

- "Emergent routing" - describes the phenomenon
- "Zero routing" - catchy, emphasizes zero parameters
- "Signature routing" - describes the mechanism
- "Self-routing" - the layer routes itself

I like "Zero Routing" for marketing, "Signature-based routing" for technical docs.

Class names:
- `ZeroRoutingFFN`? 
- `SelfRoutingFFN`?
- `TriXFFN`? (implies it's THE TriX way)

Let's go with `TriXFFN` - it's the TriX-native FFN that uses all the TriX goodness including emergent routing.

---

## Testing strategy

Need tests for:
1. Signature computation correctness
2. Routing consistency (similar inputs -> same route)
3. Routing discrimination (different inputs -> can route differently)
4. Gradient flow (training works)
5. Inference speedup (packed mode is fast)
6. API compatibility (drop-in replacement for GatedFFN)

Can reuse the experimental tests, clean them up.

---

## Migration path

1. Add `get_signature()` to TriXLinear
2. Add optional `gate=None` with auto-routing to TriXLinear
3. Create `TriXFFN` class using emergent routing
4. Create `TriXTransformerBlock` using `TriXFFN`
5. Update examples to use new classes
6. Deprecation warnings on old GatedFFN? Or keep both?

Probably keep both - GatedFFN for experimentation/comparison, TriXFFN as recommended default.

---

## Open questions

1. Should signatures use STE for the sign() operation? Currently no gradient through sign. Is that okay?

   Thinking... the routing decision doesn't need gradients. Weights get gradients from the main forward pass. Routing adapts as a side effect. So no STE needed for signatures. This is actually a feature!

2. What if all tiles converge to similar signatures?

   Saw some evidence of this in experiments. Might need diversity encouragement. But maybe not - if task needs diversity, tiles will differentiate to reduce loss.

   Could add optional signature diversity regularization if needed. But start without it.

3. How does this interact with QAT (quantization-aware training)?

   Signatures use sign() which is already ternary. Should be fine. QAT affects the main weights, signatures automatically become "more ternary" as training progresses.

---

## Summary of implementation plan (rough)

1. Extend TriXLinear with `get_signature()` and optional auto-routing
2. Create TriXFFN with emergent routing
3. Create TriXBlock (transformer block) 
4. Add comprehensive tests
5. Update documentation and examples
6. Consider TriXModel (full transformer) as stretch goal

Let me reflect on this in the next doc...
