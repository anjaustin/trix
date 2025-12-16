# Mesa 3: Path Compilation - Convergence
*The synthesis.*

---

## The Answer

**Yes. Paths can be compiled.**

Discovery (training) → Profiling (observation) → Compilation (crystallization) → Execution (exploitation)

This is JIT compilation for neural routing.

---

## The Architecture

```
                    ┌─────────────────────┐
                    │   CLASS HINT        │
                    │   (if known)        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
         ┌─────────│   DISPATCH TABLE    │─────────┐
         │ miss    │   (compiled paths)  │  hit    │
         │         └─────────────────────┘         │
         ▼                                         ▼
┌─────────────────┐                     ┌─────────────────┐
│ DYNAMIC ROUTING │                     │  STATIC PATH    │
│ (learned, flex) │                     │ (compiled, fast)│
└────────┬────────┘                     └────────┬────────┘
         │                                       │
         └───────────────┬───────────────────────┘
                         ▼
              ┌─────────────────────┐
              │   EXECUTE TILES     │
              │   (organ compute)   │
              └──────────┬──────────┘
                         │
                         ▼
                      OUTPUT
```

**Two modes, one system.**

---

## The Process

### 1. Train (Discovery)
```
Normal TriX training.
Tiles specialize. Paths emerge.
v2 claim tracking records everything.
```

### 2. Profile (Observation)
```
For each class:
    paths = observed routing paths
    canonical = most_common(paths)
    stability = consistency(paths)
```

### 3. Compile (Crystallization)
```
dispatch_table = {}
for class_id, path, stability in profiles:
    if stability > threshold:
        dispatch_table[class_id] = freeze(path)
```

### 4. Execute (Exploitation)
```
if class in dispatch_table:
    use compiled path (O(1) lookup, no routing)
else:
    use dynamic routing (discovery mode)
```

### 5. Monitor (Living System)
```
Track path drift.
Recompile when distributions shift.
Add new classes as they stabilize.
```

---

## What Compilation Provides

| Benefit | Description |
|---------|-------------|
| **Speed** | Skip routing computation for known classes |
| **Determinism** | Same class → same path → same result |
| **Interpretability** | Dispatch table is a readable program |
| **Optimization** | Dead tile elimination, path fusion |
| **Export** | Dispatch table can become static code |

---

## The v2 Requirement

| Compilation Phase | v2 Feature Required |
|-------------------|---------------------|
| Profiling | Claim tracking (see paths) |
| Analysis | Path statistics (from tracking) |
| Compilation | Surgery (freeze paths) |
| Monitoring | Claim tracking (detect drift) |
| Recompilation | Surgery (unfreeze, refreeze) |

**v1 cannot compile. No observability, no surgery.**

v2 was designed for this even before we knew it.

---

## The Full Stack

```
Mesa 1: Discovery
   TriX finds structure (92% purity)
   
Mesa 2: Partnership  
   v2 enables observation and editing
   
Mesa 3: Compilation
   Paths crystallize into dispatch tables
   
Mesa 4: ??? 
   Compiled TriX + Transplanted Organs = ???
```

---

## Mesa 4 Preview

Compiled paths (routing) + Transplanted organs (compute) = **SYNTHESIZED PROCESSOR**

```
Input → Compiled Dispatch → Frozen Path → Proven Organ → Output
           (learned)         (frozen)      (engineered)
```

- Routing: Discovered by TriX, compiled to table
- Compute: Engineered by FLYNNCONCEIVABLE, transplanted to tiles
- Result: 100% accurate, deterministic, fast, interpretable

This is the factory output. A processor synthesized from learned structure and proven components.

---

## The Minimal Implementation

```python
class CompiledTriX(nn.Module):
    def __init__(self, base_model):
        self.layers = base_model.layers
        self.dispatch = {}  # class → path
    
    def compile(self, profiling_data, stability_threshold=0.8):
        for class_id, paths in profiling_data.items():
            canonical = Counter(paths).most_common(1)[0][0]
            stability = paths.count(canonical) / len(paths)
            if stability > stability_threshold:
                self.dispatch[class_id] = canonical
    
    def forward(self, x, class_hint=None):
        if class_hint in self.dispatch:
            return self._execute_compiled(x, self.dispatch[class_hint])
        else:
            return self._execute_dynamic(x)
    
    def _execute_compiled(self, x, path):
        for layer_idx, tile_idx in enumerate(path):
            x = self.layers[layer_idx].tiles[tile_idx](x)
        return x
    
    def _execute_dynamic(self, x):
        for layer in self.layers:
            x, _ = layer.route_and_execute(x)
        return x
```

---

## The Conclusion

**Paths are programs. Programs can be compiled.**

TriX discovers the programs (routing paths).
v2 lets us observe them (claim tracking).
Compilation crystallizes them (dispatch table).
Inference executes them (direct or dynamic).

This is the third mesa: **Learning that becomes code.**

---

*The factory is not just possible. It's architected.*
