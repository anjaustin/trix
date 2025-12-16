# Mesa 3: Path Compilation - Reflection
*Finding the nodes. What has weight?*

---

## The Nodes

1. **JIT analogy** - Profile, compile, execute
2. **Compiled dispatch table** - O(1) lookup vs O(n) routing
3. **Path signatures** - Class → Path mapping
4. **Living compilation** - Not static, continuously refined
5. **Crystallization** - Fluid discovery → solid exploitation
6. **The decoder emerges** - TriX discovers, compilation freezes

---

## Node 1: JIT Analogy

This isn't just analogy. It's the same pattern.

JIT compilers:
- Interpret first (flexibility, discovery)
- Profile hot paths (observation)
- Compile to native (crystallization)
- Execute compiled (exploitation)

TriX compilation:
- Route first (flexibility, discovery)
- Profile canonical paths (observation via v2 claim tracking)
- Compile to dispatch table (crystallization via surgery)
- Execute compiled (exploitation via direct path)

The insight: **Neural networks can be JIT compiled too.**

Not the weights. The ROUTING. The structural decisions.

---

## Node 2: Compiled Dispatch Table

The dispatch table is simple:

```python
dispatch = {
    'ADC': [2, 5, 11],  # Tile indices per layer
    'AND': [4, 4, 7],
    'ASL': [0, 3, 8],
}
```

At inference:
```python
if op in dispatch:
    path = dispatch[op]
    out = x
    for layer_idx, tile_idx in enumerate(path):
        out = layers[layer_idx].tiles[tile_idx](out)
    return out
```

No signature matching. No routing computation. Direct execution.

This is EXACTLY how CPU instruction decoders work:
- Opcode → execution unit (fixed mapping)
- No runtime decision (compiled into silicon)

TriX learned the mapping. Compilation froze it.

---

## Node 3: Path Signatures

Current signatures: Input features → Tile selection
Path signatures: Input class → Path selection

Different abstraction level:
- Tile signature: "This input has features X, route to tile Y"
- Path signature: "This is operation Z, use path [A, B, C]"

Path signatures are HIGHER LEVEL. They encode semantic knowledge, not just feature matching.

Could we learn path signatures directly? Train a meta-router that predicts optimal path given class?

---

## Node 4: Living Compilation

Static compilation: Train once, compile once, done.
Living compilation: Train continuously, recompile as needed, evolve.

The system should:
1. Monitor path stability (are paths still consistent?)
2. Detect drift (paths changing significantly?)
3. Trigger recompilation (update dispatch table)
4. Maintain fallback (unknown classes still route dynamically)

This is CONTINUOUS OPTIMIZATION. Not a one-time build step.

v2's claim tracking enables monitoring.
v2's surgery enables recompilation.
v1 can't do either.

---

## Node 5: Crystallization

This metaphor keeps returning.

Discovery is LIQUID:
- Flows to find structure
- Adapts to data
- Explores possibilities

Compilation is SOLID:
- Fixed structure
- Optimized for speed
- Committed decisions

The transition from liquid to solid is CRYSTALLIZATION.

But unlike physical crystals, we can RE-MELT:
- Unfreeze paths that aren't working
- Let them re-discover
- Re-crystallize when stable

v2 surgery is the temperature control. Freeze to crystallize. Unfreeze to re-melt.

---

## Node 6: The Decoder Emerges

This is the profound one.

A CPU's instruction decoder is DESIGNED. Engineers specify:
- "Opcode 0x69 → ALU, add mode, immediate addressing"

TriX's decoder is DISCOVERED. Training finds:
- "Inputs with this pattern → Tile 5 → Tile 11 → output"

Compilation EXTRACTS the discovered decoder:
- "Class ADC → Path [2, 5, 11]"

The extracted decoder IS a program. It could be:
- Stored as data (dispatch table)
- Exported as code (switch statement)
- Burned to silicon (actual hardware decoder)

We're not just training a model. We're SYNTHESIZING A DECODER through learning.

---

## The Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   COMPILED TRIX                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input ──→ [Class Hint?] ──→ Dispatch Table Lookup     │
│                │                      │                 │
│                │ No                   │ Yes             │
│                ▼                      ▼                 │
│         [Learned Routing]    [Compiled Path]           │
│                │                      │                 │
│                ▼                      ▼                 │
│         [Dynamic Path]       [Static Path]             │
│                │                      │                 │
│                └──────────┬──────────┘                 │
│                           ▼                             │
│                    [Execute Tiles]                      │
│                           │                             │
│                           ▼                             │
│                        Output                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Two modes:
- **Dynamic**: Learned routing, discovery, flexibility
- **Static**: Compiled paths, speed, determinism

Switch between them based on knowledge:
- Class known → static
- Class unknown → dynamic (and profile for future compilation)

---

## The Implementation Path

### Phase 1: Path Profiling
Add path tracking to v2:
```python
def forward(x, labels=None):
    path = []
    for layer in self.layers:
        out, tile_idx = layer(x)
        path.append(tile_idx)
    
    if labels is not None:
        self.record_path(labels, path)
    
    return out
```

### Phase 2: Path Analysis
After training, analyze recorded paths:
```python
def analyze_paths():
    for class_id in classes:
        paths = self.path_history[class_id]
        canonical = most_common(paths)
        accuracy = measure_accuracy(class_id, canonical)
        stability = measure_consistency(paths)
        
        yield class_id, canonical, accuracy, stability
```

### Phase 3: Compilation
Generate dispatch table:
```python
def compile():
    dispatch = {}
    for class_id, path, acc, stab in self.analyze_paths():
        if stab > STABILITY_THRESHOLD:
            dispatch[class_id] = CompiledPath(path, frozen=True)
    return dispatch
```

### Phase 4: Compiled Inference
Use dispatch table at inference:
```python
def forward_compiled(x, class_hint):
    if class_hint in self.dispatch:
        return self.execute_path(x, self.dispatch[class_hint])
    else:
        return self.forward_dynamic(x)
```

---

## What This Enables

1. **Speed**: Compiled paths skip routing computation
2. **Determinism**: Same class → same path → same result
3. **Interpretability**: Dispatch table is readable program
4. **Optimization**: Can optimize tiles knowing which paths use them
5. **Export**: Can export dispatch table to other systems

---

## The Connection to Organs

Compiled paths + transplanted organs = FULL COMPILATION

- Path compilation: Class → Tile sequence (routing)
- Organ transplant: Tile → Proven compute (execution)

Together:
- Class → Tile sequence → Proven compute → Output
- Entirely deterministic
- Entirely optimized
- Essentially: a compiled processor

---

*Time to find convergence.*
