# Mesa 3: Path Compilation - Raw Exploration
*If paths are programs, can we compile them?*

---

## The Question

During training, TriX discovers paths - sequences of tile selections across layers.

"ADC with overflow" → L1:Tile2 → L2:Tile5 → L3:Tile11
"AND operation" → L1:Tile4 → L2:Tile4 → L3:Tile7

These paths ARE programs. They emerged from learning.

Can we:
1. Discover the optimal path per class?
2. Freeze/memorialize those paths?
3. At inference, skip routing and use compiled paths directly?

---

## The Analogy: JIT Compilation

JIT compilers in VMs (Java, JavaScript, Python):
1. Start with interpreted execution (slow, flexible)
2. Profile which paths are hot (frequently executed)
3. Compile hot paths to native code (fast, fixed)
4. Execute compiled paths directly

TriX could do the same:
1. Start with learned routing (flexible, discovers structure)
2. Profile which paths are canonical per class
3. Compile canonical paths to static dispatch (fast, fixed)
4. At inference, dispatch directly without routing computation

---

## The Mechanism

### Training Phase (Discovery)
```
Input → Routing → Path Selection → Computation → Output
           ↓
    [Record path taken]
           ↓
    Path: [L1:T2, L2:T5, L3:T11]
    Class: ADC_overflow
```

### Compilation Phase (Optimization)
```
For each class:
    paths = all paths observed for this class
    canonical = most_common(paths) or most_accurate(paths)
    compiled_table[class] = canonical
```

### Inference Phase (Execution)
```
If class is known:
    path = compiled_table[class]  # O(1) lookup
    Execute path directly (skip routing)
Else:
    Use learned routing (fallback)
```

---

## What This Buys Us

### Speed
- Routing computation: O(num_tiles * d_model) per layer
- Compiled dispatch: O(1) table lookup
- For known classes, inference is MUCH faster

### Determinism
- Learned routing can be slightly stochastic (numerical precision)
- Compiled paths are fixed
- Same input → guaranteed same path

### Interpretability
- Compiled table IS the program
- "ADC_overflow uses path [2,5,11]" is readable
- Can verify paths make sense

### Optimization Target
- Once paths are compiled, can optimize tile compute for those paths
- Dead code elimination: if no path uses Tile 9, remove it
- Path fusion: if two classes share L1:T2→L2:T5, share that computation

---

## The v2 Connection

v2 enables this:

| Requirement | v2 Feature |
|-------------|------------|
| Know which classes take which paths | Claim tracking |
| Freeze paths that work | Surgery (freeze) |
| Store compiled paths | Signature table |
| Bypass routing for known cases | Score override |

v1 can't do this. No observability, no surgery.

---

## New Concept: Path Signature

Currently, signatures route individual inputs to tiles.

A PATH SIGNATURE could route input CLASSES to paths:

```python
path_signatures = {
    'ADC': PathSignature(layers=[2, 5, 11], frozen=True),
    'AND': PathSignature(layers=[4, 4, 7], frozen=True),
    'ASL': PathSignature(layers=[0, 3, 8], frozen=True),
}

def forward(x, class_hint=None):
    if class_hint and class_hint in path_signatures:
        # Compiled path - direct execution
        path = path_signatures[class_hint]
        return execute_path(x, path)
    else:
        # Learned routing - discovery mode
        return route_and_execute(x)
```

---

## The Compilation Process

### Step 1: Profile
Train normally, record all (input_class, path) pairs.

### Step 2: Analyze
For each class, compute:
- Path frequency distribution
- Path accuracy (which path gives best results?)
- Path stability (does the path vary or is it consistent?)

### Step 3: Select
Choose canonical path per class:
- Option A: Most frequent (the "natural" path)
- Option B: Most accurate (the "best" path)
- Option C: Most stable (the "reliable" path)

### Step 4: Compile
Store canonical paths in dispatch table.
Optionally freeze the tiles in each path.

### Step 5: Verify
Run inference with compiled paths.
Compare accuracy to learned routing.
Should be equal or better (we picked optimal paths).

---

## The Feedback Loop

Compilation isn't one-time. It's iterative:

```
Train → Profile → Compile → Deploy → Monitor → Retrain → Recompile
                                        ↓
                              [New patterns detected]
                                        ↓
                              [Add to compiled table]
```

The system LEARNS new paths and COMPILES them as they stabilize.

Living compilation. Not static.

---

## Edge Cases

### Unknown Class
Input arrives with no class hint, or class not in compiled table.
Fallback to learned routing. Profile the path taken. Maybe compile later.

### Class Ambiguity
Input could be multiple classes. Which compiled path?
Option A: Use routing to disambiguate
Option B: Ensemble multiple paths
Option C: Require explicit class hint

### Path Drift
Training continues, paths shift. Compiled table becomes stale.
Need recompilation trigger: "path distribution changed significantly"

### Path Conflict
Two classes want the same path. Is that okay?
Maybe. If the path works for both, it's shared compute.
If accuracy differs, need separate paths.

---

## The Deeper Insight

Compilation is CRYSTALLIZATION.

Learned routing is fluid - it adapts, explores, discovers.
Compiled paths are solid - they're commitments, optimizations.

The fluid phase finds structure.
The solid phase exploits structure.

Both are needed. v2 enables both.

---

## What This Means for 6502

If we compile paths for 6502 operations:

```
compiled_6502 = {
    'ADC': [ALU_tile, carry_tile, flag_tile],
    'AND': [LOGIC_tile, flag_tile],
    'ASL': [SHIFT_tile, carry_tile, flag_tile],
    ...
}
```

At inference:
- Opcode known → path known → direct execution
- No routing overhead
- Deterministic behavior
- Essentially: we've compiled a CPU instruction decoder

TriX discovered the decoder. Compilation froze it.

---

## Fragments

- Could compiled paths be exported as actual code? TriX → static dispatch → C code?
- Path compilation is like "attention pattern crystallization" but for routing
- The compiled table is a PROGRAM in a declarative language
- What if paths have conditionals? "If overflow, take path A, else path B"
- Could we learn the compilation policy itself? Meta-learning for path selection

---

*End raw dump. What's actually here?*
