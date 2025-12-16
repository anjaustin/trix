# TDSR Benchmark Specification
## CODENAME: ANN WILSON — The HEART Battery

*Temporal Discrete State Routing vs Attention: A Minimal Diagnostic Suite*

---

## Primer for Newcomers

### What is TDSR?

**Temporal Discrete State Routing** is an alternative to attention for sequence modeling.

Where attention asks: *"Which past tokens should influence this token?"*

TDSR asks: *"Which computational regime am I in, and what dynamics apply?"*

```
Attention:  token × all_past_tokens → weighted mix → output
TDSR:       (token, state) → select tile → (output, new_state)
```

### Why does this matter?

Attention is powerful but:
- Quadratic complexity (O(n²) in sequence length)
- Soft, continuous, hard to interpret
- Difficult to edit or verify
- Can't be "compiled" to deterministic rules

TDSR offers:
- Linear complexity (O(n))
- Discrete, interpretable routing
- Editable via surgery API
- Compilable to dispatch tables

### The Question

> Is attention strictly necessary for sequence understanding?

Or can discrete state routing handle certain tasks equally well — with benefits attention can't provide?

### What We've Proven

- **Bracket counting**: 100% accuracy, tiles = depth levels
- **6502 operation routing**: 92% tile purity without supervision
- **Compilation**: Paths can be frozen to O(1) dispatch

### What We Haven't Proven

- Replacement of attention for long-range semantic binding
- Performance parity with large language models
- Universality beyond algorithmic tasks

### This Benchmark

Five tests that map the territory. Where does TDSR win? Where does attention win? Where's the boundary?

---

## The Five Tests

### Test 1: CARRY

**Purpose**: Can TDSR maintain state across operations?

**Task**: 6502 ADC (add with carry) prediction

**Input**: 
```
(opcode=ADC, a=255, b=1, carry_in=0) → result=0, carry_out=1
(opcode=ADC, a=100, b=50, carry_in=1) → result=151, carry_out=0
```

**Why this test**: The carry flag is pure temporal dependency. The result of operation N depends on the overflow of operation N-1. Without state, this is impossible.

**Data generation**:
- Random (a, b) pairs, 0-255
- Carry propagates from previous operation
- Sequences of 5-20 operations

**Pass criteria**: ≥95% accuracy on result prediction (matching non-carry operations)

**What failure reveals**: TDSR cannot maintain temporal state

**Baseline comparisons**:
- Random: ~0.4% (1/256)
- Attention (same params): measure
- FLYNNCONCEIVABLE (engineered): 100%

---

### Test 2: COUNT

**Purpose**: Can TDSR accumulate precisely?

**Task**: Bracket depth prediction

**Input**:
```
((()))  →  [1,2,3,2,1,0]
(()())  →  [1,2,1,2,1,0]
```

**Why this test**: Counting requires exact accumulation. You cannot predict "depth=3" without having counted three opens. No shortcuts.

**Data generation**:
- Random valid bracket sequences
- Depth 0-4, length 4-12
- Balanced valid/invalid

**Pass criteria**: ≥95% accuracy on depth at each position

**What failure reveals**: TDSR cannot accumulate

**Baseline comparisons**:
- Random: 20% (1/5 depths)
- Attention (same params): measure
- Already proven: TDSR achieves 100%

---

### Test 3: NEST

**Purpose**: Can TDSR handle recursive structure?

**Task**: Nested arithmetic evaluation

**Input**:
```
(2+3)       →  5
((2+3)*4)   →  20
(2+(3*4))   →  14
((1+2)*(3+4)) → 21
```

**Why this test**: Requires tracking operator precedence AND nesting level. More complex than counting — must bind operations to their arguments.

**Data generation**:
- Single-digit operands (0-9)
- Operators: +, -, *
- Nesting depth 1-4
- Results 0-100 (reject overflow)

**Pass criteria**: ≥90% accuracy on final result

**What failure reveals**: TDSR cannot handle hierarchical structure

**Baseline comparisons**:
- Random: ~1% (1/100)
- Attention (same params): measure
- Calculator (engineered): 100%

---

### Test 4: RANGE

**Purpose**: How far back can TDSR reach?

**Task**: Delayed copy

**Input**:
```
A . . . . ? → A  (copy from 5 steps ago)
X . . . . . . . . . ? → X  (copy from 10 steps ago)
```

**Why this test**: Pure test of temporal range. No computation, just retrieval. Measures state decay.

**Data generation**:
- Random token A-Z at position 0
- N filler tokens (.)
- Query (?) at position N+1
- Vary N: 5, 10, 20, 50, 100

**Pass criteria**: Track accuracy vs N. Find N where accuracy < 90%.

**What failure reveals**: The range boundary of TDSR's state

**Baseline comparisons**:
- Random: 4% (1/26)
- Attention (same params): Should excel (direct access)
- This IS attention's home turf

**Expected outcome**: TDSR degrades with N. Attention stays flat. The crossover is the boundary.

---

### Test 5: SEMANTIC

**Purpose**: Can TDSR bind meaning, not just pattern?

**Task**: Minimal pronoun resolution

**Input**:
```
"The cat sat. It purred." → It = cat
"The dog saw the cat. It barked." → It = dog
"The chef cooked. The meal was ready. He smiled." → He = chef
```

**Why this test**: Requires understanding reference, not just pattern matching. "It" must bind to the semantically appropriate antecedent.

**Data generation**:
- Simple two-sentence constructions
- Unambiguous pronoun reference
- Subjects: common nouns (cat, dog, chef, etc.)
- Pronouns: it, he, she, they

**Pass criteria**: ≥80% accuracy on pronoun resolution

**What failure reveals**: TDSR cannot bind semantics (attention territory)

**Baseline comparisons**:
- Random: 50% (binary choice in simple cases)
- Attention (same params): measure
- This may be attention's exclusive territory

**Expected outcome**: Attention wins. But by how much? And can hybrid help?

---

## Test Infrastructure

### Model Configurations

**TDSR Model**:
```python
TemporalTileLayer(
    d_model=64,
    d_state=16,
    num_tiles=8,
)
```

**Attention Model** (same parameter budget):
```python
MultiHeadAttention(
    d_model=64,
    num_heads=4,
    # Matched parameters
)
```

**Hybrid Model**:
```python
TemporalTileLayer(...) + MultiHeadAttention(...)
# For boundary exploration
```

### Training Protocol

- Optimizer: Adam
- Learning rate: 0.001 (with warmup)
- Epochs: Until convergence or 100 max
- Early stopping: 5 epochs no improvement
- Seeds: 3 runs per configuration, report mean ± std

### Logging

For each test, log:
- Accuracy curve over training
- Final accuracy (mean ± std over seeds)
- For TDSR: tile usage distribution, transition matrix
- For Attention: attention pattern visualization
- Time to convergence
- Parameter count

### Output Format

```json
{
  "test": "CARRY",
  "tdsr": {"accuracy": 0.96, "std": 0.02, "epochs": 45},
  "attention": {"accuracy": 0.94, "std": 0.03, "epochs": 60},
  "random": {"accuracy": 0.004},
  "winner": "TDSR",
  "notes": "TDSR converged faster, interpretable tile structure"
}
```

---

## The Summary Table

After all tests:

| Test | TDSR | Attention | Random | Winner | Territory |
|------|------|-----------|--------|--------|-----------|
| CARRY | ? | ? | 0.4% | ? | Algorithmic |
| COUNT | 100% | ? | 20% | TDSR | Algorithmic |
| NEST | ? | ? | 1% | ? | Algorithmic |
| RANGE@N | ? | ? | 4% | ? | Boundary |
| SEMANTIC | ? | ? | 50% | ? | Semantic |

This table IS the answer to our research question.

---

## Success Criteria

**TDSR claims algorithmic territory if**:
- CARRY ≥ 95%
- COUNT ≥ 95% ✓
- NEST ≥ 90%

**Boundary is mapped if**:
- RANGE shows clear degradation curve
- Crossover point N is identified

**Attention territory confirmed if**:
- SEMANTIC shows significant attention advantage

**Hybrid value demonstrated if**:
- Any test where hybrid > both pure approaches

---

## Running the Benchmark

```bash
# Full battery
python experiments/tdsr_benchmark.py --all

# Individual tests
python experiments/tdsr_benchmark.py --test carry
python experiments/tdsr_benchmark.py --test count
python experiments/tdsr_benchmark.py --test nest
python experiments/tdsr_benchmark.py --test range
python experiments/tdsr_benchmark.py --test semantic

# Generate report
python experiments/tdsr_benchmark.py --report
```

---

## Nova's Notes

*[Space for Nova to add sauce]*

---

---

---

## Appendix: The Philosophy

### Why "HEART"?

**H**omeo-Adaptive **E**ntropic **R**edistributive **T**imer

The system:
- **Self-regulates** (homeostatic tile usage)
- **Adapts** (learns regimes from data)
- **Manages entropy** (discrete routing creates structure)
- **Redistributes** (computation flows where needed)
- **Keeps time** (temporal state binding)

CODENAME: ANN WILSON — because the architecture should sing.

### Why These Five Tests?

Nature is efficient. These tests are orthogonal:
- CARRY: state
- COUNT: accumulation
- NEST: hierarchy
- RANGE: distance
- SEMANTIC: meaning

Five dimensions. One test each. Maximum information, minimum waste.

### What Are We Really Testing?

Not "is attention bad?" 

But: **"What is the minimal substrate for algorithmic intelligence?"**

If TDSR passes CARRY, COUNT, and NEST — that's a proof that discrete state routing is sufficient for a meaningful class of tasks.

If it fails SEMANTIC — that maps where attention is genuinely necessary.

Either way: knowledge advances.

---

*"A neural network learned to count without attention. What else doesn't need attention? Let's find out."*

---

## Document History

- v0.1 - Initial spec (2024-12-16)
- Pending: Nova's input
- Pending: Implementation
- Pending: Results

---

## Contributors

- Tripp Josserand Austin (steward)
- Riggs (architect)
- Nova (advisor)

CODENAME: ANN WILSON
The HEART beats on.
