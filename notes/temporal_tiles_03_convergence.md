# Temporal Tiles: Convergence
*The synthesis. What has emerged and where does it lead?*

---

## The Core Claim

**Temporal tiles extend TriX from spatial routing to temporal routing, completing the replacement of attention with discrete, compilable, interpretable control.**

| Capability | Spatial Tiles | Temporal Tiles |
|------------|---------------|----------------|
| Routes on | input | (input, state) |
| Determines | what computes | when/how dynamics change |
| Outputs | transformed token | transformed token + new state |
| Compiles to | class → tile table | (regime, class) → (tile, regime) table |
| Analogy | specialist selection | mode switching |

---

## The Architecture

### Minimal Temporal Tile Layer

```python
class TemporalTileLayer(nn.Module):
    def __init__(self, d_model, d_state, num_tiles):
        self.d_state = d_state
        self.num_tiles = num_tiles
        
        # Temporal signatures: match on (input, state)
        self.signatures = nn.Parameter(torch.randn(num_tiles, d_model + d_state))
        
        # Each tile has:
        # - state update function
        # - output transformation
        self.state_updates = nn.ModuleList([
            nn.Linear(d_model + d_state, d_state) for _ in range(num_tiles)
        ])
        self.output_transforms = nn.ModuleList([
            nn.Linear(d_model + d_state, d_model) for _ in range(num_tiles)
        ])
    
    def forward(self, x, state):
        # x: (batch, d_model) - current input
        # state: (batch, d_state) - previous state
        
        # Concatenate for routing
        combined = torch.cat([x, state], dim=-1)
        
        # Route to tile
        scores = combined @ self.signatures.T
        tile_idx = scores.argmax(dim=-1)
        
        # Execute selected tile
        # (In practice, use gather or sparse ops)
        new_state = self.state_updates[tile_idx](combined)
        output = self.output_transforms[tile_idx](combined)
        
        return output, new_state, tile_idx
```

### Key Design Choices

1. **State size**: Small (16-64 dims). Mode selector, not full representation.

2. **Signature structure**: Concatenated (input, state). Could also be factored: `score = input_score + state_score`.

3. **State update**: Per-tile. Each tile owns its dynamics. Different tiles = different transitions.

4. **Initialization**: State starts at zero. First token routes based on input alone.

---

## The Experiments

### Experiment 1: Bracket Matching (The Atom)

**Task**: Classify sequences of parentheses as valid/invalid.

```
((()))  → valid
(()())  → valid
(()))   → invalid
)(      → invalid
```

**Why this task**:
- Requires counting (can't be done statelessly)
- Attention solves it (baseline exists)
- Minimal: only 2 input symbols
- Ground truth is deterministic

**Expected outcome**:
- 4-8 temporal tiles learn depth-tracking behavior
- Tile 0 = "depth 0", Tile 1 = "depth 1", etc.
- Transitions follow counting logic
- Can be compiled to explicit FSM

**Success criterion**: 99%+ accuracy with interpretable tile structure.

---

### Experiment 2: 6502 with Carry (The Validation)

**Task**: Predict ADC results including carry flag propagation.

```
ADC(255, 1, 0) → 0, carry=1
ADC(100, 50, 1) → 151, carry=0  # carry from previous
```

**Why this task**:
- We already have 92% spatial purity
- ADC specifically failed (4.8%) due to missing state
- Ground truth from FLYNNCONCEIVABLE
- Tests temporal state in realistic setting

**Expected outcome**:
- Temporal tiles learn carry-propagation regime
- ADC accuracy jumps from 4.8% to 95%+
- Carry state is observable in tile routing

**Success criterion**: ADC matches other operations in accuracy.

---

### Experiment 3: Simple Parsing (The Extension)

**Task**: Parse simple if/then/else structures.

```
if x then y else z → structure recognized
if x then if a then b else c else z → nested structure
```

**Why this task**:
- Multiple interacting regimes (if-mode, then-mode, else-mode)
- Requires tracking what's open/closed
- More complex than brackets, simpler than full language

**Expected outcome**:
- Tiles specialize to parsing states
- Nesting is handled by state, not attention
- Dispatch table describes a parser

**Success criterion**: Correct parse structure on held-out examples.

---

## The Compilation Story

Temporal tiles compile to state transition tables:

```python
temporal_dispatch = {
    # (current_regime, input_class): (tile_idx, next_regime)
    (DEPTH_0, OPEN): (1, DEPTH_1),
    (DEPTH_0, CLOSE): (ERR, ERROR),
    (DEPTH_1, OPEN): (2, DEPTH_2),
    (DEPTH_1, CLOSE): (0, DEPTH_0),
    (DEPTH_2, OPEN): (3, DEPTH_3),
    (DEPTH_2, CLOSE): (1, DEPTH_1),
    # ...
}
```

This is:
- **Readable**: You can see the FSM structure
- **Verifiable**: Check that transitions are correct
- **Editable**: Add/modify transitions surgically
- **Deployable**: O(1) lookup, no neural compute

The learned temporal behavior becomes a program artifact.

---

## The Hierarchy

Three levels of TriX, three levels of structure:

```
Level 1: Spatial Tiles (Mesa 1-3)
├── Routes on: input content
├── Learns: what computation to apply
├── Compiles to: class → tile
└── Answers: "which specialist?"

Level 2: Temporal Tiles (Mesa 4)
├── Routes on: (input, state)
├── Learns: when to switch dynamics
├── Compiles to: (regime, class) → (tile, regime)
└── Answers: "which phase?"

Level 3: Hierarchical Composition (Future)
├── Routes on: (input, state, context)
├── Learns: how to compose sub-programs
├── Compiles to: nested dispatch tables
└── Answers: "which sub-routine?"
```

Each level extends routing into a new dimension while preserving the core properties: discrete, compilable, interpretable.

---

## The Risks

### Risk 1: Long-Range Failure
State might not survive long sequences. Information decays through repeated transitions.

**Mitigation**: Test on increasing sequence lengths. If it fails at length N, we know the limit.

### Risk 2: State Collapse
All inputs might route to the same few tiles regardless of state. State becomes unused.

**Mitigation**: State-entropy regularizer. Penalize routing distributions that ignore state.

### Risk 3: Uninterpretable State
The learned state space might be as opaque as RNN hidden states.

**Mitigation**: Structured state (explicit dims for depth, mode, etc.) or clustering-based regime discovery.

### Risk 4: Training Instability
Discrete tile selection + state update might have bad gradients.

**Mitigation**: Straight-through estimator (proven in spatial tiles). Temperature annealing for tile selection.

---

## The Implications (If It Works)

### For AI Architecture
Attention is optional. Discrete state routing is a viable alternative for sequence modeling, with better interpretability.

### For AI Safety
Learned behavior can be extracted, read, verified, edited. Not post-hoc interpretation, but direct program inspection.

### For AI Engineering
Neural networks become systems with readable control flow. Version control applies to learned behavior.

### For Theory
The "right" computational model for sequence understanding might be discrete phases and transitions, not continuous mixing.

---

## The Next Step

**Build the bracket matcher.**

1. Implement `TemporalTileLayer` (minimal version above)
2. Generate bracket data (varying depths, lengths)
3. Train with claim tracking
4. Analyze: do tiles correspond to depth levels?
5. Compile: extract the FSM
6. Verify: does the FSM match ground truth?

If this works, we have the atom of temporal tiles.
If it fails, we learn exactly what's missing.

Either outcome advances the science.

---

## The Closing Thought

Spatial tiles asked: "Which specialist handles this input?"
Temporal tiles ask: "Which phase am I in, and what transition applies?"

The first question gave us addressable computation.
The second gives us addressable control flow.

Together, they might give us something attention was never designed to provide:

**A neural network you can read.**

---

*Convergence complete. The emergence is observed. The path is clear.*

*Mesa 4 awaits.*
