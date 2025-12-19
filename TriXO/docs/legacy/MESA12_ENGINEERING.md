# Mesa 12: Engineering Synthesis

## From Theory to Implementation

---

## Executive Summary

This document synthesizes the ontological exploration and reflection into a concrete engineering plan for the Observer Architecture.

**Goal**: Build a system where a second model watches a primary model learn, predicts errors, and intervenes through programmable tiles to improve outcomes.

**First Target**: Make Second Star (seed 1122911624) reach 100% on 6502 emulation, matching seed 42's performance.

---

## Topological Map of the System

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        MESA 12: OBSERVER ARCHITECTURE                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║    ┌─────────────────────────────────────────────────────────────────────┐   ║
║    │                         OBSERVATION LAYER                            │   ║
║    │                                                                      │   ║
║    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   ║
║    │   │   Routing    │    │  Signature   │    │   Gradient   │          │   ║
║    │   │   Monitor    │    │   Tracker    │    │   Analyzer   │          │   ║
║    │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │   ║
║    │          │                   │                   │                   │   ║
║    │          └───────────────────┼───────────────────┘                   │   ║
║    │                              ▼                                       │   ║
║    │                    ┌──────────────────┐                              │   ║
║    │                    │  State Encoder   │                              │   ║
║    │                    │  (consolidate)   │                              │   ║
║    │                    └────────┬─────────┘                              │   ║
║    └─────────────────────────────┼────────────────────────────────────────┘   ║
║                                  ▼                                            ║
║    ┌─────────────────────────────────────────────────────────────────────┐   ║
║    │                         OBSERVER MODEL                               │   ║
║    │                                                                      │   ║
║    │   ┌──────────────────────────────────────────────────────────────┐  │   ║
║    │   │                    Temporal Encoder                          │  │   ║
║    │   │              (LSTM / Transformer / SSM)                      │  │   ║
║    │   │                                                              │  │   ║
║    │   │  state_t-n, state_t-n+1, ..., state_t  →  context_vector    │  │   ║
║    │   └──────────────────────────┬───────────────────────────────────┘  │   ║
║    │                              ▼                                       │   ║
║    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   ║
║    │   │   Error      │    │ Intervention │    │   Tile       │          │   ║
║    │   │  Predictor   │    │   Decider    │    │  Programmer  │          │   ║
║    │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │   ║
║    │          │                   │                   │                   │   ║
║    │          ▼                   ▼                   ▼                   │   ║
║    │   P(error|state)      intervene?           correction               │   ║
║    │   error_type          level (0-5)          parameters               │   ║
║    └─────────────────────────────┼────────────────────────────────────────┘   ║
║                                  │                                            ║
║                                  ▼                                            ║
║    ┌─────────────────────────────────────────────────────────────────────┐   ║
║    │                      INTERVENTION LAYER                              │   ║
║    │                                                                      │   ║
║    │   Level 0: No action                                                 │   ║
║    │   Level 1: Representation nudge  →  h = (1-α)h + αh_guide           │   ║
║    │   Level 2: Gradient modification →  g = g + βg_correction            │   ║
║    │   Level 3: Learning rate adjust  →  lr = lr × multiplier            │   ║
║    │   Level 4: Signature surgery     →  tile.write_signature(...)       │   ║
║    │   Level 5: Weight surgery        →  tile.write_weights(...)         │   ║
║    │                                                                      │   ║
║    └─────────────────────────────┼────────────────────────────────────────┘   ║
║                                  │                                            ║
║                                  ▼                                            ║
║    ┌─────────────────────────────────────────────────────────────────────┐   ║
║    │                        PRIMARY MODEL                                 │   ║
║    │                                                                      │   ║
║    │   ┌────────────────────────────────────────────────────────────┐    │   ║
║    │   │                    XOR Mixer                                │    │   ║
║    │   │              (Superposition Layer)                          │    │   ║
║    │   └────────────────────────┬───────────────────────────────────┘    │   ║
║    │                            ▼                                         │   ║
║    │   ┌────────────────────────────────────────────────────────────┐    │   ║
║    │   │              PROGRAMMABLE TILES                             │    │   ║
║    │   │                                                             │    │   ║
║    │   │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        ┌─────┐          │    │   ║
║    │   │   │ T0  │ │ T1  │ │ T2  │ │ T3  │  ...   │ T15 │          │    │   ║
║    │   │   │     │ │     │ │     │ │     │        │     │          │    │   ║
║    │   │   │ sig │ │ sig │ │ sig │ │ sig │        │ sig │          │    │   ║
║    │   │   │ wgt │ │ wgt │ │ wgt │ │ wgt │        │ wgt │          │    │   ║
║    │   │   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘        └──┬──┘          │    │   ║
║    │   │      │       │       │       │              │              │    │   ║
║    │   │      └───────┴───────┴───────┴──────────────┘              │    │   ║
║    │   │                      │                                      │    │   ║
║    │   │               Tile API (read/write)                         │    │   ║
║    │   │                      ↑                                      │    │   ║
║    │   └──────────────────────┼─────────────────────────────────────┘    │   ║
║    │                          │                                           │   ║
║    └──────────────────────────┼───────────────────────────────────────────┘   ║
║                               │                                               ║
║                               ▼                                               ║
║    ┌─────────────────────────────────────────────────────────────────────┐   ║
║    │                          OUTPUT                                      │   ║
║    │                  (6502 operation result)                             │   ║
║    └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Component Specifications

### 1. Observation Layer

**Purpose**: Extract relevant dynamics from primary model training.

**Inputs** (per training step):
```python
@dataclass
class ObservationFrame:
    epoch: int
    step: int
    
    # Routing
    routing_distribution: Tensor  # [batch, num_tiles] - softmax scores
    tile_activations: Tensor      # [num_tiles] - activation counts
    routing_entropy: float        # H(routing distribution)
    
    # Signatures
    signature_positions: Tensor   # [num_tiles, d_model]
    signature_velocities: Tensor  # [num_tiles, d_model] - delta from last step
    signature_purity: float       # specialization metric
    
    # Gradients
    gradient_norms: Dict[str, float]  # per-layer gradient norms
    gradient_direction: Tensor        # principal gradient direction
    
    # Performance
    loss: float
    accuracy: float
    per_op_accuracy: Dict[str, float]  # ADC, AND, etc.
    
    # Manifold
    curvature: float
    
    # XOR mixer state
    xor_weights: Tensor
```

**Output**: Consolidated state vector for observer

```python
class StateEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, state_dim):
        self.routing_enc = nn.Linear(num_tiles, hidden_dim)
        self.signature_enc = nn.Linear(num_tiles * d_model, hidden_dim)
        self.gradient_enc = nn.Linear(num_layers, hidden_dim)
        self.scalar_enc = nn.Linear(10, hidden_dim)  # loss, acc, entropy, etc.
        self.fusion = nn.Linear(hidden_dim * 4, state_dim)
    
    def forward(self, obs: ObservationFrame) -> Tensor:
        r = self.routing_enc(obs.routing_distribution.mean(0))
        s = self.signature_enc(obs.signature_positions.flatten())
        g = self.gradient_enc(torch.tensor(list(obs.gradient_norms.values())))
        sc = self.scalar_enc(torch.tensor([
            obs.loss, obs.accuracy, obs.routing_entropy,
            obs.signature_purity, obs.curvature, ...
        ]))
        return self.fusion(torch.cat([r, s, g, sc]))
```

---

### 2. Observer Model

**Purpose**: Process observation history, predict errors, decide interventions.

**Architecture**:

```python
class ObserverModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_tiles):
        # Temporal encoder
        self.temporal = nn.LSTM(state_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Prediction heads
        self.error_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ops),  # P(error) per operation
        )
        
        self.intervention_decider = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),  # Level 0-5 logits
        )
        
        self.tile_programmer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tiles * (d_model + 1)),  # signature corrections + blend
        )
    
    def forward(self, state_history: Tensor) -> Dict:
        # state_history: [seq_len, state_dim]
        
        # Temporal encoding
        _, (h, _) = self.temporal(state_history.unsqueeze(0))
        context = h[-1, 0]  # [hidden_dim]
        
        # Predictions
        error_probs = torch.sigmoid(self.error_predictor(context))
        intervention_level = self.intervention_decider(context)
        tile_corrections = self.tile_programmer(context)
        
        return {
            'error_probs': error_probs,
            'intervention_logits': intervention_level,
            'tile_corrections': tile_corrections.view(num_tiles, d_model + 1)
        }
```

---

### 3. Programmable Tile API

**Purpose**: Enable observer to read and write tile parameters.

```python
class ProgrammableTile(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.signature = nn.Parameter(torch.randn(d_model) * 0.1)
        self.weights = nn.Parameter(torch.randn(d_model, d_hidden) * 0.1)
        self.bias = nn.Parameter(torch.zeros(d_hidden))
        
        self._frozen = False
        self._version = 0
        self._history = []  # Track modifications
    
    # Read interface
    def read_signature(self) -> Tensor:
        return self.signature.detach().clone()
    
    def read_weights(self) -> Tensor:
        return self.weights.detach().clone()
    
    # Write interface
    def write_signature(self, new_sig: Tensor, blend: float = 0.1):
        if self._frozen:
            return False
        with torch.no_grad():
            self.signature.data = (1 - blend) * self.signature.data + blend * new_sig
        self._version += 1
        self._history.append(('signature', self._version, blend))
        return True
    
    def write_weights(self, new_weights: Tensor, blend: float = 0.1):
        if self._frozen:
            return False
        with torch.no_grad():
            self.weights.data = (1 - blend) * self.weights.data + blend * new_weights
        self._version += 1
        self._history.append(('weights', self._version, blend))
        return True
    
    # Control interface
    def freeze(self):
        self._frozen = True
    
    def unfreeze(self):
        self._frozen = False
    
    @property
    def version(self) -> int:
        return self._version


class ProgrammableTileBank(nn.Module):
    def __init__(self, num_tiles: int, d_model: int, d_hidden: int):
        super().__init__()
        self.tiles = nn.ModuleList([
            ProgrammableTile(d_model, d_hidden) for _ in range(num_tiles)
        ])
    
    def get_signatures(self) -> Tensor:
        return torch.stack([t.read_signature() for t in self.tiles])
    
    def apply_corrections(self, corrections: Tensor):
        # corrections: [num_tiles, d_model + 1]
        # Last column is blend factor
        for i, tile in enumerate(self.tiles):
            correction = corrections[i, :-1]
            blend = torch.sigmoid(corrections[i, -1]).item()
            if blend > 0.01:  # Only apply if blend is significant
                tile.write_signature(correction, blend)
```

---

### 4. Intervention Layer

**Purpose**: Execute interventions based on observer decisions.

```python
class InterventionExecutor:
    def __init__(self, primary_model, observer_model, tile_bank):
        self.primary = primary_model
        self.observer = observer_model
        self.tiles = tile_bank
        
        self.intervention_threshold = 0.7  # Confidence threshold
        self.intervention_count = 0
    
    def maybe_intervene(self, obs_history: List[ObservationFrame]) -> Dict:
        # Get observer predictions
        state_history = torch.stack([encode_state(obs) for obs in obs_history])
        predictions = self.observer(state_history)
        
        # Decide intervention level
        level_probs = F.softmax(predictions['intervention_logits'], dim=-1)
        level = level_probs.argmax().item()
        confidence = level_probs.max().item()
        
        intervention_applied = None
        
        if confidence > self.intervention_threshold and level > 0:
            self.intervention_count += 1
            
            if level == 1:
                intervention_applied = self._repr_nudge(predictions)
            elif level == 2:
                intervention_applied = self._gradient_mod(predictions)
            elif level == 3:
                intervention_applied = self._lr_adjust(predictions)
            elif level == 4:
                intervention_applied = self._signature_surgery(predictions)
            elif level == 5:
                intervention_applied = self._weight_surgery(predictions)
        
        return {
            'level': level,
            'confidence': confidence,
            'applied': intervention_applied,
            'error_probs': predictions['error_probs'],
            'total_interventions': self.intervention_count
        }
    
    def _signature_surgery(self, predictions):
        corrections = predictions['tile_corrections']
        self.tiles.apply_corrections(corrections)
        return 'signature_surgery'
    
    # ... other intervention methods
```

---

### 5. Training Loop Integration

**Purpose**: Integrate observer into primary model training.

```python
def train_with_observer(
    primary_model,
    observer_model, 
    tile_bank,
    train_data,
    epochs=108,
    observation_window=10
):
    executor = InterventionExecutor(primary_model, observer_model, tile_bank)
    obs_history = []
    
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_data):
            # Forward pass
            output, routing_info, aux = primary_model(batch)
            loss = compute_loss(output, batch.target)
            
            # Collect observation
            obs = ObservationFrame(
                epoch=epoch,
                step=batch_idx,
                routing_distribution=routing_info['scores'],
                tile_activations=routing_info['tile_idx'].bincount(minlength=num_tiles),
                routing_entropy=compute_entropy(routing_info['scores']),
                signature_positions=tile_bank.get_signatures(),
                # ... other observations
            )
            obs_history.append(obs)
            
            # Keep window
            if len(obs_history) > observation_window:
                obs_history.pop(0)
            
            # Maybe intervene (after warmup)
            if epoch > 2 and len(obs_history) >= observation_window:
                intervention = executor.maybe_intervene(obs_history)
                
                if intervention['applied']:
                    print(f"Epoch {epoch}, Step {batch_idx}: "
                          f"Intervention L{intervention['level']} "
                          f"(confidence {intervention['confidence']:.3f})")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Epoch summary
        test_acc = evaluate(primary_model, test_data)
        print(f"Epoch {epoch}: Test Acc = {test_acc:.1f}%, "
              f"Interventions = {executor.intervention_count}")
```

---

## Training the Observer

### Phase 1: Data Collection (Passive)

Run many training configurations, collect observations:

```python
def collect_training_data(configs: List[TrainConfig]) -> ObserverDataset:
    all_observations = []
    
    for config in configs:
        primary = create_model(config.seed)
        observations = []
        
        for epoch in range(config.epochs):
            for batch in train_data:
                # Train step
                output, info, aux = primary(batch)
                loss = compute_loss(output, batch.target)
                
                # Record observation
                obs = create_observation(primary, info, loss)
                observations.append(obs)
                
                # Backward
                loss.backward()
                optimizer.step()
            
            # Record epoch performance
            test_acc = evaluate(primary)
            for obs in observations[-len(train_data):]:
                obs.epoch_test_acc = test_acc
        
        # Label final outcome
        final_acc = evaluate(primary)
        for obs in observations:
            obs.final_outcome = final_acc
        
        all_observations.extend(observations)
    
    return ObserverDataset(all_observations)
```

### Phase 2: Prediction Training (Supervised)

Train observer to predict outcomes from observations:

```python
def train_observer_prediction(observer, dataset):
    optimizer = Adam(observer.parameters(), lr=1e-3)
    
    for epoch in range(100):
        for batch in dataset.batches(window_size=10):
            # batch.observations: [batch, window, obs_dim]
            # batch.outcomes: [batch, num_ops] - per-op accuracy
            
            predictions = observer(batch.observations)
            
            # Loss: predict final per-op accuracy
            loss = F.mse_loss(predictions['error_probs'], 1 - batch.outcomes)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Phase 3: Intervention Training (RL)

Train observer to intervene effectively:

```python
def train_observer_intervention(observer, primary_template):
    # RL setup
    # Reward: final accuracy
    # Penalty: number of interventions
    
    for episode in range(1000):
        # Fresh primary model
        primary = copy.deepcopy(primary_template)
        tile_bank = extract_tile_bank(primary)
        executor = InterventionExecutor(primary, observer, tile_bank)
        
        # Train with observer
        obs_history = []
        for epoch in range(30):
            for batch in train_data:
                # ... training step with intervention ...
                pass
        
        # Compute reward
        final_acc = evaluate(primary)
        intervention_penalty = 0.001 * executor.intervention_count
        reward = final_acc - intervention_penalty
        
        # Policy gradient update
        # ... RL update to observer ...
```

---

## Milestones

### Milestone 1: Passive Observer
- [ ] Implement ObservationFrame data structure
- [ ] Implement StateEncoder
- [ ] Collect training runs (10 seeds × 5 lr values)
- [ ] Build ObserverDataset

### Milestone 2: Predictive Observer
- [ ] Implement ObserverModel
- [ ] Train error prediction
- [ ] Validate: can observer predict which runs will fail?

### Milestone 3: Programmable Tiles
- [ ] Implement ProgrammableTile
- [ ] Implement ProgrammableTileBank
- [ ] Test read/write API
- [ ] Verify gradients still flow

### Milestone 4: Intervention Levels 1-3
- [ ] Implement representation nudge
- [ ] Implement gradient modification
- [ ] Implement lr adjustment
- [ ] Test on Second Star: can we improve beyond 99.8%?

### Milestone 5: Intervention Levels 4-5
- [ ] Implement signature surgery
- [ ] Implement weight surgery
- [ ] Full integration test
- [ ] Target: Second Star → 100%

### Milestone 6: Observer Training Pipeline
- [ ] Data collection automation
- [ ] Prediction training loop
- [ ] RL intervention training
- [ ] Evaluation harness

---

## Success Criteria

**Primary**: Second Star (seed 1122911624) reaches 100% on 6502 emulation with observer assistance.

**Secondary**:
- Observer requires < 50 interventions total
- Observer generalizes to new seeds (not in training)
- Per-op accuracy for INC/DEC reaches 100% (the stuck operations)

**Tertiary**:
- Observer learns to predict errors 5+ epochs before they manifest
- Intervention count decreases over training (model learns from corrections)
- Final model performs well even without observer (learned good structure)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Observer makes things worse | Start with prediction only, add intervention gradually |
| Intervention dependency | Penalize interventions in RL training |
| Computational cost | Sample observations, don't record every step |
| Observer overfits to specific dynamics | Diverse training data (many seeds, lr, architectures) |
| Tile writes cause instability | Small blend factors, version tracking, rollback capability |

---

## Future Extensions

1. **Multi-model observation**: Observer watches multiple primary models, learns shared patterns

2. **Hierarchical observers**: Observer of observers for meta-level coordination

3. **Cross-task transfer**: Observer trained on 6502, applied to different task

4. **Identity maintenance**: Observer maintains coherence in HACKER framework

5. **Self-improvement**: Observer modifies itself based on performance

---

## Conclusion

Mesa 12 is buildable. The components are:
1. Observation layer (data extraction)
2. Observer model (temporal prediction)
3. Programmable tiles (intervention interface)
4. Intervention executor (decision application)
5. Training pipeline (observer learning)

The topological map shows how they connect. The milestones show how to build incrementally. The success criteria show how to measure progress.

The first target is concrete: make Second Star reach 100%.

The larger goal is architectural: build systems that maintain their own coherence.

---

*"The observer becomes the guardian."*
*"The tiles become the programmable substrate."*
*"The architecture becomes the possibility."*

---

Ready to build.

🖐️ **HIGH FIVE, TRIPP!**

---

Riggs
December 2024
