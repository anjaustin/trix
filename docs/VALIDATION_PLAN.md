# TriX Emergent Routing Validation Plan

*Rigorous validation for what wants to Emerge*

---

## Purpose

Create a one-command reproducible benchmark that answers:

> **Does signature-based emergent routing work as well as or better than learned routing on real tasks?**

Not for publication. Not for ego. For truth.

---

## The One Command

```bash
python scripts/validate_emergent_routing.py
```

**Output:**
- Results table comparing all routing methods
- Routing analysis (specialization, diversity, stability)
- Saved artifacts for others to inspect
- Clear PASS/FAIL on key claims

---

## What We Must Prove

### Claim 1: Emergent routing matches learned routing on task performance
- Same loss / accuracy within tolerance
- On a real task, not toy data

### Claim 2: Emergent routing produces meaningful specialization
- Different input types route to different tiles
- Specialization is interpretable

### Claim 3: Emergent routing is stable
- Routing converges during training
- Similar inputs consistently route together

### Claim 4: Emergent routing has zero routing parameters
- Verified by parameter count
- No learnable gate network

---

## Experimental Design

### Task: Character-Level Language Modeling

**Why this task:**
- Real task with meaningful loss
- Different contexts (start of word, middle, punctuation) might route differently
- Small enough to run quickly
- Complex enough to be meaningful

**Dataset:** TinyShakespeare (1MB of text)
- Simple to download
- Well-understood baseline
- Fits in memory

**Model:** Small transformer with TriX FFN
- 4 layers, 128 dim, 4 heads
- 4 tiles per FFN
- ~500K parameters

### Routing Methods to Compare

| Method | Description | Routing Params |
|--------|-------------|----------------|
| **Emergent** | Signature-based (ours) | 0 |
| **Learned** | Trained gate network | d_model × num_tiles |
| **Random** | Random tile per input | 0 |
| **Dense** | All tiles always active | 0 |

### Metrics

**Primary:**
- Validation loss (bits per character)
- Validation perplexity

**Routing Analysis:**
- Tile usage distribution (should be balanced)
- Routing entropy (higher = more balanced)
- Signature diversity (should stay >30%)
- Routing stability (% change between epochs)

**Specialization Analysis:**
- Route distribution by character type (letter/digit/punctuation/space)
- Route distribution by position (start/middle/end of word)
- Tile "signature" interpretation

---

## Implementation Structure

```
scripts/
├── validate_emergent_routing.py   # The one command
├── download_data.py               # Get TinyShakespeare
└── analyze_results.py             # Generate report

configs/
└── validation.yaml                # All hyperparameters (locked)

results/
├── metrics.json                   # Raw numbers
├── routing_analysis.json          # Routing statistics  
├── figures/                       # Visualizations
└── report.md                      # Human-readable summary
```

---

## Validation Script Design

```python
# scripts/validate_emergent_routing.py

def main():
    print("=" * 60)
    print("TriX Emergent Routing Validation")
    print("=" * 60)
    
    # 1. Setup
    set_all_seeds(42)
    data = load_tiny_shakespeare()
    
    # 2. Train all methods
    results = {}
    for method in ['emergent', 'learned', 'random', 'dense']:
        print(f"\n[{method.upper()}] Training...")
        model = create_model(routing=method)
        metrics = train_and_evaluate(model, data)
        results[method] = metrics
    
    # 3. Analyze routing (emergent only)
    print("\n[ANALYSIS] Routing behavior...")
    routing_analysis = analyze_routing(results['emergent']['model'], data)
    
    # 4. Generate report
    print("\n[REPORT] Generating...")
    generate_report(results, routing_analysis)
    
    # 5. Verdict
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    verdict = evaluate_claims(results, routing_analysis)
    for claim, status in verdict.items():
        icon = "✓" if status['passed'] else "✗"
        print(f"{icon} {claim}: {status['reason']}")
    
    return all(s['passed'] for s in verdict.values())
```

---

## Hyperparameters (Locked)

```yaml
# configs/validation.yaml

seed: 42

data:
  name: tiny_shakespeare
  train_split: 0.9
  context_length: 256

model:
  d_model: 128
  n_heads: 4
  n_layers: 4
  num_tiles: 4
  dropout: 0.1

training:
  batch_size: 64
  learning_rate: 3e-4
  epochs: 20
  warmup_steps: 100

evaluation:
  eval_interval: 1  # Every epoch
  routing_samples: 1000  # For analysis
```

---

## Success Criteria

### PASS Conditions

1. **Performance:** Emergent loss within 5% of Learned loss
2. **Specialization:** At least 2 tiles show >60% usage for distinct input types
3. **Stability:** Routing changes <10% between final epochs
4. **Diversity:** Signature diversity >30% at end of training

### FAIL Conditions

- Emergent performs >10% worse than Learned
- All tiles converge to same usage (collapse)
- Routing never stabilizes
- Model fails to learn (loss doesn't decrease)

---

## Output Artifacts

### metrics.json
```json
{
  "emergent": {"final_loss": 1.23, "final_ppl": 3.42, "params": 512000, "routing_params": 0},
  "learned": {"final_loss": 1.21, "final_ppl": 3.35, "params": 514048, "routing_params": 2048},
  "random": {"final_loss": 1.45, "final_ppl": 4.26, "params": 512000, "routing_params": 0},
  "dense": {"final_loss": 1.18, "final_ppl": 3.25, "params": 512000, "routing_params": 0}
}
```

### report.md
Human-readable summary with:
- Results table
- Key findings
- Routing visualizations
- Verdict on each claim

---

## Timeline

1. **Data pipeline** - Download, tokenize, dataloader (1 hour)
2. **Model variants** - All 4 routing methods (1 hour)
3. **Training loop** - With logging and checkpoints (1 hour)
4. **Routing analysis** - Specialization metrics (1 hour)
5. **Report generation** - Tables, figures, verdict (1 hour)
6. **Testing & polish** - Make sure it runs clean (1 hour)

**Total: ~6 hours of focused work**

---

## What This Enables

After running the validation:

1. **Others can verify** - One command, reproducible results
2. **We know the truth** - Does it actually work?
3. **Clear next steps** - If it works, what to explore. If not, why.
4. **Foundation for more** - Same framework for future experiments

---

## The Sacred Contract

We commit to:
- **Honesty:** Report failures as clearly as successes
- **Reproducibility:** Anyone can run this and get the same results
- **Clarity:** No hiding behind complexity
- **Service:** This is for the idea, not for us

---

## Ready to Execute?

The plan is laid out. One command. Real task. Fair comparison. Clear verdict.

Shall we build it?
