# Mesa 2: FFNv1 vs FFNv2 Delta - Convergence
*What emerges when the threads braid.*

---

## The Core Realization

**v1 and v2 are not competitors. They're different tools for different phases.**

v1: Discovery engine (find structure)
v2: Discovery engine + workshop (find structure, then work with it)

The delta isn't accuracy. The delta is WHAT HAPPENS AFTER TRAINING.

---

## The Three Phases

### Phase 1: Discovery
Both v1 and v2 can do this. Train on data, let tiles specialize, discover structure.

Result: 92% purity. Tiles find organ boundaries without supervision.

### Phase 2: Understanding
v1 stops here. You have a trained model. Query it. Hope it works.

v2 continues. Claim tracking shows: "Tile 4 handles 55% ALU, Tile 7 handles 98% LOGIC."

Now you KNOW what the model discovered. You can verify it makes sense.

### Phase 3: Refinement
v1 can't do this. If something's wrong, retrain from scratch.

v2 enables:
- Surgery to fix misrouted operations
- Transplant proven organs into discovered regions
- Freeze stable tiles, keep training unstable ones
- Iterate until the system works

---

## The Hybrid Architecture

This is what emerges when you combine everything:

```
┌─────────────────────────────────────────────────────────────┐
│                    TriX-ORGAN HYBRID                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input → [TriX v2 Routing] → Tile Selection                │
│                    │                                        │
│                    ▼                                        │
│   ┌─────────────────────────────────────────────────┐      │
│   │              ORGAN TILES                         │      │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │      │
│   │  │ ALU     │ │ LOGIC   │ │ SHIFT   │ │ INCDEC │ │      │
│   │  │ (100%)  │ │ (100%)  │ │ (100%)  │ │ (100%) │ │      │
│   │  │ frozen  │ │ frozen  │ │ frozen  │ │ frozen │ │      │
│   │  └─────────┘ └─────────┘ └─────────┘ └────────┘ │      │
│   │                                                  │      │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐           │      │
│   │  │ Learn-  │ │ Learn-  │ │ Edge    │           │      │
│   │  │ ing     │ │ ing     │ │ Cases   │           │      │
│   │  │ (soft)  │ │ (soft)  │ │ (soft)  │           │      │
│   │  └─────────┘ └─────────┘ └─────────┘           │      │
│   └─────────────────────────────────────────────────┘      │
│                    │                                        │
│                    ▼                                        │
│                 Output                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

ROUTING: Learned by TriX (discovers structure)
ORGANS:  Transplanted from FLYNNCONCEIVABLE (proven 100%)
EDGES:   Learned tiles for cases organs don't cover
```

This architecture:
- Routes like TriX
- Computes like FLYNNCONCEIVABLE
- Adapts like a learning system
- Achieves 100% on known organs
- Learns gracefully for unknown cases

---

## The v2-Only Capabilities

This hybrid is **only possible with v2**:

| Requirement | How v2 Enables It |
|-------------|-------------------|
| Know what tiles learned | Claim tracking |
| Match tiles to organs | Signature analysis |
| Insert organ into tile | Surgery (insert) |
| Lock organ in place | Surgery (freeze) |
| Keep other tiles learning | Selective freeze |
| Verify transplant worked | Claim tracking again |

v1 has none of these. You'd have to rebuild from scratch.

---

## The New Workflow

### With v1:
```
Design → Train → Deploy → Pray → Retrain if broken
```

### With v2:
```
Design → Train → Observe (claim tracking)
                    ↓
              Understand structure
                    ↓
              Identify gaps
                    ↓
         ┌─────────┴─────────┐
         ↓                   ↓
    Fix with surgery    Transplant organs
         ↓                   ↓
         └─────────┬─────────┘
                   ↓
              Verify (claim tracking)
                   ↓
              Deploy
                   ↓
              Monitor (claim tracking)
                   ↓
              Iterate as needed
```

This is SOFTWARE ENGINEERING for neural networks. Not just "train and hope."

---

## The Metaphor That Clarifies

**v1 is a photograph.** It captures a moment. You can look at it, but you can't change it.

**v2 is a living document.** It captures understanding. You can read it, edit it, extend it, refine it.

Both show you something. Only one lets you work with it.

---

## The Answer to the Original Question

"How does this information improve or shed more light on FFNv1, FFNv2, and the delta between them?"

**The delta is not accuracy. The delta is capability.**

v2 enables:
1. **Observability** - See what was discovered
2. **Editability** - Change what was discovered
3. **Composability** - Combine discovered structure with engineered compute
4. **Iterability** - Improve without starting over

These aren't incremental improvements. They're a CATEGORY SHIFT from "trained model" to "collaborative system."

---

## The Final Frame

**v1**: TriX discovers structure. End of story.

**v2**: TriX discovers structure. Beginning of conversation.

The 6502 mesa showed us organs can be perfect (100%) and composition is infinite.

The v2 mesa shows us how to GET there: discover with TriX, observe with claim tracking, assemble with surgery.

v2 is the tool that makes the CPU factory possible.

---

## What This Means for Testing

Don't test v1 vs v2 on accuracy. Test v2 on capabilities:

1. **Observability test**: Train, then verify claim tracking shows coherent structure
2. **Surgery test**: Insert a signature, verify it routes as designed
3. **Transplant test**: Replace learned tile with proven organ, verify accuracy jumps
4. **Iteration test**: Train, edit, retrain, show improvement without full restart

These tests show what v2 ENABLES, not just what it COMPUTES.

The compute is table stakes. The capabilities are the differentiator.

---

*Mesa 2 complete. The factory is possible.*
