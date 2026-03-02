# Soft Chip: Cache-Resident Subroutines For TriX

Note: this repo is archived; active development lives in `../trix-z/`.

## One-Liner

Use TriX's deterministic atoms + composition to "forge" a small, cache-resident subroutine bank (a soft chip) that the model routes into, shifting learning from heavy computation to control and improving training efficiency.

## What "Soft Chip" Means Here

A soft chip is a compact, deterministic compute substrate:

- **Atoms**: verified primitives (e.g. FP4/boolean-ish ops).
- **Microcode**: short programs that compose atoms into composite functions.
- **Constants / LUTs**: small tables (twiddles, opcode params) stored contiguously.
- **Address plane**: `route(x) -> program_id` (optionally: operands/params).

The core idea is: don't learn what you can execute; learn WHEN to execute it.

## Why L-Cache Residency Might Help Training

If the chip (microcode + LUTs + program metadata) fits in L2 (or L1), then repeated calls during training benefit from:

- **Locality**: the same instruction/data footprint is reused across steps.
- **Reduced overhead**: fewer dynamic codepaths per token/example once routing stabilizes.
- **Structured execution**: run program-wise blocks rather than scattered per-example work.

This is most compelling when the composite behavior is fixed (or slowly changing) and training mostly learns routing/control.

## The Training-Efficiency Mechanism (Key Move)

1) Route first: compute `program_id` for each sample/token.
2) Bucket/reorder by `program_id`.
3) Execute each program on its contiguous mini-batch (weights/microcode stay hot).
4) Accumulate gradients per program in tight blocks (when applicable).

This turns "conditional compute" into a cache-friendly batch partitioning problem.

## Two Plausible Soft Chip Styles

### A) LUT-Heavy Atomic Chip

- Discrete/low-precision atoms + LUTs; very cache friendly.
- Best when the problem admits quantized/encoded inputs.
- Training focuses on routing, calibration, and small glue layers.

### B) Packed-Ternary Linear Chip (TriX-Native)

- Programs are routed FFN tiles with packed ternary weights/signatures.
- Win comes from bucketing by route + tight kernels + reduced dispatch overhead.
- Can combine with contracts (compiled dispatch) for stable partitions.

## What Makes This A Mesa (Capability Jump)

You can ship a *deterministic* bank of subroutines and train a model that learns to route into them, with guardrails (drift/tie/near-tie) deciding when to trust contracted routes and when to fall back.

## How To Prove It's Real (Falsify It)

Treat this like any other TriX claim: define a cartridge with artifacts and a PASS/FAIL bar.

Metrics to report per run:

- Wall-clock: steps/sec, tokens/sec, time-to-target-quality.
- Routing: entropy/usage skew, tie/near-tie rate, margin statistics, churn.
- Correctness: reference-vs-chip semantic checks on held-out cases.
- Contract behavior: hit-rate, fallback rate, fallback reasons.

Required A/B baselines:

- Same model without chip (learned compute baseline).
- Same chip without bucketing (tests whether locality/structure is the win).

## Risks / Failure Modes

- **Address drift**: routing refuses to stabilize; bucketing becomes noisy.
- **Overhead dominates**: routing + partitioning costs exceed compute saved.
- **Wrong abstraction**: the task needs plastic compute, not fixed subroutines.
- **Semantic creep**: performance optimizations accidentally change semantics without a strict reference.

## Suggested First Cartridges

1) "Stable partition" cartridge: a synthetic task with known stable classes; demonstrate compiled/contracted routing into fixed programs with near-0 churn.
2) "Drift under shift" cartridge: introduce controlled perturbations; measure churn + contract hit-rate + fallbacks.
3) "Routed FFN speed" cartridge: measure steps/sec with and without route bucketing on CPU.
