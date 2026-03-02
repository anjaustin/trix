# Spec: Coherent Routed Execution Engine (CPU)

Note: this repo is archived; active development lives in `../trix-z/`.

## Goal

Design a CPU execution path for routed compute that:

- increases cache locality via route bucketing,
- avoids coherency ping-pong via write partitioning,
- supports contracted/compiled dispatch with explicit fallbacks,
- keeps semantics unchanged relative to the reference execution.

This is a spec for implementation in `../trix-z/`.

## Non-Goals

- GPU kernels.
- Changing routing semantics.
- Magical speedups without falsification.

## High-Level Architecture

Inputs per step:

- activations `X` (tokens/examples)
- routing module `route(X) -> ids` (tile_id/program_id)
- per-id compute payload (tile weights, microcode params)
- optional contract table (compiled dispatch)

Outputs per step:

- activations `Y`
- routing telemetry + drift telemetry

### Core Phases

1) Route
- Compute ids (dynamic) OR use contract when guard passes.

2) Bucket
- Stable partition inputs by id into contiguous buckets.

3) Execute
- For each id, run its program/tile over the bucket.

4) Reduce (training)
- Accumulate per-thread gradients, then merge at barrier.

## Data Model

### Immutable "Chip Blob"

A compact, read-only struct published once per step/epoch:

- program metadata table (offsets, shapes)
- microcode / LUT bytes
- packed weights/signatures (if used)
- contract table (if used)

Update mechanism:

- build next blob off-thread
- publish pointer swap at barrier
- reclaim old blobs when no worker holds a reference (RCU)

### Buckets

Represent bucketing as two arrays:

- `perm`: indices of inputs, grouped by id
- `ranges`: for each id, [start, end) in `perm`

Constraints:

- deterministic given ids + tie-break policy
- stable ordering within an id bucket (optional, but aids determinism)

## Threading Model

### Scheduling

Preferred scheduling is per-id or per-range chunks:

- each worker processes one or more bucket ranges
- avoid splitting a single range across many workers when the per-id payload is small (overhead)

### Writes

Training requires gradients. Avoid shared writes in hot loops:

- allocate per-thread gradient buffers for each tile/program parameter shard
- align/pad buffers to cache-line boundaries

Reduction:

- barrier
- merge per-thread buffers into the master weights (single-writer or tree reduce)

## Contract + Guard Integration

Contracted dispatch path:

- inputs with a class hint (or other stable hint) attempt contract lookup
- guard checks confidence / drift policy
- on pass: id is fixed from contract
- on fail: fall back to dynamic routing

Telemetry requirements:

- record per-call: compiled hit/miss, fallback reason, near-tie/margin stats

## Correctness Contract

The coherent engine must match the reference semantics under fixed seeds.
Where exact equality is unrealistic (floating point order changes), define acceptable tolerances and test them.

Falsification tests:

- same ids, compare bucketed vs non-bucketed output
- compare contracted vs dynamic for declared stable partitions

## Benchmark Cartridges

Cartridge 1: Routed FFN speed (CPU)

- A/B: route bucketing off vs on
- A/B: shared gradient accumulation vs per-thread + reduce
- report: steps/sec, time/step breakdown (route, bucket, execute, reduce)

Cartridge 2: Drift under continued training

- monitor churn + contract hit rate while weights update
- confirm guardrails produce explainable fallbacks

Cartridge 3: Stability under shift

- perturb inputs or signatures
- measure near-tie rate, margin shifts, churn

## Implementation Notes (CPU Pitfalls)

- False sharing: pad counters and per-thread buffers.
- Atomics: keep them out of the hot path.
- Memory allocation: preallocate perm/range buffers; reuse.
- Determinism: make tie-break and bucket ordering explicit.
