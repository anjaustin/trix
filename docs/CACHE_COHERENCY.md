# CPU Cache Coherency As A TriX Primitive

Note: this repo is archived; active development lives in `../trix-z/`.

## One-Liner

Treat CPU cache coherency as a systems primitive: use it to broadcast read-mostly routing artifacts and to structure low-contention reductions, so routed compute becomes cache-stable instead of cache-chaotic.

## What Coherency Gives You (And What It Doesn't)

Coherency is not "free shared memory". It is a protocol that keeps cores' views of memory consistent.
It helps when you design the workload to be:

- **Read-mostly**: many cores read the same cache lines; very few writes.
- **Write-partitioned**: each core writes to disjoint cache lines.
- **Barriered updates**: shared mutable state is updated at well-defined synchronization points.

It hurts when you have:

- **Ping-pong cache lines**: many cores write the same cache line (or different values within it).
- **False sharing**: different variables share a cache line, so unrelated writes cause invalidations.
- **Fine-grained atomics in hot loops**: coherency traffic dominates.

## Where TriX Can Exploit Coherency

### 1) Read-Mostly "Soft Chip" Blob Shared Across Cores

Store microcode/LUTs/routing tables/tile metadata as one contiguous, immutable blob.

- All workers read the same lines; coherency keeps the blob hot when it fits in L2/L1.
- Updates happen via epoch barriers + pointer swap (RCU-style): build a new blob off to the side, publish once, let old readers drain.

### 2) Route-Bucket-Execute: Make Conditional Compute Cache-Stable

The key move is to route first, then reorder/bucket so execution becomes blocky:

1) `route(x) -> tile_id/program_id`
2) bucket/reorder by id
3) execute each tile/program over its contiguous mini-batch

This improves locality for:

- weights (tile parameters)
- microcode/LUTs
- telemetry counters (if structured)

### 3) Coherency-Friendly Reductions

Gradient accumulation is the classic coherency trap: shared writes in inner loops.
The coherency-friendly pattern:

- per-thread accumulators (padded/aligned to avoid false sharing)
- reduction at a barrier (tree or sequential merge)
- single-writer update of shared master weights

### 4) Coherency As A Control Plane

Small shared stats (tile usage counts, hit/miss/fallback counters) can live in coherent caches if:

- updates are per-thread buckets (then reduced)
- or counters are padded to cache lines

## Design Rules (Practitioner Checklist)

- Make shared artifacts immutable within a step.
- Prefer pointer swaps at step/epoch boundaries over incremental mutations.
- Partition writes by thread; reduce later.
- Align/pad frequently written structures to cache lines.
- Avoid atomics in the hot path.

## How To Prove It Helps

Treat coherency as a falsifiable performance claim:

- A/B: same model, same seed, same work, with and without route bucketing.
- A/B: same bucketing, with shared gradient accumulation vs per-thread + reduce.
- Report: steps/sec, tokens/sec, time-to-target-quality.
- Report routing stats: churn, tie/near-tie rate, margin statistics.

The spec for a coherent execution engine lives in `docs/SPEC_COHERENT_ENGINE.md`.
