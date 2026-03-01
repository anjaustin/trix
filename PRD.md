# PRD: Routing As A Primitive (TriX Continuation)

Owner: (you)
Last updated: 2026-03-01
Status: Draft

## 0. One-Liner

Make "routing" a first-class, testable, deployable compute primitive: observable during training, editable when needed, compilable into deterministic contracts for known cases, guarded with fallbacks, and accelerated without changing semantics.

## 1. Problem

Conditional compute systems (tiles/experts/memory) often fail in practice because routing is:
- hard to observe (no stable introspection),
- hard to control (no safe manual edits),
- hard to freeze (deployment drift),
- hard to trust (no crisp invariants),
- hard to reproduce (benchmarks and claims not tied to a single runnable harness),
- fragile to environment (native acceleration and optional deps break baseline expectations).

TriX already contains promising building blocks (e.g. `SparseLookupFFNv2`, "surgery", `CompiledDispatch`, kernel packing, compiler/FP4 lane), but needs a coherent product-quality research arc.

## 2. Target Users

- Research engineers iterating on conditional compute architectures who need trustworthy instrumentation and repeatable results.
- Systems-minded ML engineers who want routing behavior that can be audited and stabilized for deployment.
- Contributors exploring "compute-as-routing" and circuit-like semantics (compiler lane).

## 3. Goals (What Success Looks Like)

G1) Addressable routing: define what an "address" is and enforce invariants around it.
- "Address" can mean: tile id, signature id, compiled class id, or circuit atom id.

G2) Routing lifecycle is real and safe:
- observe -> edit -> compile -> guard -> monitor

G3) Reproducible evidence:
- 2-3 canonical benchmarks/demos that can be run by a new reader and produce stable, expected outcomes.

G4) Correctness is independent from acceleration:
- native kernels accelerate; reference paths define semantics.

G5) Repo hygiene:
- clear separation of core vs native vs experiments; tests and installation succeed on supported platforms.

## 4. Non-Goals

- Proving broad mathematical claims (e.g. number theory claims) as part of the core library.
- Building a full training framework; we focus on routing primitives + instrumentation + benchmarks.
- Maximizing raw throughput at the expense of semantic clarity.

## 5. Principles

- Semantics first: a correct, portable reference implementation exists for every accelerated path.
- Invariants over vibes: novel claims must be tied to tests, metrics, and a runnable harness.
- Guarded determinism: compile/freeze where needed; fall back gracefully elsewhere.
- Make addresses meaningful: stability and interpretability are features, not accidents.

## 6. Product Surface (Proposed)

### 6.1 Address Types

Define and standardize:
- Tile address: integer tile id.
- Signature address: stable identifier for a signature representation (e.g. packed ternary / XOR compressed).
- Class address: stable class id used by compiled dispatch.
- Atom address: compiler lane atom id (AND/XOR/etc.).

### 6.2 Routing Contract

The routing system must provide:
- `route(x) -> addr` (or top-k addresses)
- `explain(addr)` (introspection / stats)
- `edit(addr, ...)` (surgery primitives)
- `compile(address_set, ...) -> contract`
- `guard(contract, ...)` (drift detection + fallback)

### 6.3 Minimal Routing Primitive (Reference Kernel)

Baseline (hard routing) to treat as the conceptual primitive:

```python
signature = tile_weights.sum(dim=0).sign()  # What the tile wants
score = input @ signature                   # How well input matches
route = (score == scores.max())             # Send to best match
```

Notes:
- This is a piecewise-partitioning of input space by dot-product similarity to tile signatures.
- The rest of the system work is making this primitive stable and operable at scale (normalization, tie-breaking/top-k, load balancing, differentiable surrogates during training, and compile/guard for deployment).

## 7. Workstreams (Maps to 1..6)

### WS1: Address Contract + Invariants ("Address is a contract")

Deliverables:
- A written spec: what is an address, what must remain stable, what is allowed to drift.
- A metrics suite:
  - stability across seeds (agreement %)
  - drift under continued training (KL/JS of routing distribution, addr churn)
  - collapse metrics (entropy, usage skew/Gini)
  - distribution shift sensitivity (addr churn vs held-out shift set)
- CI tests that enforce minimum standards for stability/collapse on small synthetic tasks.

Requirements:
- R1: Provide a stable, versioned schema for routing logs (JSONL or similar).
- R2: Every routing module exposes a common minimal telemetry interface.

### WS2: Routing Lifecycle Core (observe -> edit -> compile -> guard -> monitor)

Deliverables:
- A single "canonical" lifecycle API (even if internally it wraps existing modules).
- A "surgery" safety model:
  - idempotent edits
  - reversible edits (or explicit non-reversible operations)
  - explicit pre/post validation
- Guardrails:
  - confidence thresholds
  - fallback policies
  - contract invalidation rules

Requirements:
- R3: Compiled contracts must be reproducible given fixed seeds and identical weights.
- R4: Contract usage must be observable at runtime (hit/miss, fallback reason).

### WS3: Canonical Compute-As-Routing Benchmarks

Deliverables:
- 2-3 benchmark "cartridges" with:
  - fixed seeds
  - expected outputs
  - metrics produced as artifacts
  - CPU-only baseline success

Suggested benchmark types (pick 2-3):
- Algorithmic circuits: adders/mux/alu-like truth tables (compiler lane overlap).
- Multi-skill toy domain: clear subroutines, measurable specialization.
- Structured classification with explicit concept partitions.

Requirements:
- R5: Each benchmark runs in < 5 minutes on CPU for a small config.
- R6: Benchmarks report: quality, cost (active tiles), stability, and determinism.

### WS4: Representation + Compression With Explicit Guarantees

Deliverables:
- For XOR signature compression (and any other compression):
  - formal statement of what is preserved (e.g. distance ordering constraints)
  - adversarial cases and where it fails
  - tests that assert the preserved properties

Requirements:
- R7: Compression must be optional and swap-in; routing semantics remain equivalent under stated conditions.

### WS5: Native Acceleration As An Implementation Detail

Deliverables:
- A "reference correctness" suite comparing:
  - Python reference
  - PyTorch vectorized fallback
  - native kernel (when present)
- Cross-platform library discovery and graceful fallback.

Requirements:
- R8: CI must pass without native builds; native builds add speed, not correctness.
- R9: Native-vs-reference mismatch must fail tests with clear diagnostics.

### WS6: Repo Structure + Communication Hygiene

Deliverables:
- Clear tiers:
  - `core/` (or existing `src/trix/` core modules): dependency-light, always tested
  - `native/` (or `src/trix/kernel/`): optional acceleration
  - `experiments/`: heavy deps and speculative narratives
- Installation extras:
  - `trix[dev]` already exists
  - add `trix[experiments]` or `trix[number-theory]` if needed
- Docs that separate:
  - guarantees (backed by tests)
  - benchmarks (repro scripts)
  - speculation (clearly labeled)

Requirements:
- R10: A new reader can run `pip install -e ".[dev]" && pytest` on supported platforms and get green.

## 8. Milestones

M0: Baseline hygiene (done/ongoing)
- versions aligned; tests pass on macOS with fallbacks.

M1: Address contract v1
- spec + telemetry schema + stability/collapse metrics + CI tests on small synthetic routing task.

M2: Lifecycle API v1
- observe/edit/compile/guard/monitor path integrated for at least one routing module (recommend: `SparseLookupFFNv2`).

M3: Benchmark suite v1
- two canonical benchmarks + expected outputs + artifact reports.

M4: Compression guarantees
- XOR compression spec + property tests + failure mode docs.

M5: Native correctness harness
- consistent reference-vs-native comparison tests and failure diagnostics.

M6 (Mesa 14): Address Space Mesa
- Ship a durable, portable address artifact bundle (compressed signatures + dispatch contract + metadata + validation report).
- Normalize telemetry across routing backends (backend name, tie/near-tie, margins, compression stats snapshot, contract hit/miss/fallback).
- Make contract lifecycle explicit: validate -> ship -> monitor -> invalidate/rebuild.
- Promote drift-under-optimization to a first-class benchmark artifact (churn curve + contract hit-rate curve).

## 9. Metrics

Quality:
- task metric (accuracy / exact match / loss) per benchmark.

Cost:
- mean active tiles per token
- wall-clock latency (reference and native when available)

Stability:
- address agreement across seeds
- address churn across training steps
- contract hit rate and fallback rate

Trust:
- % of claims in README/CHANGELOG with a runnable script producing an expected artifact.

## 10. Risks

- Address drift is fundamental: if addresses cannot be stabilized without killing learning, the thesis fails.
- Overfitting benchmarks: success on toy tasks may not generalize.
- Complexity creep: too many routing variants; focus may fragment.
- Native kernel divergence: performance work risks semantic mismatch without a strict harness.

## 11. Open Questions

- What is the canonical address space: tile ids, signatures, or compiled classes (or a layered composition)?
- Where should the lifecycle API live (new module vs adapting existing ones)?
- Which two benchmarks best demonstrate "compute-as-routing" without requiring large-scale training?
