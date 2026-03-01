# Address Contract (v1)

This document defines what an "address" means in TriX-style routing systems and the invariants we expect from it.

Status: draft
Last updated: 2026-03-01

## 1. Definitions

An **address** is an identifier that selects a member of a fixed family of computations.

The minimal routing form is:

1) compute an address: `addr = route(x, state)`
2) dispatch compute: `y = f_addr(x, state)`

TriX currently uses several address-like notions. For v1, we standardize the following address types.

### Address Types

- `tile_id` (int)
  - Selects one tile/expert module from a bank of `num_tiles`.

- `signature_id` (string/int)
  - Selects a particular signature representation (e.g. raw ternary signature vs compressed signature).
  - This is a *representation address*, not a compute address.

- `class_id` (int)
  - Selects a compiled dispatch path for a known class/hint (e.g. `CompiledDispatch`).

- `atom_id` (string)
  - Selects a verified circuit atom (compiler lane), e.g. `XOR`, `SUM`, `CARRY`.

## 2. Invariants

These invariants are written as engineering properties that can be tested.

### I1: Validity

- Every produced `tile_id` must satisfy: `0 <= tile_id < num_tiles`.

### I2: Determinism (given fixed inputs)

Given fixed inputs, fixed weights/signatures, and fixed routing configuration:

- routing is deterministic (bit-exact) in evaluation mode.

### I3: Distribution Health (no collapse)

Routing should not collapse to a single address absent an explicit reason.

Observable proxies (v1):
- usage entropy is not near-zero
- usage skew/Gini is not near-one
- `max_count / total` is bounded

### I4: Stability Under Small Perturbations

Under small perturbations to signatures (or small weight drift), routing should not churn arbitrarily.

Observable proxy (v1):
- churn rate between routes at `t` and `t+Δ` is bounded for a small, controlled perturbation.

### I5: Contractability

There exists a subset of addresses (e.g. stable classes) that can be compiled into a deterministic dispatch contract with explicit guardrails and a fallback path.

## 3. Telemetry Schema (JSONL)

We log routing and stability events in JSONL (one JSON object per line).

Required fields (v1):
- `schema_version` (int) : currently `1`
- `event` (string) : `routing` | `stability`
- `run_id` (string) : stable id for this run
- `seed` (int)
- `address_type` (string) : currently `tile_id`
- `tiles` (int)
- `dim` (int)
- `inputs` (int)

Routing fields:
- `routing_ms` (number)
- `routes_per_s` (number)
- `entropy_nats` (number)
- `gini` (number)
- `max_tile` (int)
- `max_count` (int)

Stability fields:
- `flip_prob` (number)
- `churn` (number)

Notes:
- v1 telemetry is intentionally minimal: it is meant to be lightweight and dependency-free.

## 4. Test Harness

The native C++ harness in `native/` provides:
- a routing microbenchmark (distribution health metrics)
- a stability benchmark (churn under controlled perturbations)
- a unit/invariant test suite (`ctest`)
