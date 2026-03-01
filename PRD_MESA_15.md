# PRD: Mesa 15 - Address Integrity

Owner: (you)
Last updated: 2026-03-01
Status: Draft

## 0. One-Liner

Make TriX addressable routing artifacts trustworthy to ship: integrity checks, compatibility rules, drift/invalidation policies, and an enforceable runtime contract (ABI + allow/deny policy).

## 1. Why Mesa 15 Exists

Mesa 14 made the address plane real:
- addresses are observable (telemetry),
- portable (bundles),
- and deployable (compiled contracts).

Mesa 15 is about **integrity and governance**:

If addresses are going to be treated like software artifacts, we need:
- a compatibility policy,
- tamper detection,
- reproducibility expectations,
- drift policies,
- and a runtime enforcement layer.

Without this, "addressable intelligence" remains a demo rather than an operational capability.

## 2. Target Users

- ML engineers who want determinism and auditability in conditional compute.
- Teams shipping models who need to detect drift/regressions and rollback safely.
- Safety-minded users who want to restrict internal subroutines that may execute.

## 3. Goals

G1) Bundle integrity: detect tampering and accidental corruption.
G2) Bundle compatibility: define what "loadable" means across versions.
G3) Drift policy: define when a contract is invalid and what to do.
G4) Runtime enforcement: prevent untrusted addresses from executing.
G5) Developer experience: the integrity story is one command away.

## 4. Non-Goals

- Cryptographic identity management / PKI / multi-party signing (v1: integrity, not identity).
- Proving global safety properties of learned systems.
- A fully general policy language (start with allow/deny + quotas + fallbacks).

## 5. Core Concepts

### 5.1 Address Plane

An address plane is the namespace that selects subroutines (tiles/atoms).

In TriX v1 this is primarily:
- `tile_id` addresses derived from signatures and routing.

### 5.2 Bundle

A bundle is firmware-like packaging of address plane artifacts:
- compressed signatures
- optional compiled dispatch contract
- metadata/config
- optional validation
- optional state_dict

See: `docs/BUNDLE_SCHEMA.md` (v1).

### 5.3 Contract

A contract is a compiled mapping from a hint/class/region -> tile(s) with guardrails and fallbacks.

Example: `CompiledDispatch`.

### 5.4 Integrity

Integrity = "what I loaded is exactly what was exported".

### 5.5 Compatibility

Compatibility = "this bundle was produced by a version of TriX that promises a stable meaning for these fields".

### 5.6 Drift

Drift = "the mapping between inputs/classes and addresses changed materially".

Drift is expected; policy determines whether it is tolerated or invalidates a contract.

## 6. Deliverables

### D1) Bundle Manifest + Checksums

Add `manifest.json` to bundles.

Contents:
- `manifest_version` (int)
- `bundle_schema_version` (from `bundle.json.meta.schema_version`)
- `created_at`
- `trix_version`
- `file_hashes`: map filename -> sha256
- `config_fingerprint`: stable hash of key config fields
- `address_plane_fingerprint`: stable hash of compressed signatures export
- `contract_fingerprint`: stable hash of dispatch table (if present)
- `validation_fingerprint`: stable hash of validation report (if present)

Notes:
- sha256 is sufficient for tamper detection.
- This is not identity signing. Later, we can add signatures.

CLI:
- `trix bundle manifest --outdir <bundle>` (generate/refresh)
- `trix bundle verify --outdir <bundle>` (verify hashes + schema)

### D2) Bundle Compatibility Policy

Define and document:

- What changes break compatibility?
  - schema version bump
  - routing backend semantic change
  - dispatch table format change
  - compressed signature encoding change

- What changes are allowed?
  - adding optional fields
  - additional telemetry fields

Implementation:
- `BundleCompatibility` checker that returns:
  - `compatible: bool`
  - `warnings: list[str]`
  - `errors: list[str]`

CLI:
- `trix load-bundle --outdir ... --validate` prints compatibility summary.

### D3) Drift Policies + Invalidation Rules

Define a drift policy schema:

```json
{
  "policy_version": 1,
  "drift_threshold": 0.2,
  "max_churn": 0.1,
  "max_near_tie_rate": 0.3,
  "min_margin_mean": 1e-3,
  "on_violation": "fallback"  // or "fail" or "recompile"
}
```

Drift signals (v1):
- churn over a fixed eval set
- near-tie rate and tie rate
- compiled contract hit rate
- list of drifted classes (`CompiledDispatch.check_drift`)

Invalidation rules:
- Contract invalid if drifted classes exceed threshold
- Contract invalid if churn curve exceeds max
- Contract invalid if near-tie rate spikes (routing ambiguity)

Action on invalidation:
- fallback: run dynamic routing
- fail: throw a runtime error
- recompile: rebuild dispatch and emit new bundle/manifest

### D4) Address ABI (Minimal)

Define a minimal ABI for tiles/subroutines so we can enforce policies:

- input: `Tensor[B,T,D]`
- output: `Tensor[B,T,D]`
- side outputs (optional): routing_info, aux_losses

ABI constraints:
- output shape must match input shape for drop-in mode
- deterministic under eval mode (given fixed weights)

Why:
- allows tooling to treat tiles as subroutines for validation.

### D5) Runtime Policy Enforcement

Implement a policy layer that can be used at runtime:

Policy types (v1):
- allow list of tile ids
- deny list of tile ids
- maximum active tiles per token (if/when top-k exists)
- fallback behavior

Enforcement points:
- inside compiled dispatch (prevent dispatch to disallowed tile)
- inside routing (if backend produces disallowed tile)

Telemetry:
- record policy violations and fallbacks

### D6) Make It One Command

DX goal:

- `trix doctor` stays the trust handshake.
- Add `trix bundle verify` as the second handshake.

Minimal expected flow:
1) `trix export-bundle ... --validate`
2) `trix bundle verify --outdir bundle_dir`
3) ship `bundle_dir`
4) on target: `trix load-bundle --outdir bundle_dir --validate`

## 7. Tests

Bundle integrity:
- corrupt a file -> verification fails
- change bundle.json -> manifest mismatch

Compatibility:
- missing optional files handled gracefully
- schema version mismatch produces clear errors

Policy enforcement:
- deny a tile referenced by compiled dispatch -> fallback and telemetry

Drift policy:
- drift benchmark output can be evaluated against policy thresholds

## 8. Benchmarks

Add to suite:
- "policy enforcement" microbench (tiny)
- "bundle verify" time

## 9. Risks

- Overengineering security before nailing semantics.
- Making policy too powerful too early (hard to reason about).
- Confusing integrity (tamper detection) with identity (signing).

## 10. Open Questions

- Should bundles be single-file archives (tar/zip) vs directories?
- What is the canonical fingerprint of signatures? (packed bytes vs exported json)
- Where should drift eval sets live? (embedded in bundle vs provided externally)
- How should compiled dispatch behave when a class_hint is missing? (always fallback?)
