# Runtime Policy Enforcement (Mesa 15)

Mesa 15 includes a minimal runtime policy for `tile_id` addresses.

## Policy

`AddressPolicyV1` supports:
- allow list (optional)
- deny list (always applied)
- violation behavior: `fallback` | `fail` | `noop`

## Enforcement Points

- Compiled dispatch: if a compiled class maps to a denied tile, the call falls back to dynamic routing.
- Routing (SparseLookupFFNv2): if routing selects a denied tile and `on_violation=fallback`, it reroutes to the best allowed tile.

## Telemetry

When policy enforcement triggers, routing metadata includes:
- `policy_violation_rate`
- `policy_num_violations`
- `policy_fallback_applied`
- `policy_on_violation`
