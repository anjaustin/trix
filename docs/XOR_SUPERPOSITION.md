# XOR Superposition (Compression Guarantees)

This doc describes the lossless signature compression used to reduce signature storage while preserving routing decisions under stated conditions.

## What Is Guaranteed

1) Lossless signature compression
- `decompress_all(compress(sigs)) == sigs` exactly for ternary signatures `{-1,0,+1}`.

2) Routing equivalence (conditional)
- Dot-product routing can be implemented as XOR+POPCNT distance **only** under the conditions in `docs/DOT_POPCOUNT_EQUIVALENCE.md`.

In particular:
- If inputs are in `{+1,-1}^d`, then `argmax(dot)` is equivalent to `argmin(popcount_distance)` with the 2-bit encoding.
- If inputs can contain zeros, equivalence is recovered by masking out coordinates where `x==0` when computing popcount distance.

## Failure Modes (Expected)

- If you compute popcount distance without masking while inputs contain zeros, equivalence is not universal.
- Tie-degenerate geometries can collapse routing unless tie-breaking/guards are applied.

## Where It Lives

- Implementation: `src/trix/nn/xor_superposition.py`
- Tests:
  - `tests/test_xor_superposition.py` (lossless compress/decompress + equivalence tests)

## Integration

- `SparseLookupFFNv2` supports an opt-in routing backend `flat_popcount` that uses packed XOR+POPCNT distances:
  - set `routing_backend="flat_popcount"` at construction, or pass `backend="flat_popcount"` to `route(...)`.
  - this backend ternarizes inputs via `sign(x)`.
