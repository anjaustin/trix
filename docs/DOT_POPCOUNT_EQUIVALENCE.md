# Dot/Popcount Equivalence (Ternary Routing)

This note specifies the exact conditions under which a dot-product routing rule can be implemented as a popcount (Hamming-on-bits) distance computation.

Status: draft
Last updated: 2026-03-01

## 1. The Routing Rule

Given input `x` and tile signatures `s_t`, a common routing rule is:

```
t* = argmax_t dot(x, s_t)
```

In TriX, we typically use ternary representations `{-1, 0, +1}`.

## 2. Encoding

We encode ternary values into 2-bit codes:

- `0  -> 00`
- `+1 -> 01`
- `-1 -> 10`

This encoding has an important property when `x_i` is non-zero:

- The popcount of `code(x_i) XOR code(s_i)` equals the per-coordinate cost:
  - 0 bits differ for match (`s_i == x_i`)
  - 1 bit differs for neutral (`s_i == 0`)
  - 2 bits differ for mismatch (`s_i == -x_i`)

## 3. Theorem (Exact)

Let `x ∈ {+1, -1}^d` (no zeros), and `s ∈ {-1, 0, +1}^d`.

Define:

- `D(x, s) = popcount(code(x) XOR code(s))` summed over coordinates.

Then:

```
argmax_s dot(x, s) == argmin_s D(x, s)
```

Proof sketch:

For each coordinate, if `x_i` is non-zero then:

- `dot_i = x_i * s_i ∈ {+1, 0, -1}`
- `cost_i = popcount(code(x_i) XOR code(s_i)) ∈ {0, 1, 2}`

And `dot_i = 1 - cost_i`. Summing over `i`, `dot(x,s) = d - D(x,s)`.

## 4. Inputs With Zeros

If `x ∈ {-1, 0, +1}^d` can contain zeros, the equivalence is no longer universal if you compute popcount over all coordinates, because dot ignores coordinates where `x_i = 0`.

However, an exact equivalence is recovered by masking:

- define a mask that ignores coordinates where `x_i == 0`
- compute popcount only over non-zero coordinates of `x`

## 5. Practical Implication

This is the precise condition needed to justify replacing dot-product routing with a packed XOR+POPCNT distance computation.

If a codepath claims "argmax(dot) = argmin(hamming)" it must also state:

- the ternary encoding
- whether inputs contain zeros
- whether masking is applied
