# Native Correctness Harness

TriX includes optional native acceleration for packed ternary matmul.

M5 goal: ensure the native path is always a semantics-preserving acceleration of the reference path.

## What We Test

- `pack_weights` native output matches the Python reference packer
- `unpack_weights(pack_weights(w)) == w` for ternary `w`
- `trix_forward` matches a pure-PyTorch reference implementation for a variety of shapes and gate masks

These tests are written to pass even when the native library is not present.
When native is present, we add a stricter native-vs-reference equivalence check.

Files:
- `tests/test_kernel_reference_harness.py`
