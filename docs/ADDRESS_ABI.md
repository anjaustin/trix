# Address ABI (Minimal, v1)

This ABI is the minimal contract expected by Mesa 15 tooling and policy checks.

## Tile/Subroutine ABI

- Input: `Tensor[B, T, D]`
- Output: `Tensor[B, T, D]`

Constraints:
- Output shape matches input shape (drop-in FFN compatible).
- Deterministic in `eval()` given fixed weights.

Optional side outputs (internal):
- `routing_info` (e.g. `tile_idx`, backend name, policy fields)
- `aux_losses` (training-time regularizers)
