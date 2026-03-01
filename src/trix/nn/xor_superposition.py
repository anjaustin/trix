"""XOR superposition signature compression.

This module provides a lossless compression scheme for ternary signatures by
storing:
- one base signature (centroid)
- sparse deltas for each signature (positions + values)

It also provides packed (2-bit) utilities and Hamming distance computations
that can be used for routing.

Important equivalence note:
- Dot-product routing and XOR+POPCNT routing are exactly equivalent under the
  conditions documented in `docs/DOT_POPCOUNT_EQUIVALENCE.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SparseDelta:
    """Sparse delta from base signature.

    Stores only positions where `sig != base` and the ternary value at those
    positions.
    """

    positions: torch.Tensor  # int16
    values: torch.Tensor  # int8 in {-1,0,+1}

    @property
    def memory_bytes(self) -> int:
        return int(self.positions.numel() * 2 + self.values.numel() * 1)

    @property
    def nnz(self) -> int:
        return int(self.positions.numel())

    def to(self, device: torch.device) -> "SparseDelta":
        return SparseDelta(
            positions=self.positions.to(device), values=self.values.to(device)
        )


class CompressionStats(NamedTuple):
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    mean_delta_sparsity: float
    max_delta_sparsity: float
    num_signatures: int
    dim: int


def pack_ternary_to_uint8(ternary: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1,0,+1} into uint8 with 2 bits per value.

    Encoding per value:
    - 0  -> 00
    - +1 -> 01
    - -1 -> 10

    Layout: 4 ternary values per byte; for group [v0,v1,v2,v3]:
      byte = v0 | (v1<<2) | (v2<<4) | (v3<<6)
    """

    shape = ternary.shape
    dim = int(shape[-1])
    packed_dim = (dim + 3) // 4

    flat = ternary.reshape(-1, dim)
    batch = flat.shape[0]

    if dim % 4 != 0:
        padding = 4 - (dim % 4)
        flat = F.pad(flat, (0, padding), value=0)

    grouped = flat.reshape(batch, packed_dim, 4)
    encoded = torch.zeros_like(grouped, dtype=torch.uint8)
    encoded[grouped > 0] = 1
    encoded[grouped < 0] = 2

    packed = (
        (encoded[..., 0])
        | (encoded[..., 1] << 2)
        | (encoded[..., 2] << 4)
        | (encoded[..., 3] << 6)
    )
    return packed.reshape(*shape[:-1], packed_dim)


def unpack_uint8_to_ternary(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack 2-bit ternary encoding back to float32 {-1,0,+1}."""

    shape = packed.shape
    packed_dim = int(shape[-1])
    flat = packed.reshape(-1, packed_dim)
    batch = flat.shape[0]

    b0 = (flat >> 0) & 0x03
    b1 = (flat >> 2) & 0x03
    b2 = (flat >> 4) & 0x03
    b3 = (flat >> 6) & 0x03
    expanded = torch.stack([b0, b1, b2, b3], dim=-1).reshape(batch, -1)

    out = torch.zeros_like(expanded, dtype=torch.float32)
    out[expanded == 1] = 1.0
    out[expanded == 2] = -1.0
    out = out[:, :dim]
    return out.reshape(*shape[:-1], dim)


_POPCOUNT_LUT = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int32)


def popcount_vectorized(x: torch.Tensor) -> torch.Tensor:
    lut = _POPCOUNT_LUT.to(x.device)
    return lut[x.long()]


def _mask_nonzero_groups(packed_x: torch.Tensor) -> torch.Tensor:
    """Mask bits for coordinates where x != 0.

    For each 2-bit group in packed_x:
      group == 00 -> mask 00
      group != 00 -> mask 11
    """
    # group masks
    m0 = ((packed_x & 0x03) != 0).to(torch.uint8) * 0x03
    m1 = ((packed_x & 0x0C) != 0).to(torch.uint8) * 0x0C
    m2 = ((packed_x & 0x30) != 0).to(torch.uint8) * 0x30
    m3 = ((packed_x & 0xC0) != 0).to(torch.uint8) * 0xC0
    return (m0 | m1 | m2 | m3).to(torch.uint8)


def popcount_distance_packed(
    query: torch.Tensor, signatures: torch.Tensor, *, ignore_x_zeros: bool
) -> torch.Tensor:
    """Compute packed XOR+POPCNT distances.

    Args:
        query: [B, packed_dim] uint8
        signatures: [N, packed_dim] uint8
        ignore_x_zeros: if True, coordinates where query has 00 contribute 0

    Returns:
        [B, N] int distance (bit distance)
    """

    xor = query.unsqueeze(1) ^ signatures.unsqueeze(0)  # [B,N,P]
    if ignore_x_zeros:
        m = _mask_nonzero_groups(query).unsqueeze(1)
        xor = xor & m
    return popcount_vectorized(xor).sum(dim=-1)


class CompressedSignatures:
    """Lossless XOR superposition compression for ternary signatures."""

    def __init__(self):
        self.base_packed: Optional[torch.Tensor] = None  # [packed_dim] uint8
        self.base_ternary: Optional[torch.Tensor] = None  # [dim] float32
        self.deltas: List[SparseDelta] = []
        self.dim: int = 0
        self.num_signatures: int = 0
        self._device: torch.device = torch.device("cpu")

    def compress(self, signatures: torch.Tensor) -> "CompressedSignatures":
        if signatures.dim() != 2:
            raise ValueError("signatures must have shape [num_sigs, dim]")

        sigs = torch.sign(signatures).to(torch.float32)
        self.num_signatures = int(sigs.shape[0])
        self.dim = int(sigs.shape[1])
        self._device = sigs.device

        base = torch.sign(sigs.sum(dim=0))
        self.base_ternary = base.clone()
        self.base_packed = pack_ternary_to_uint8(base.unsqueeze(0)).squeeze(0)

        self.deltas = []
        for i in range(self.num_signatures):
            sig = sigs[i]
            diff_mask = sig != base
            pos = torch.where(diff_mask)[0].to(torch.int16)
            vals = sig[diff_mask].to(torch.int8)
            self.deltas.append(SparseDelta(positions=pos, values=vals))

        return self

    def decompress(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.num_signatures:
            raise IndexError(f"index {index} out of range")
        if self.base_ternary is None:
            raise RuntimeError("not compressed")

        out = self.base_ternary.clone()
        d = self.deltas[index]
        if d.positions.numel():
            out[d.positions.long()] = d.values.to(torch.float32)
        return out

    def decompress_all(self) -> torch.Tensor:
        return torch.stack([self.decompress(i) for i in range(self.num_signatures)])

    def to(self, device: torch.device) -> "CompressedSignatures":
        self._device = device
        if self.base_packed is not None:
            self.base_packed = self.base_packed.to(device)
        if self.base_ternary is not None:
            self.base_ternary = self.base_ternary.to(device)
        self.deltas = [d.to(device) for d in self.deltas]
        return self

    def get_compression_stats(self) -> CompressionStats:
        original_bytes = self.num_signatures * self.dim * 4
        base_bytes = (
            int(self.base_packed.numel()) if self.base_packed is not None else 0
        )
        delta_bytes = sum(d.memory_bytes for d in self.deltas)
        compressed_bytes = base_bytes + int(delta_bytes)

        sparsities = [(d.nnz / self.dim) if self.dim else 0.0 for d in self.deltas]
        mean_s = float(sum(sparsities) / len(sparsities)) if sparsities else 0.0
        max_s = float(max(sparsities)) if sparsities else 0.0

        ratio = (
            (original_bytes / compressed_bytes) if compressed_bytes else float("inf")
        )
        return CompressionStats(
            original_bytes=int(original_bytes),
            compressed_bytes=int(compressed_bytes),
            compression_ratio=float(ratio),
            mean_delta_sparsity=float(mean_s),
            max_delta_sparsity=float(max_s),
            num_signatures=int(self.num_signatures),
            dim=int(self.dim),
        )

    def export(self) -> dict:
        """Export as a JSON-serializable dict."""
        if self.base_packed is None or self.base_ternary is None:
            raise RuntimeError("not compressed")

        return {
            "schema_version": 1,
            "dim": int(self.dim),
            "num_signatures": int(self.num_signatures),
            "base_packed": self.base_packed.detach().cpu().tolist(),
            # Store base_ternary to allow exact decompression without unpacking.
            "base_ternary": self.base_ternary.detach().cpu().to(torch.int8).tolist(),
            "deltas": [
                {
                    "positions": d.positions.detach().cpu().to(torch.int16).tolist(),
                    "values": d.values.detach().cpu().to(torch.int8).tolist(),
                }
                for d in self.deltas
            ],
        }

    @staticmethod
    def from_export(
        data: dict, *, device: Optional[torch.device] = None
    ) -> "CompressedSignatures":
        """Import from an export dict."""
        if int(data.get("schema_version", 0)) != 1:
            raise ValueError("unsupported compressed signature schema")

        dim = int(data["dim"])
        n = int(data["num_signatures"])

        dev = device or torch.device("cpu")

        out = CompressedSignatures()
        out.dim = dim
        out.num_signatures = n
        out._device = dev

        out.base_packed = torch.tensor(
            data["base_packed"], dtype=torch.uint8, device=dev
        )
        base_tern = torch.tensor(data["base_ternary"], dtype=torch.int8, device=dev).to(
            torch.float32
        )
        if base_tern.numel() != dim:
            raise ValueError("base_ternary has wrong dim")
        out.base_ternary = base_tern

        deltas = []
        for d in data.get("deltas", []):
            pos = torch.tensor(d["positions"], dtype=torch.int16, device=dev)
            vals = torch.tensor(d["values"], dtype=torch.int8, device=dev)
            deltas.append(SparseDelta(positions=pos, values=vals))
        if len(deltas) != n:
            raise ValueError("delta count mismatch")
        out.deltas = deltas

        return out
