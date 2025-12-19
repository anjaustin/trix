"""
XOR Routing: Superposition-based Signature Matching

"Two numbers in superposition. Work = nothing. Memory = nothing."

Instead of:
    for tile in tiles:
        score = input @ tile.signature.T  # O(tiles × dim)

Do:
    base_dist = popcount(input ^ base)     # O(dim)
    for delta in sparse_deltas:            # O(k) where k << tiles
        score = base_dist ^ delta

Memory: 32KB → 1KB (if tiles are similar)
Work: O(tiles × dim) → O(dim + k)
Hardware: XOR and POPCNT are 1-cycle operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import numpy as np


def ternary_to_bits(ternary: torch.Tensor) -> torch.Tensor:
    """
    Convert ternary {-1, 0, +1} to 2-bit representation.
    
    Encoding:
        -1 → 01
         0 → 00
        +1 → 10
    
    Args:
        ternary: [..., dim] tensor of {-1, 0, +1}
    
    Returns:
        [..., dim, 2] tensor of bits
    """
    bits = torch.zeros(*ternary.shape, 2, dtype=torch.uint8, device=ternary.device)
    bits[..., 0] = (ternary > 0).to(torch.uint8)   # High bit: positive
    bits[..., 1] = (ternary < 0).to(torch.uint8)   # Low bit: negative
    return bits


def bits_to_ternary(bits: torch.Tensor) -> torch.Tensor:
    """
    Convert 2-bit representation back to ternary.
    
    Args:
        bits: [..., dim, 2] tensor of bits
    
    Returns:
        [..., dim] tensor of {-1, 0, +1}
    """
    return bits[..., 0].long() - bits[..., 1].long()


def xor_distance(a_bits: torch.Tensor, b_bits: torch.Tensor) -> torch.Tensor:
    """
    Compute XOR distance (Hamming distance) between bit representations.
    
    Args:
        a_bits: [..., dim, 2]
        b_bits: [..., dim, 2]
    
    Returns:
        [...] distance (number of differing bits)
    """
    xor = a_bits ^ b_bits
    return xor.sum(dim=(-1, -2))


def popcount(x: torch.Tensor) -> torch.Tensor:
    """Count number of 1 bits. Works on any integer tensor."""
    # For small values, lookup is fastest
    # For larger, use parallel bit counting
    count = torch.zeros_like(x)
    while x.any():
        count += x & 1
        x = x >> 1
    return count


# Lookup table for popcount of bytes 0-255
_POPCOUNT_LUT = torch.tensor([bin(i).count('1') for i in range(256)], dtype=torch.int32)


def popcount_vectorized(x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized population count using lookup table.

    Args:
        x: uint8 tensor of any shape

    Returns:
        int32 tensor with popcount of each byte
    """
    lut = _POPCOUNT_LUT.to(x.device)
    return lut[x.long()]


def pack_ternary_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary {-1, 0, +1} to 2-bit representation in uint8.

    Encoding: +1 -> 01, -1 -> 10, 0 -> 00
    4 values per byte.

    Args:
        x: [batch, dim] ternary tensor

    Returns:
        [batch, (dim+3)//4] uint8 tensor
    """
    batch, dim = x.shape
    packed_dim = (dim + 3) // 4

    # Pad to multiple of 4
    if dim % 4 != 0:
        padding = 4 - (dim % 4)
        x = torch.nn.functional.pad(x, (0, padding), value=0)

    # Reshape to [batch, packed_dim, 4]
    grouped = x.reshape(batch, -1, 4)

    # Encode: +1 -> 01 (1), -1 -> 10 (2), 0 -> 00 (0)
    encoded = torch.zeros_like(grouped, dtype=torch.uint8)
    encoded[grouped > 0] = 1
    encoded[grouped < 0] = 2

    # Pack: [v0, v1, v2, v3] -> v0<<6 | v1<<4 | v2<<2 | v3
    packed = (
        (encoded[..., 0] << 6) |
        (encoded[..., 1] << 4) |
        (encoded[..., 2] << 2) |
        encoded[..., 3]
    )

    return packed


def hamming_distance_batch(
    query: torch.Tensor,
    signatures: torch.Tensor
) -> torch.Tensor:
    """
    Compute Hamming distances from query to all signatures.

    Uses packed 2-bit representation for efficiency.

    Args:
        query: [batch, packed_dim] packed queries (uint8)
        signatures: [num_sigs, packed_dim] packed signatures (uint8)

    Returns:
        [batch, num_sigs] distances
    """
    # Broadcast XOR: [batch, 1, packed_dim] ^ [1, num_sigs, packed_dim]
    xor = query.unsqueeze(1) ^ signatures.unsqueeze(0)
    # [batch, num_sigs, packed_dim]

    bit_counts = popcount_vectorized(xor)
    return bit_counts.sum(dim=-1)


class XORSignatures(nn.Module):
    """
    XOR-compressed signature storage.
    
    Stores one base signature + sparse deltas for other tiles.
    Routing via Hamming distance instead of matmul.
    """
    
    def __init__(self, num_tiles: int, dim: int):
        super().__init__()
        self.num_tiles = num_tiles
        self.dim = dim
        
        # Base signature (ternary)
        self.base = nn.Parameter(torch.zeros(dim))
        
        # Deltas stored as sparse indices + values
        # For now, store full deltas (optimize later)
        self.deltas = nn.Parameter(torch.zeros(num_tiles - 1, dim))
        
        # Initialize randomly
        nn.init.uniform_(self.base, -1, 1)
        nn.init.uniform_(self.deltas, -0.5, 0.5)
    
    def get_base_bits(self) -> torch.Tensor:
        """Get base signature as bits."""
        base_ternary = self.base.sign()
        return ternary_to_bits(base_ternary)
    
    def get_all_signatures(self) -> torch.Tensor:
        """Reconstruct all signatures from base + deltas."""
        base = self.base.sign()
        
        signatures = [base]
        for i in range(self.num_tiles - 1):
            # XOR delta (in ternary space, this is sign flip where delta != 0)
            delta = self.deltas[i].sign()
            # Where delta is non-zero, flip the base sign
            sig = base * (1 - delta.abs()) + delta * delta.abs()
            signatures.append(sig.sign())
        
        return torch.stack(signatures)
    
    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to nearest tile using XOR distance.
        
        Args:
            x: [..., dim] input
        
        Returns:
            tile_idx: [...] winning tile index
            distances: [..., num_tiles] distances to all tiles
        """
        # Get input as bits
        x_sign = x.sign()
        x_bits = ternary_to_bits(x_sign)
        
        # Get all signatures
        sigs = self.get_all_signatures()
        sig_bits = ternary_to_bits(sigs)  # [num_tiles, dim, 2]
        
        # Compute distances
        # x_bits: [..., dim, 2]
        # sig_bits: [num_tiles, dim, 2]
        # We want: [..., num_tiles]
        
        batch_shape = x.shape[:-1]
        x_flat = x_bits.reshape(-1, self.dim, 2)
        
        distances = []
        for t in range(self.num_tiles):
            dist = xor_distance(x_flat, sig_bits[t].unsqueeze(0))
            distances.append(dist)
        
        distances = torch.stack(distances, dim=-1)  # [batch, num_tiles]
        distances = distances.reshape(*batch_shape, self.num_tiles)
        
        # Winner is minimum distance (most similar)
        tile_idx = distances.argmin(dim=-1)
        
        return tile_idx, distances
    
    def get_compression_stats(self) -> Dict:
        """Analyze compression potential."""
        sigs = self.get_all_signatures()
        
        # Count differences between consecutive signatures
        diffs = []
        for i in range(1, self.num_tiles):
            diff = (sigs[i] != sigs[0]).sum().item()
            diffs.append(diff)
        
        avg_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        min_diff = np.min(diffs)
        
        # Full storage
        full_bits = self.num_tiles * self.dim * 2
        
        # XOR storage (base + sparse deltas)
        base_bits = self.dim * 2
        delta_bits = sum(d * 2 for d in diffs)  # Only store differing positions
        xor_bits = base_bits + delta_bits
        
        compression = full_bits / xor_bits if xor_bits > 0 else float('inf')
        
        return {
            'full_bits': full_bits,
            'xor_bits': xor_bits,
            'compression_ratio': compression,
            'avg_delta_size': avg_diff,
            'max_delta_size': max_diff,
            'min_delta_size': min_diff,
        }


class XORRouter(nn.Module):
    """
    Full XOR-based router for TriX.
    
    Combines:
    - XOR signature matching (content)
    - Positional B-spline (spatial)
    - State-based selection (temporal)
    """
    
    def __init__(
        self,
        d_model: int,
        num_tiles: int,
        num_states: int = 1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_states = num_states
        self.max_seq_len = max_seq_len
        
        # XOR signatures for content routing
        self.signatures = XORSignatures(num_tiles, d_model)
        
        # Per-state signature modulation (for temporal routing)
        if num_states > 1:
            self.state_masks = nn.Parameter(torch.ones(num_states, num_tiles))
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route using XOR distance + optional position/state modulation.
        
        Args:
            x: [B, T, D] or [B, D]
            positions: [B, T] or [B]
            states: [B, T] or [B]
        
        Returns:
            tile_idx: winning tiles
            scores: routing scores (negative distance)
        """
        # XOR routing (content)
        tile_idx, distances = self.signatures.route(x)
        
        # Convert distance to score (lower distance = higher score)
        scores = -distances.float()
        
        # State modulation (temporal)
        if states is not None and self.num_states > 1:
            # Mask certain tiles based on state
            state_mask = self.state_masks[states]  # [..., num_tiles]
            scores = scores * state_mask
            tile_idx = scores.argmax(dim=-1)
        
        return tile_idx, scores


def test_xor_routing():
    """Test XOR routing."""
    print("=" * 60)
    print("XOR ROUTING TEST")
    print("=" * 60)
    
    num_tiles = 16
    dim = 64
    batch = 32
    
    router = XORRouter(d_model=dim, num_tiles=num_tiles)
    
    # Random input
    x = torch.randn(batch, dim)
    
    # Route
    tile_idx, scores = router(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Tile indices: {tile_idx.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Winning tiles: {tile_idx[:10].tolist()}")
    
    # Compression stats
    stats = router.signatures.get_compression_stats()
    print(f"\nCompression stats:")
    print(f"  Full storage: {stats['full_bits']:,} bits ({stats['full_bits']//8:,} bytes)")
    print(f"  XOR storage:  {stats['xor_bits']:,} bits ({stats['xor_bits']//8:,} bytes)")
    print(f"  Compression:  {stats['compression_ratio']:.2f}x")
    print(f"  Avg delta:    {stats['avg_delta_size']:.1f} / {dim} dims")
    
    print("\n" + "=" * 60)
    print("XOR routing works!")
    print("=" * 60)


if __name__ == "__main__":
    test_xor_routing()
