#!/usr/bin/env python3
"""
Repro script: XOR Signature Compression

Claim: XOR superposition achieves high compression with lossless routing.
This script creates random ternary signatures with high structural similarity
(as observed in trained models), compresses them, and verifies:
  1. Decompression is lossless (roundtrip exact).
  2. Routing decisions are preserved (argmax agreement = 100%).
  3. Compression ratio is reported.

Expected output: see repro_xor_compression.expected.json

Usage:
    python scripts/repro/repro_xor_compression.py
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from trix.nn.xor_superposition import CompressedSignatures


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    num_tiles = 64
    d_model = 512

    # Generate a centroid and small deltas (mimics trained signature similarity).
    centroid = torch.randint(-1, 2, (d_model,)).float()
    flip_rate = 0.01  # 99% similarity between signatures
    sigs = centroid.unsqueeze(0).expand(num_tiles, -1).clone()
    for i in range(num_tiles):
        mask = torch.rand(d_model) < flip_rate
        sigs[i, mask] = torch.randint(-1, 2, (mask.sum().item(),)).float()

    # Compress
    cs = CompressedSignatures()
    compressed = cs.compress(sigs)
    decompressed = compressed.decompress_all()

    # Check lossless roundtrip
    roundtrip_exact = torch.equal(sigs, decompressed)

    # Check routing agreement on random inputs
    num_queries = 1000
    queries = torch.randn(num_queries, d_model)
    original_scores = queries @ sigs.T
    original_routes = original_scores.argmax(dim=1)

    decompressed_scores = queries @ decompressed.T
    decompressed_routes = decompressed_scores.argmax(dim=1)

    agreement = (original_routes == decompressed_routes).float().mean().item()

    # Compression ratio
    stats = compressed.get_compression_stats()
    ratio = stats.compression_ratio

    results = {
        "roundtrip_exact": roundtrip_exact,
        "routing_agreement": round(agreement, 4),
        "compression_ratio": round(ratio, 1),
        "num_tiles": num_tiles,
        "d_model": d_model,
        "flip_rate": flip_rate,
    }

    print(json.dumps(results, indent=2))

    # Compare against expected
    expected_path = Path(__file__).with_suffix(".expected.json")
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())
        ok = True
        if results["roundtrip_exact"] != expected["roundtrip_exact"]:
            print(
                f"FAIL: roundtrip_exact = {results['roundtrip_exact']}, expected {expected['roundtrip_exact']}"
            )
            ok = False
        if results["routing_agreement"] < expected["routing_agreement"]:
            print(
                f"FAIL: routing_agreement = {results['routing_agreement']}, expected >= {expected['routing_agreement']}"
            )
            ok = False
        if results["compression_ratio"] < expected["min_compression_ratio"]:
            print(
                f"FAIL: compression_ratio = {results['compression_ratio']}, expected >= {expected['min_compression_ratio']}"
            )
            ok = False
        if ok:
            print("PASS")
        sys.exit(0 if ok else 1)
    else:
        print(f"(no expected output file at {expected_path}; skipping comparison)")


if __name__ == "__main__":
    main()
