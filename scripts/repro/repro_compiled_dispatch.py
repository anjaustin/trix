#!/usr/bin/env python3
"""
Repro script: Compiled Dispatch Agreement

Claim: Compiled dispatch produces identical routing to dynamic dispatch
for stable classes.

This script trains a small SparseLookupFFNv2 model on a synthetic
classification task, compiles stable classes, and verifies:
  1. Compiled dispatch agrees with dynamic dispatch (agreement rate).
  2. Dispatch table is serializable (roundtrip).

Expected output: see repro_compiled_dispatch.expected.json

Usage:
    python scripts/repro/repro_compiled_dispatch.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from trix.nn import SparseLookupFFNv2, CompiledDispatch


def main():
    torch.manual_seed(42)

    d_model = 64
    num_tiles = 8
    num_classes = 4
    num_samples = 200
    num_epochs = 50

    # Create synthetic data with class structure
    X = torch.randn(num_samples, d_model)
    labels = torch.randint(0, num_classes, (num_samples,))
    # Give each class a distinct centroid so routing can specialize
    for c in range(num_classes):
        mask = labels == c
        X[mask] += torch.randn(d_model) * 3

    # Build model
    ffn = SparseLookupFFNv2(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=4,
        ternary_weight=0.01,
        sparsity_weight=0.01,
    )
    optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3)

    # Train
    ffn.train()
    for epoch in range(num_epochs):
        output, info, aux = ffn(X.unsqueeze(1), labels=labels)
        loss = output.squeeze(1).pow(2).mean() + aux.get("total_aux", 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compile
    ffn.eval()
    compiler = CompiledDispatch(ffn)

    # Profile all classes (uses the claim matrix built during training)
    compiler.profile_all(num_classes=num_classes)

    compiler.compile_stable(threshold=0.3)

    # Measure agreement
    agree = 0
    total = 0
    for c in range(num_classes):
        x_c = X[labels == c].unsqueeze(1)

        # Dynamic path
        _, info_dyn, _ = ffn(x_c)
        tiles_dyn = info_dyn["tile_idx"]

        # Compiled path
        _, info_comp, _ = compiler.forward(x_c, class_hint=c, confidence=0.9)
        tiles_comp = info_comp["tile_idx"]

        agree += (tiles_dyn == tiles_comp).sum().item()
        total += tiles_dyn.numel()

    agreement = agree / total if total > 0 else 0.0

    # Serialization roundtrip
    table = compiler.export_dispatch_table()

    # Convert numpy/torch scalars to native Python for JSON
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    table_clean = make_serializable(table)
    json_str = json.dumps(table_clean)
    reimported = json.loads(json_str)
    serial_ok = reimported == table_clean

    results = {
        "agreement_rate": round(agreement, 4),
        "num_compiled_classes": len(compiler.dispatch),
        "serialization_roundtrip": serial_ok,
    }

    print(json.dumps(results, indent=2))

    # Compare against expected
    expected_path = Path(__file__).with_suffix(".expected.json")
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())
        ok = True
        if results["agreement_rate"] < expected["min_agreement_rate"]:
            print(
                f"FAIL: agreement_rate = {results['agreement_rate']}, expected >= {expected['min_agreement_rate']}"
            )
            ok = False
        if results["serialization_roundtrip"] != expected["serialization_roundtrip"]:
            print(
                f"FAIL: serialization_roundtrip = {results['serialization_roundtrip']}"
            )
            ok = False
        if ok:
            print("PASS")
        sys.exit(0 if ok else 1)
    else:
        print(f"(no expected output file at {expected_path}; skipping comparison)")


if __name__ == "__main__":
    main()
