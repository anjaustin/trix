#!/usr/bin/env python3
"""
Basic TriX Usage Example

Demonstrates:
1. Creating a TriXLinear layer
2. Training with STE
3. Packing weights for fast inference
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim

from trix import TriXLinear, GatedFFN


def main():
    print("=" * 60)
    print("TriX Basic Usage Example")
    print("=" * 60)
    
    # 1. Basic TriXLinear layer
    print("\n[1] TriXLinear Layer")
    print("-" * 40)
    
    layer = TriXLinear(
        in_features=256,
        out_features=512,
        num_tiles=4
    )
    
    print(f"Layer: {layer}")
    print(f"Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    
    # Forward pass with all tiles active
    batch_size = 8
    x = torch.randn(batch_size, 256)
    gate = torch.ones(batch_size, 4)  # All tiles active
    
    layer.train()
    output = layer(x, gate)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 2. Training example
    print("\n[2] Training with STE")
    print("-" * 40)
    
    # Simple training loop
    optimizer = optim.Adam(layer.parameters(), lr=0.01)
    target = torch.randn(batch_size, 512)
    
    for step in range(5):
        optimizer.zero_grad()
        output = layer(x, gate)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1}: loss = {loss.item():.4f}")
    
    # 3. GatedFFN example
    print("\n[3] GatedFFN Layer")
    print("-" * 40)
    
    ffn = GatedFFN(
        d_model=256,
        expansion=4,
        num_tiles=4,
        noise_scale=1.0
    )
    
    x_3d = torch.randn(4, 16, 256)  # [batch, seq_len, d_model]
    ffn.train()
    output, gates = ffn(x_3d)
    
    print(f"FFN input shape: {x_3d.shape}")
    print(f"FFN output shape: {output.shape}")
    print(f"Gates shape: {gates.shape}")
    
    # Gate distribution
    gate_usage = gates.mean(dim=(0, 1))
    print(f"Gate usage: {gate_usage.tolist()}")
    
    # 4. Packed inference
    print("\n[4] Packed Inference")
    print("-" * 40)
    
    try:
        from trix import is_neon_available
        
        if is_neon_available():
            layer.eval()
            layer.pack()
            
            # Run inference with packed weights
            output_packed = layer(x, gate)
            print(f"Packed inference output shape: {output_packed.shape}")
            print("NEON kernel active")
        else:
            print("NEON kernel not available (build the C++ library)")
            print("Using PyTorch fallback")
    except Exception as e:
        print(f"Packed inference unavailable: {e}")
    
    print("\n" + "=" * 60)
    print("Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
