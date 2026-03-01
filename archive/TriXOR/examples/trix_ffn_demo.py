#!/usr/bin/env python3
"""
TriXFFN Demo - Zero-Parameter Emergent Routing

Demonstrates the signature-based routing mechanism where routing
decisions emerge from weight structure - no learned gate network needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim

from trix import TriXFFN, TriXBlock, TriXStack


def demo_basic_usage():
    """Basic TriXFFN usage."""
    print("=" * 60)
    print("Demo 1: Basic TriXFFN Usage")
    print("=" * 60)
    
    # Create FFN with emergent routing
    ffn = TriXFFN(d_model=64, num_tiles=4)
    
    # Forward pass - routing happens automatically!
    x = torch.randn(8, 64)
    output, routing = ffn(x)
    
    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {output.shape}")
    print(f"Routing shape: {routing.shape}")
    print(f"\nRouting is one-hot: {routing.sum(dim=-1).mean():.1f} (should be 1.0)")


def demo_routing_analysis():
    """Analyze routing behavior."""
    print("\n" + "=" * 60)
    print("Demo 2: Routing Analysis")
    print("=" * 60)
    
    ffn = TriXFFN(d_model=64, num_tiles=4)
    
    # Get tile signatures
    signatures = ffn.get_tile_signatures()
    print(f"\nTile signatures shape: {signatures.shape}")
    print(f"Signatures are ternary: values in {signatures.unique().tolist()}")
    
    # Analyze routing on sample data
    x = torch.randn(100, 64)
    stats = ffn.get_routing_stats(x)
    
    print(f"\nRouting statistics:")
    print(f"  Tile usage: {[f'{u:.1%}' for u in stats['tile_usage']]}")
    print(f"  Balance: {stats['balance']:.2f} (1.0 = perfect)")
    print(f"  Dominant tile: {stats['dominant_tile']}")
    
    # Signature diversity
    diversity = ffn.get_signature_diversity()
    print(f"  Signature diversity: {diversity:.1%}")


def demo_input_dependent_routing():
    """Show that different inputs route differently."""
    print("\n" + "=" * 60)
    print("Demo 3: Input-Dependent Routing")
    print("=" * 60)
    
    ffn = TriXFFN(d_model=64, num_tiles=4)
    
    # Create two types of inputs
    x_positive = torch.abs(torch.randn(50, 64))  # All positive
    x_negative = -torch.abs(torch.randn(50, 64))  # All negative
    
    stats_pos = ffn.get_routing_stats(x_positive)
    stats_neg = ffn.get_routing_stats(x_negative)
    
    print(f"\nPositive inputs route to: {[f'{u:.0%}' for u in stats_pos['tile_usage']]}")
    print(f"Negative inputs route to: {[f'{u:.0%}' for u in stats_neg['tile_usage']]}")
    
    # Different dominant tiles?
    if stats_pos['dominant_tile'] != stats_neg['dominant_tile']:
        print("\n[OK] Different input types route to different tiles!")
    else:
        print("\n[!] Same dominant tile (try different random seed)")


def demo_training():
    """Show routing evolution during training."""
    print("\n" + "=" * 60)
    print("Demo 4: Training with Emergent Routing")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    ffn = TriXFFN(d_model=64, num_tiles=4, dropout=0.0)
    optimizer = optim.Adam(ffn.parameters(), lr=0.01)
    
    # Training data
    x = torch.randn(64, 64)
    target = torch.randn(64, 64)
    
    print(f"\n{'Step':<8} {'Loss':<12} {'Tile Usage':<30} {'Diversity'}")
    print("-" * 70)
    
    for step in range(51):
        optimizer.zero_grad()
        out, _ = ffn(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            stats = ffn.get_routing_stats(x)
            diversity = ffn.get_signature_diversity()
            usage = " ".join([f"{u:.0%}" for u in stats['tile_usage']])
            print(f"{step:<8} {loss.item():<12.4f} {usage:<30} {diversity:.1%}")


def demo_transformer_block():
    """Using TriXBlock in a transformer."""
    print("\n" + "=" * 60)
    print("Demo 5: TriXBlock Transformer")
    print("=" * 60)
    
    block = TriXBlock(d_model=64, n_heads=4, num_tiles=4)
    
    # Process a sequence
    x = torch.randn(4, 16, 64)  # [batch, seq_len, d_model]
    output, routing = block(x, is_causal=True)
    
    print(f"\nInput shape:   {x.shape}")
    print(f"Output shape:  {output.shape}")
    print(f"Routing shape: {routing.shape}")
    
    # Routing stats
    stats = block.get_routing_stats(x)
    print(f"\nRouting: {[f'{u:.0%}' for u in stats['tile_usage']]}")


def demo_ideal_input_routing():
    """Each tile's signature routes to itself."""
    print("\n" + "=" * 60)
    print("Demo 6: Ideal Input Routing")
    print("=" * 60)
    
    ffn = TriXFFN(d_model=32, num_tiles=4)
    signatures = ffn.get_tile_signatures()
    
    print("\nTesting: Does each tile's 'ideal' input route to itself?")
    print("-" * 40)
    
    for t in range(4):
        # A tile's signature IS its ideal input
        ideal_input = signatures[t].unsqueeze(0)
        routing = ffn.compute_routing(ideal_input)
        winner = routing.argmax().item()
        
        status = "MATCH" if winner == t else f"MISMATCH -> T{winner}"
        print(f"  Tile {t} ideal input routes to: {status}")


def main():
    print("\n" + "=" * 60)
    print("TriX Zero-Parameter Emergent Routing Demo")
    print("=" * 60)
    print("\nKey insight: Routing emerges from weight signatures.")
    print("No learned gate network. Zero routing parameters.")
    print("Three lines of code:\n")
    print("    signatures = [tile.weight.sum(dim=0).sign() for tile in tiles]")
    print("    scores = input @ torch.stack(signatures).T")
    print("    winner = scores.argmax(dim=-1)")
    
    demo_basic_usage()
    demo_routing_analysis()
    demo_input_dependent_routing()
    demo_training()
    demo_transformer_block()
    demo_ideal_input_routing()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nThe simplest solution won: read routing from the weights.")


if __name__ == "__main__":
    main()
