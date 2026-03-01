#!/usr/bin/env python3
"""
TriX Quick Start for NVIDIA Engineers

Drop-in replacement for your FFN layers with 2-bit sparse routing.

Key features:
- 16x memory compression (2-bit weights)
- 4x speedup potential (sparse routing)
- Zero-parameter routing (emergent from weights)
- Works with any transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# INSTALLATION
# =============================================================================
# 
# pip install torch  # PyTorch 2.0+
# pip install -e .   # Install trix from this repo
#
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix import HierarchicalTriXFFN, HierarchicalTriXBlock, SparseTriXFFN


# =============================================================================
# EXAMPLE 1: Replace Your FFN
# =============================================================================

def example_ffn_replacement():
    """
    Replace a standard FFN with TriX.
    
    Before:
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
    
    After:
        self.ffn = HierarchicalTriXFFN(d_model, num_tiles=16)
    """
    print("=" * 60)
    print("Example 1: FFN Replacement")
    print("=" * 60)
    
    d_model = 512
    batch_size = 32
    seq_len = 128
    
    # Your input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Standard FFN (for comparison)
    standard_ffn = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.ReLU(),
        nn.Linear(d_model * 4, d_model),
    )
    
    # TriX FFN - drop-in replacement
    trix_ffn = HierarchicalTriXFFN(
        d_model=d_model,
        num_tiles=16,           # 16 specialists
        tiles_per_cluster=4,    # 4 clusters of 4 tiles
    )
    
    # Compare sizes
    standard_params = sum(p.numel() for p in standard_ffn.parameters())
    trix_params = sum(p.numel() for p in trix_ffn.parameters())
    
    print(f"Standard FFN params: {standard_params:,}")
    print(f"TriX FFN params:     {trix_params:,}")
    print(f"Ratio: {trix_params/standard_params:.2%}")
    
    # Forward pass
    with torch.no_grad():
        standard_out = standard_ffn(x)
        trix_out, routing_info, aux_losses = trix_ffn(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {trix_out.shape}")
    print(f"Tiles used:   {routing_info['tile_idx'].unique().tolist()[:10]}...")
    
    return trix_ffn


# =============================================================================
# EXAMPLE 2: Full Transformer Block
# =============================================================================

def example_transformer_block():
    """
    Use a complete transformer block with TriX FFN.
    
    Includes:
    - Multi-head attention
    - TriX FFN with residuals + normalization
    - Causal masking support
    """
    print("\n" + "=" * 60)
    print("Example 2: Transformer Block")
    print("=" * 60)
    
    d_model = 256
    n_heads = 8
    batch_size = 16
    seq_len = 64
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # TriX transformer block
    block = HierarchicalTriXBlock(
        d_model=d_model,
        n_heads=n_heads,
        num_tiles=16,
        tiles_per_cluster=4,
    )
    
    # Forward with causal masking (for autoregressive models)
    output, routing_info, aux_losses = block(x, is_causal=True)
    
    params = sum(p.numel() for p in block.parameters())
    print(f"Block params: {params:,}")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return block


# =============================================================================
# EXAMPLE 3: Training Loop Integration
# =============================================================================

def example_training():
    """
    How to integrate TriX into your training loop.
    
    Key: Add aux_losses to your loss function.
    """
    print("\n" + "=" * 60)
    print("Example 3: Training Integration")
    print("=" * 60)
    
    d_model = 128
    vocab_size = 1000
    
    # Simple model with TriX
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.trix = HierarchicalTriXFFN(d_model, num_tiles=8, tiles_per_cluster=4)
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embed(x)
            x, routing, aux = self.trix(x)
            return self.head(x), aux
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Fake data
    x = torch.randint(0, vocab_size, (8, 32))
    y = torch.randint(0, vocab_size, (8, 32))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    logits, aux_losses = model(x)
    
    # Main loss
    task_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    # IMPORTANT: Add auxiliary losses for load balancing
    total_loss = task_loss + aux_losses.get('total_aux', 0)
    
    total_loss.backward()
    optimizer.step()
    
    print(f"Task loss: {task_loss.item():.4f}")
    print(f"Aux loss:  {aux_losses.get('total_aux', 0):.4f}")
    print(f"Total:     {total_loss.item():.4f}")
    
    return model


# =============================================================================
# EXAMPLE 4: Inference Mode (Faster)
# =============================================================================

def example_inference():
    """
    Inference mode disables auxiliary loss computation.
    """
    print("\n" + "=" * 60)
    print("Example 4: Inference Mode")
    print("=" * 60)
    
    ffn = HierarchicalTriXFFN(d_model=256, num_tiles=16)
    x = torch.randn(1, 64, 256)
    
    # Training mode
    ffn.train()
    _, _, aux_train = ffn(x)
    print(f"Training mode - aux_losses: {list(aux_train.keys())}")
    
    # Inference mode
    ffn.eval()
    with torch.no_grad():
        output, routing, aux_eval = ffn(x)
    print(f"Inference mode - aux_losses: {list(aux_eval.keys())}")
    print(f"Output shape: {output.shape}")


# =============================================================================
# EXAMPLE 5: Simple Version (4 tiles)
# =============================================================================

def example_simple():
    """
    If you want something simpler, use SparseTriXFFN with 4 tiles.
    Proven to work well with less complexity.
    """
    print("\n" + "=" * 60)
    print("Example 5: Simple 4-Tile Version")
    print("=" * 60)
    
    ffn = SparseTriXFFN(d_model=256, num_tiles=4)
    x = torch.randn(8, 32, 256)
    
    output, gate, aux = ffn(x)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Gate:   {gate.shape} (one-hot routing)")
    print(f"Params: {sum(p.numel() for p in ffn.parameters()):,}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TriX Quick Start")
    print("  2-Bit Sparse Ternary Neural Networks")
    print("  'Qdrant with a brain at every address'")
    print("=" * 60)
    
    example_ffn_replacement()
    example_transformer_block()
    example_training()
    example_inference()
    example_simple()
    
    print("\n" + "=" * 60)
    print("  Ready to plug into your project!")
    print("=" * 60)
    print("""
Key Points:
1. Replace your FFN with HierarchicalTriXFFN
2. Add aux_losses['total_aux'] to your loss during training
3. Use .eval() mode for inference (no aux computation)
4. More tiles = more specialists = better quality

Questions? Check docs/BUILD_LOG.md for the full journey.
""")
