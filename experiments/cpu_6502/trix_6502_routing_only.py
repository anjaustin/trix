#!/usr/bin/env python3
"""
TriX 6502 Routing Test

Simplified question: Does TriX routing naturally cluster operations by category?

We DON'T care about prediction accuracy here.
We care about: given diverse 6502 operations, do tiles specialize?

Test:
1. Generate mixed operations
2. Route through TriX (just the routing, not full training)
3. Measure which tiles claim which operations
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/flynnconceivable')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from trix.nn import SparseLookupFFN, SparseLookupFFNv2

# Categories
OPCODES = ['ADC', 'SBC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC', 'CMP']
OP_TO_CAT = {
    'ADC': 'ALU', 'SBC': 'ALU',
    'AND': 'LOGIC', 'ORA': 'LOGIC', 'EOR': 'LOGIC',
    'ASL': 'SHIFT', 'LSR': 'SHIFT', 'ROL': 'SHIFT', 'ROR': 'SHIFT',
    'INC': 'INCDEC', 'DEC': 'INCDEC',
    'CMP': 'COMPARE',
}


def generate_operation_vectors(n_samples=10000, d_model=64):
    """
    Generate input vectors that represent different 6502 operations.
    
    Each vector encodes: opcode (one-hot) + A value + B value + carry
    """
    data = []
    
    for _ in range(n_samples):
        # Random operation
        op = np.random.choice(OPCODES)
        op_idx = OPCODES.index(op)
        
        # Random values
        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 2)
        
        # Create vector
        vec = np.zeros(d_model, dtype=np.float32)
        
        # Opcode one-hot (first 12 dims)
        vec[op_idx] = 1.0
        
        # A value (binary, next 8 dims)
        for i in range(8):
            vec[12 + i] = (a >> i) & 1
        
        # B value (binary, next 8 dims)  
        for i in range(8):
            vec[20 + i] = (b >> i) & 1
        
        # Carry (dim 28)
        vec[28] = c
        
        # Category embedding (dims 29-33) - this is what we want routing to discover
        cat = OP_TO_CAT[op]
        cat_idx = ['ALU', 'LOGIC', 'SHIFT', 'INCDEC', 'COMPARE'].index(cat)
        vec[29 + cat_idx] = 1.0
        
        # Add some noise to remaining dims
        vec[34:] = np.random.randn(d_model - 34) * 0.1
        
        data.append({
            'vec': vec,
            'op': op,
            'cat': cat,
            'a': a, 'b': b, 'c': c
        })
    
    return data


def test_routing_specialization(num_tiles=16, use_v2=False, n_samples=10000):
    """
    Test whether TriX routing discovers operation categories.
    """
    
    print("=" * 70)
    print(f"TRIX 6502 ROUTING SPECIALIZATION TEST")
    print(f"Tiles: {num_tiles}, Version: {'v2' if use_v2 else 'v1'}")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 64
    
    # Generate data
    print(f"\n[1] Generating {n_samples:,} operation vectors...")
    data = generate_operation_vectors(n_samples, d_model)
    
    # Count categories
    cat_counts = defaultdict(int)
    for d in data:
        cat_counts[d['cat']] += 1
    print("Category distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    
    # Create TriX FFN
    print(f"\n[2] Creating TriX FFN...")
    kwargs = {}
    if use_v2:
        kwargs = {'ternary_weight': 0.01, 'sparsity_weight': 0.01, 'diversity_weight': 0.01}
    
    FFN = SparseLookupFFNv2 if use_v2 else SparseLookupFFN
    ffn = FFN(d_model=d_model, num_tiles=num_tiles, tiles_per_cluster=4, **kwargs).to(device)
    
    # Run routing WITHOUT training - just see where things route initially
    print(f"\n[3] Testing INITIAL routing (no training)...")
    tile_assignments_initial = test_routing(ffn, data, device)
    analyze_routing(tile_assignments_initial, "Initial (random)")
    
    # Now train the FFN on a simple reconstruction task
    # This should encourage specialization
    print(f"\n[4] Training on reconstruction task...")
    train_reconstruction(ffn, data, device, epochs=30)
    
    # Test routing AFTER training
    print(f"\n[5] Testing TRAINED routing...")
    tile_assignments_trained = test_routing(ffn, data, device)
    analyze_routing(tile_assignments_trained, "After training")
    
    return tile_assignments_trained


def test_routing(ffn, data, device):
    """Route all samples through FFN and record tile assignments."""
    ffn.eval()
    
    tile_assignments = defaultdict(lambda: defaultdict(int))
    
    vecs = torch.tensor(np.array([d['vec'] for d in data]), device=device)
    ops = [d['op'] for d in data]
    cats = [d['cat'] for d in data]
    
    batch_size = 1000
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_vecs = vecs[i:i+batch_size].unsqueeze(1)  # (B, 1, d)
            
            _, routing_info, _ = ffn(batch_vecs)
            tiles = routing_info['tile_idx'].squeeze(-1).cpu().numpy()
            
            for j, tile in enumerate(tiles):
                tile_assignments[int(tile)][ops[i + j]] += 1
    
    return tile_assignments


def train_reconstruction(ffn, data, device, epochs=30, batch_size=256):
    """Train FFN on simple reconstruction task."""
    
    vecs = torch.tensor(np.array([d['vec'] for d in data]), device=device)
    
    # Target: reconstruct first 34 dims (opcode + values + carry + category)
    targets = vecs[:, :34].clone()
    
    optimizer = torch.optim.Adam(ffn.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        ffn.train()
        perm = torch.randperm(len(data), device=device)
        total_loss = 0
        
        for i in range(0, len(data) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            batch_vecs = vecs[idx].unsqueeze(1)
            
            out, _, aux_losses = ffn(batch_vecs)
            out = out.squeeze(1)
            
            # Reconstruction loss on meaningful dims
            loss = F.mse_loss(out[:, :34], targets[idx]) + aux_losses.get('total_aux', torch.tensor(0.0, device=device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/(len(data)//batch_size):.4f}")


def analyze_routing(tile_assignments, phase):
    """Analyze routing specialization."""
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {phase}")
    print(f"{'='*60}")
    
    # Per-tile breakdown
    all_purities = []
    
    for tile in sorted(tile_assignments.keys()):
        op_counts = tile_assignments[tile]
        total = sum(op_counts.values())
        if total < 50:
            continue
        
        # Aggregate by category
        cat_counts = defaultdict(int)
        for op, count in op_counts.items():
            cat_counts[OP_TO_CAT[op]] += count
        
        dominant_cat = max(cat_counts, key=cat_counts.get)
        purity = cat_counts[dominant_cat] / total
        all_purities.append(purity)
        
        # Top 3 operations
        top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:3]
        top_str = ', '.join([f"{op}:{cnt}" for op, cnt in top_ops])
        
        bar = '#' * int(purity * 20)
        print(f"Tile {tile:2d}: {dominant_cat:8s} [{bar:20s}] {purity:.0%} | {top_str}")
    
    # Summary
    avg_purity = np.mean(all_purities) if all_purities else 0
    print(f"\nAverage category purity: {avg_purity:.2f}")
    
    # Tiles per category
    cat_tiles = defaultdict(list)
    for tile, op_counts in tile_assignments.items():
        cat_counts = defaultdict(int)
        for op, count in op_counts.items():
            cat_counts[OP_TO_CAT[op]] += count
        total = sum(cat_counts.values())
        if total < 50:
            continue
        dominant_cat = max(cat_counts, key=cat_counts.get)
        purity = cat_counts[dominant_cat] / total
        if purity > 0.4:
            cat_tiles[dominant_cat].append((tile, purity))
    
    print("\nCategory → Tiles mapping:")
    for cat in ['ALU', 'LOGIC', 'SHIFT', 'INCDEC', 'COMPARE']:
        tiles = cat_tiles.get(cat, [])
        if tiles:
            tile_str = ', '.join([f"T{t}({p:.0%})" for t, p in sorted(tiles, key=lambda x: -x[1])])
            print(f"  {cat:8s}: {tile_str}")
        else:
            print(f"  {cat:8s}: (none)")
    
    # Verdict
    print("\n" + "-" * 60)
    if avg_purity > 0.6:
        print("STRONG SPECIALIZATION: Tiles cluster by operation category")
    elif avg_purity > 0.4:
        print("MODERATE SPECIALIZATION: Some category clustering observed")
    else:
        print("WEAK SPECIALIZATION: Tiles do not cluster by category")
    
    return avg_purity


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--tiles", type=int, default=16)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    
    test_routing_specialization(
        num_tiles=args.tiles,
        use_v2=args.v2,
        n_samples=args.samples
    )
