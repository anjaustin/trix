#!/usr/bin/env python3
"""
TriX 6502 with GeometriX (v3) - Positional Routing

Same training as trix_6502_v2_organs.py but with SparseLookupFFNv3.
Tests whether geometric routing helps with deterministic CPU operations.
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from trix.nn.sparse_lookup_v3 import SparseLookupFFNv3

# Ops and categories
OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
OP_TO_CAT = {
    'ADC': 'ALU',
    'AND': 'LOGIC', 'ORA': 'LOGIC', 'EOR': 'LOGIC',
    'ASL': 'SHIFT', 'LSR': 'SHIFT',
    'INC': 'INCDEC', 'DEC': 'INCDEC',
}
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}


def compute_op(op, a, b, c):
    """Compute 6502 operation result."""
    if op == 'ADC':
        return (a + b + c) & 0xFF
    elif op == 'AND':
        return a & b
    elif op == 'ORA':
        return a | b
    elif op == 'EOR':
        return a ^ b
    elif op == 'ASL':
        return (a << 1) & 0xFF
    elif op == 'LSR':
        return a >> 1
    elif op == 'INC':
        return (a + 1) & 0xFF
    elif op == 'DEC':
        return (a - 1) & 0xFF


def generate_data(n_per_op=1000):
    """Generate balanced dataset."""
    data = []
    for op in OPCODES:
        for _ in range(n_per_op):
            a = np.random.randint(0, 256)
            b = np.random.randint(0, 256) if op in ['ADC', 'AND', 'ORA', 'EOR'] else 0
            c = np.random.randint(0, 2) if op == 'ADC' else 0
            result = compute_op(op, a, b, c)
            data.append({'op': op, 'a': a, 'b': b, 'c': c, 'result': result})
    np.random.shuffle(data)
    return data


class TriX6502GeometriX(nn.Module):
    """TriX with GeometriX (v3) for 6502."""
    
    def __init__(self, d_model=128, num_tiles=16):
        super().__init__()
        
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        self.input_proj = nn.Linear(32 + 17, d_model)
        
        # GeometriX FFN with positional routing
        self.ffn = SparseLookupFFNv3(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
            max_seq_len=len(OPCODES),  # Position = opcode index
            position_spread=1.5,
            use_gauge=True,
            use_vortex=True,
            track_topology=True,
        )
        
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )
    
    def forward(self, op_idx, a, b, c):
        B = op_idx.shape[0]
        
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
        
        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x).unsqueeze(1)  # [B, 1, d_model]
        
        # Position = opcode index (geometric routing based on operation type)
        positions = op_idx.float().unsqueeze(1)  # [B, 1]
        
        out, info, aux = self.ffn(x, positions=positions)
        out = out.squeeze(1)
        
        result = self.result_head(out)
        return result, info, aux


def train(model, data, epochs=40, batch_size=256, device='cuda'):
    """Train with tile tracking."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare tensors
    op_idx = torch.tensor([OP_TO_IDX[d['op']] for d in data], device=device)
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    c = torch.tensor([d['c'] for d in data], device=device)
    result = torch.tensor([d['result'] for d in data], device=device)
    result_bits = torch.stack([(result >> i) & 1 for i in range(8)], dim=1).float()
    
    ops = [d['op'] for d in data]
    
    print(f"\nTraining on {len(data):,} samples, {epochs} epochs")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        model.ffn.reset_stats()
        perm = torch.randperm(len(data), device=device)
        total_loss, correct = 0, 0
        tile_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(0, len(data) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            
            res_pred, info, aux = model(op_idx[idx], a[idx], b[idx], c[idx])
            
            loss = F.binary_cross_entropy(res_pred, result_bits[idx]) + aux['total_aux']
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            pred = sum((res_pred[:, i] > 0.5).long() << i for i in range(8))
            correct += (pred == result[idx]).sum().item()
            
            # Track tiles
            tiles = info['tile_idx'].squeeze(-1).cpu().numpy()
            for j, t in enumerate(tiles):
                tile_counts[int(t)][ops[idx[j].item()]] += 1
        
        acc = correct / len(data) * 100
        avg_loss = total_loss / (len(data) // batch_size)
        
        # Category purity
        purities = []
        for tile, op_counts in tile_counts.items():
            cat_counts = defaultdict(int)
            for op, cnt in op_counts.items():
                cat_counts[OP_TO_CAT[op]] += cnt
            total = sum(cat_counts.values())
            if total > 50:
                purities.append(max(cat_counts.values()) / total)
        avg_purity = np.mean(purities) if purities else 0
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            stats = model.ffn.get_routing_stats()
            print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}, acc={acc:.1f}%, "
                  f"purity={avg_purity:.2f}, active_tiles={stats['active_tiles']}/{stats['num_tiles']}")
    
    return tile_counts


def analyze(model, tile_counts):
    """Detailed analysis."""
    print("\n" + "=" * 60)
    print("TILE SPECIALIZATION ANALYSIS (GeometriX)")
    print("=" * 60)
    
    active_tiles = []
    
    for tile in sorted(tile_counts.keys()):
        op_counts = tile_counts[tile]
        total = sum(op_counts.values())
        if total < 100:
            continue
        
        active_tiles.append(tile)
        
        cat_counts = defaultdict(int)
        for op, cnt in op_counts.items():
            cat_counts[OP_TO_CAT[op]] += cnt
        
        dominant_cat = max(cat_counts, key=cat_counts.get)
        purity = cat_counts[dominant_cat] / total
        
        top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:3]
        top_str = ', '.join([f"{op}:{cnt}" for op, cnt in top_ops])
        
        bar = '█' * int(purity * 20) + '░' * (20 - int(purity * 20))
        print(f"Tile {tile:2d}: {dominant_cat:6s} [{bar}] {purity:.0%}")
        print(f"         ops: {top_str}")
    
    # Summary
    print("\n" + "-" * 60)
    print("CATEGORY OWNERSHIP")
    print("-" * 60)
    
    cat_tiles = defaultdict(list)
    for tile in active_tiles:
        op_counts = tile_counts[tile]
        cat_counts = defaultdict(int)
        for op, cnt in op_counts.items():
            cat_counts[OP_TO_CAT[op]] += cnt
        total = sum(cat_counts.values())
        dominant_cat = max(cat_counts, key=cat_counts.get)
        purity = cat_counts[dominant_cat] / total
        if purity > 0.4:
            cat_tiles[dominant_cat].append((tile, purity))
    
    for cat in ['ALU', 'LOGIC', 'SHIFT', 'INCDEC']:
        tiles = cat_tiles.get(cat, [])
        if tiles:
            tiles_str = ', '.join([f"T{t}({p:.0%})" for t, p in sorted(tiles, key=lambda x: -x[1])])
            print(f"  {cat:6s} → {tiles_str}")
        else:
            print(f"  {cat:6s} → (no dedicated tiles)")
    
    # Diagonality: do opcodes route to position-aligned tiles?
    print("\n" + "-" * 60)
    print("OPCODE → TILE ALIGNMENT (Geometric Routing)")
    print("-" * 60)
    
    op_to_tiles = defaultdict(list)
    for tile, op_counts in tile_counts.items():
        for op, cnt in op_counts.items():
            if cnt > 50:
                op_to_tiles[op].append((tile, cnt))
    
    for op_idx, op in enumerate(OPCODES):
        tiles = sorted(op_to_tiles.get(op, []), key=lambda x: -x[1])
        if tiles:
            primary_tile = tiles[0][0]
            # Expected tile if perfectly diagonal: op_idx * num_tiles / num_ops
            expected_tile = op_idx * 16 / 8
            alignment = 1.0 - min(abs(primary_tile - expected_tile) / 8, 1.0)
            tiles_str = ', '.join([f"T{t}" for t, c in tiles[:2]])
            print(f"  {op:4s} (idx={op_idx}) → {tiles_str} | alignment={alignment:.2f}")


def verify_accuracy(model, device='cuda'):
    """Verify accuracy on fresh samples."""
    print("\n" + "=" * 60)
    print("VERIFICATION ON FRESH DATA")
    print("=" * 60)
    
    model.eval()
    
    errors_by_op = defaultdict(int)
    total_by_op = defaultdict(int)
    
    with torch.no_grad():
        for op in OPCODES:
            for _ in range(500):
                a = np.random.randint(0, 256)
                b = np.random.randint(0, 256) if op in ['ADC', 'AND', 'ORA', 'EOR'] else 0
                c = np.random.randint(0, 2) if op == 'ADC' else 0
                expected = compute_op(op, a, b, c)
                
                op_idx = torch.tensor([OP_TO_IDX[op]], device=device)
                a_t = torch.tensor([a], device=device)
                b_t = torch.tensor([b], device=device)
                c_t = torch.tensor([c], device=device)
                
                res_pred, _, _ = model(op_idx, a_t, b_t, c_t)
                pred = sum((res_pred[0, i] > 0.5).long().item() << i for i in range(8))
                
                total_by_op[op] += 1
                if pred != expected:
                    errors_by_op[op] += 1
    
    print("\nPer-operation accuracy:")
    total_correct = 0
    total_samples = 0
    for op in OPCODES:
        correct = total_by_op[op] - errors_by_op[op]
        acc = correct / total_by_op[op] * 100
        bar = '█' * int(acc / 5)
        print(f"  {op:4s}: {bar:20s} {acc:.1f}%")
        total_correct += correct
        total_samples += total_by_op[op]
    
    total_acc = total_correct / total_samples * 100
    print(f"\nOverall: {total_acc:.1f}%")
    return total_acc


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("TriX 6502 with GeometriX (v3)")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Generate data
    data = generate_data(n_per_op=1500)
    print(f"Dataset: {len(data):,} samples")
    
    # Create model
    model = TriX6502GeometriX(d_model=128, num_tiles=16)
    params = sum(p.numel() for p in model.parameters())
    ffn_params = model.ffn.get_param_count()
    print(f"Total parameters: {params:,}")
    print(f"FFN breakdown: {ffn_params}")
    
    # Train
    tile_counts = train(model, data, epochs=40, device=device)
    
    # Analyze
    analyze(model, tile_counts)
    
    # Verify
    acc = verify_accuracy(model, device)
    
    # Final summary
    print("\n" + "=" * 60)
    print("GEOMETRIX VERDICT")
    print("=" * 60)
    print(f"Final accuracy: {acc:.1f}%")
    
    stats = model.ffn.get_routing_stats()
    print(f"Active tiles: {stats['active_tiles']}/{stats['num_tiles']}")
    if stats.get('routing_cycles'):
        print(f"Routing cycles detected: {stats['routing_cycles']}")
