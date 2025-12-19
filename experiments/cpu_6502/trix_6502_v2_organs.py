#!/usr/bin/env python3
"""
TriX v2 6502 Organ Discovery

Full experiment:
1. Train TriX v2 on actual 6502 operation prediction
2. Use claim tracking to see operation → tile mapping
3. Measure accuracy AND specialization together
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/flynnconceivable')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from trix.nn import SparseLookupFFNv2
from training.data import adc_truth

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


class TriX6502v2(nn.Module):
    """TriX v2 for 6502 with larger capacity."""
    
    def __init__(self, d_model=128, num_tiles=16):
        super().__init__()
        
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        self.input_proj = nn.Linear(32 + 17, d_model)
        
        self.ffn = SparseLookupFFNv2(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
            ternary_weight=0.01,
            sparsity_weight=0.005,
            diversity_weight=0.01,
        )
        
        # Larger output head
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )
    
    def forward(self, op_idx, a, b, c, labels=None):
        B = op_idx.shape[0]
        
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
        
        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x).unsqueeze(1)
        
        out, info, aux = self.ffn(x, labels=labels)
        out = out.squeeze(1)
        
        result = self.result_head(out)
        return result, info, aux


def train(model, data, epochs=40, batch_size=256, device='cuda'):
    """Train with claim tracking."""
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
        model.ffn.reset_claim_tracking()
        perm = torch.randperm(len(data), device=device)
        total_loss, correct = 0, 0
        tile_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(0, len(data) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            
            # Use op_idx as labels for claim tracking
            res_pred, info, aux = model(op_idx[idx], a[idx], b[idx], c[idx], labels=op_idx[idx])
            
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
        
        # Island stats
        stats = model.ffn.get_island_stats()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}, acc={acc:.1f}%, "
                  f"purity={avg_purity:.2f}, ternary={stats['ternary_fraction']:.2f}, "
                  f"sparsity={stats['sparsity']:.2f}")
    
    return tile_counts


def analyze(model, tile_counts):
    """Detailed analysis."""
    print("\n" + "=" * 60)
    print("TILE SPECIALIZATION ANALYSIS")
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
        
        # Signature analysis
        sig_analysis = model.ffn.get_signature_analysis(tile)
        n_pos = len(sig_analysis['positive_dims'])
        n_neg = len(sig_analysis['negative_dims'])
        n_zero = sig_analysis['zero_count']
        
        bar = '█' * int(purity * 20) + '░' * (20 - int(purity * 20))
        print(f"Tile {tile:2d}: {dominant_cat:6s} [{bar}] {purity:.0%}")
        print(f"         sig: +{n_pos} -{n_neg} 0:{n_zero} | {top_str}")
    
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
    
    # Per-operation claim rates from v2 tracking
    print("\n" + "-" * 60)
    print("CLAIM RATES (from v2 tracking)")
    print("-" * 60)
    
    for tile in active_tiles[:6]:  # Top 6 tiles
        print(f"\nTile {tile}:")
        for op_idx, op in enumerate(OPCODES):
            rate = model.ffn.get_claim_rate(tile, op_idx)
            if rate > 0.05:
                bar = '▓' * int(rate * 20)
                print(f"  {op:4s}: {bar:20s} {rate:.0%}")


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
    for op in OPCODES:
        acc = (total_by_op[op] - errors_by_op[op]) / total_by_op[op] * 100
        bar = '█' * int(acc / 5)
        print(f"  {op:4s}: {bar:20s} {acc:.1f}%")
    
    total_acc = sum(total_by_op.values()) - sum(errors_by_op.values())
    total_acc = total_acc / sum(total_by_op.values()) * 100
    print(f"\nOverall: {total_acc:.1f}%")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Generate data
    data = generate_data(n_per_op=1500)
    print(f"Dataset: {len(data):,} samples")
    
    # Create model
    model = TriX6502v2(d_model=128, num_tiles=16)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    tile_counts = train(model, data, epochs=40, device=device)
    
    # Analyze
    analyze(model, tile_counts)
    
    # Verify
    verify_accuracy(model, device)
