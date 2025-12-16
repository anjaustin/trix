#!/usr/bin/env python3
"""
Quick TriX 6502 Experiment - Streamlined for fast iteration.
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
from training.data import adc_truth, and_truth, ora_truth, eor_truth, asl_truth, lsr_truth, inc_truth, dec_truth

# Simplified categories
OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
OP_TO_CAT = {
    'ADC': 'ALU', 
    'AND': 'LOGIC', 'ORA': 'LOGIC', 'EOR': 'LOGIC',
    'ASL': 'SHIFT', 'LSR': 'SHIFT',
    'INC': 'INCDEC', 'DEC': 'INCDEC',
}


def generate_data(samples_per_op=2000):
    """Generate balanced dataset."""
    data = []
    
    for _ in range(samples_per_op):
        a, op = np.random.randint(0, 256, 2)
        c = np.random.randint(0, 2)
        
        # ADC
        d = adc_truth(a, op, c)
        data.append({'op': 'ADC', 'a': a, 'b': op, 'c': c, 'result': d['result'], 'n': d['n'], 'z': d['z']})
        
        # LOGIC
        data.append({'op': 'AND', 'a': a, 'b': op, 'c': 0, 'result': a & op, 'n': ((a & op) >> 7) & 1, 'z': int((a & op) == 0)})
        data.append({'op': 'ORA', 'a': a, 'b': op, 'c': 0, 'result': a | op, 'n': ((a | op) >> 7) & 1, 'z': int((a | op) == 0)})
        data.append({'op': 'EOR', 'a': a, 'b': op, 'c': 0, 'result': a ^ op, 'n': ((a ^ op) >> 7) & 1, 'z': int((a ^ op) == 0)})
        
        # SHIFT
        data.append({'op': 'ASL', 'a': a, 'b': 0, 'c': 0, 'result': (a << 1) & 0xFF, 'n': ((a << 1) >> 7) & 1, 'z': int(((a << 1) & 0xFF) == 0)})
        data.append({'op': 'LSR', 'a': a, 'b': 0, 'c': 0, 'result': a >> 1, 'n': 0, 'z': int((a >> 1) == 0)})
        
        # INCDEC
        data.append({'op': 'INC', 'a': a, 'b': 0, 'c': 0, 'result': (a + 1) & 0xFF, 'n': (((a + 1) & 0xFF) >> 7) & 1, 'z': int(((a + 1) & 0xFF) == 0)})
        data.append({'op': 'DEC', 'a': a, 'b': 0, 'c': 0, 'result': (a - 1) & 0xFF, 'n': (((a - 1) & 0xFF) >> 7) & 1, 'z': int(((a - 1) & 0xFF) == 0)})
    
    np.random.shuffle(data)
    return data


class TriX6502(nn.Module):
    def __init__(self, d_model=64, num_tiles=12, use_v2=False, **kwargs):
        super().__init__()
        self.op_embed = nn.Embedding(len(OPCODES), 16)
        self.input_proj = nn.Linear(16 + 17, d_model)
        
        FFN = SparseLookupFFNv2 if use_v2 else SparseLookupFFN
        self.ffn = FFN(d_model=d_model, num_tiles=num_tiles, tiles_per_cluster=4, **kwargs)
        
        self.result_head = nn.Linear(d_model, 8)
        self.flag_head = nn.Linear(d_model, 2)
    
    def forward(self, op_idx, a, b, c, labels=None):
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x).unsqueeze(1)
        
        out, info, aux = self.ffn(x, labels=labels) if hasattr(self.ffn, 'reset_claim_tracking') else (self.ffn(x)[0], self.ffn(x)[1], self.ffn(x)[2])
        out = out.squeeze(1)
        
        return torch.sigmoid(self.result_head(out)), torch.sigmoid(self.flag_head(out)), info, aux


def train(model, data, epochs=20, device='cuda'):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    
    op_map = {op: i for i, op in enumerate(OPCODES)}
    op_idx = torch.tensor([op_map[d['op']] for d in data], device=device)
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    c = torch.tensor([d['c'] for d in data], device=device)
    result = torch.tensor([d['result'] for d in data], device=device)
    
    result_bits = torch.stack([(result >> i) & 1 for i in range(8)], dim=1).float()
    
    ops = [d['op'] for d in data]
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(data), device=device)
        total_loss, correct = 0, 0
        tile_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(0, len(data) - 256, 256):
            idx = perm[i:i+256]
            res_pred, flag_pred, info, aux = model(op_idx[idx], a[idx], b[idx], c[idx])
            
            loss = F.binary_cross_entropy(res_pred, result_bits[idx]) + aux.get('total_aux', torch.tensor(0.0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            pred = sum((res_pred[:, i] > 0.5).long() << i for i in range(8))
            correct += (pred == result[idx]).sum().item()
            
            tiles = info['tile_idx'].squeeze(-1).cpu().numpy()
            for j, t in enumerate(tiles):
                tile_counts[int(t)][ops[idx[j].item()]] += 1
        
        # Compute purity per category
        cat_purity = {}
        for tile, op_counts in tile_counts.items():
            cat_counts = defaultdict(int)
            for op, cnt in op_counts.items():
                cat_counts[OP_TO_CAT[op]] += cnt
            total = sum(cat_counts.values())
            if total > 0:
                cat_purity[tile] = max(cat_counts.values()) / total
        
        avg_purity = np.mean(list(cat_purity.values())) if cat_purity else 0
        acc = correct / len(data) * 100
        print(f"Epoch {epoch+1:2d}: loss={total_loss/(len(data)//256):.4f}, acc={acc:.1f}%, purity={avg_purity:.2f}")
    
    return tile_counts


def analyze(tile_counts):
    print("\n" + "=" * 60)
    print("TILE SPECIALIZATION ANALYSIS")
    print("=" * 60)
    
    for tile in sorted(tile_counts.keys()):
        op_counts = tile_counts[tile]
        total = sum(op_counts.values())
        if total < 100:
            continue
        
        # Category breakdown
        cat_counts = defaultdict(int)
        for op, cnt in op_counts.items():
            cat_counts[OP_TO_CAT[op]] += cnt
        
        dominant = max(cat_counts, key=cat_counts.get)
        purity = cat_counts[dominant] / total
        
        top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:3]
        top_str = ', '.join([f"{op}:{cnt}" for op, cnt in top_ops])
        
        print(f"Tile {tile:2d}: {dominant:6s} ({purity:.0%}) | {top_str}")
    
    # Summary
    print("\n" + "-" * 60)
    cat_tiles = defaultdict(list)
    for tile, op_counts in tile_counts.items():
        cat_counts = defaultdict(int)
        for op, cnt in op_counts.items():
            cat_counts[OP_TO_CAT[op]] += cnt
        if sum(cat_counts.values()) > 100:
            dominant = max(cat_counts, key=cat_counts.get)
            purity = cat_counts[dominant] / sum(cat_counts.values())
            if purity > 0.5:
                cat_tiles[dominant].append((tile, purity))
    
    print("\nTiles by category (purity > 50%):")
    for cat in ['ALU', 'LOGIC', 'SHIFT', 'INCDEC']:
        tiles = cat_tiles.get(cat, [])
        if tiles:
            tile_str = ', '.join([f"{t}({p:.0%})" for t, p in tiles])
            print(f"  {cat}: {tile_str}")
        else:
            print(f"  {cat}: none")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--tiles", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"TriX 6502 Quick Experiment | {'v2' if args.v2 else 'v1'} | {args.tiles} tiles")
    print("=" * 60)
    
    data = generate_data(2000)
    print(f"Samples: {len(data):,}")
    
    kwargs = {'ternary_weight': 0.01, 'sparsity_weight': 0.01, 'diversity_weight': 0.01} if args.v2 else {}
    model = TriX6502(d_model=64, num_tiles=args.tiles, use_v2=args.v2, **kwargs)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    tile_counts = train(model, data, epochs=args.epochs)
    analyze(tile_counts)
