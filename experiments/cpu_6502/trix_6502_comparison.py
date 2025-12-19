#!/usr/bin/env python3
"""
TriX 6502: Vanilla vs GeometriX Comparison

Baseline: SparseLookupFFN (content-only routing)
Test: SparseLookupFFNv3 (geometric routing)

Same data, same training, same epochs. Pure A/B test.
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time

from trix.nn.sparse_lookup import SparseLookupFFN
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


def generate_data(n_per_op=1500):
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


class TriX6502Vanilla(nn.Module):
    """Vanilla TriX (content-only routing)."""
    
    def __init__(self, d_model=128, num_tiles=16):
        super().__init__()
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        self.input_proj = nn.Linear(32 + 17, d_model)
        
        self.ffn = SparseLookupFFN(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
        )
        
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )
    
    def forward(self, op_idx, a, b, c):
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
        
        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x).unsqueeze(1)
        
        out, info, aux = self.ffn(x)
        out = out.squeeze(1)
        
        result = self.result_head(out)
        return result, info, aux


class TriX6502GeometriX(nn.Module):
    """GeometriX (positional routing)."""
    
    def __init__(self, d_model=128, num_tiles=16):
        super().__init__()
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        self.input_proj = nn.Linear(32 + 17, d_model)
        
        self.ffn = SparseLookupFFNv3(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
            max_seq_len=len(OPCODES),
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
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
        
        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x).unsqueeze(1)
        
        # Position = opcode index
        positions = op_idx.float().unsqueeze(1)
        
        out, info, aux = self.ffn(x, positions=positions)
        out = out.squeeze(1)
        
        result = self.result_head(out)
        return result, info, aux


def train_and_evaluate(model, data, name, epochs=40, batch_size=256, device='cuda'):
    """Train model and return metrics."""
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
    
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
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
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}, acc={acc:.1f}%, purity={avg_purity:.2f}")
    
    train_time = time.time() - start_time
    
    # Final verification
    model.eval()
    errors_by_op = defaultdict(int)
    total_by_op = defaultdict(int)
    
    with torch.no_grad():
        for op in OPCODES:
            for _ in range(500):
                a_val = np.random.randint(0, 256)
                b_val = np.random.randint(0, 256) if op in ['ADC', 'AND', 'ORA', 'EOR'] else 0
                c_val = np.random.randint(0, 2) if op == 'ADC' else 0
                expected = compute_op(op, a_val, b_val, c_val)
                
                op_t = torch.tensor([OP_TO_IDX[op]], device=device)
                a_t = torch.tensor([a_val], device=device)
                b_t = torch.tensor([b_val], device=device)
                c_t = torch.tensor([c_val], device=device)
                
                res_pred, _, _ = model(op_t, a_t, b_t, c_t)
                pred = sum((res_pred[0, i] > 0.5).long().item() << i for i in range(8))
                
                total_by_op[op] += 1
                if pred != expected:
                    errors_by_op[op] += 1
    
    print(f"\nVerification (500 samples per op):")
    op_accs = {}
    for op in OPCODES:
        acc = (total_by_op[op] - errors_by_op[op]) / total_by_op[op] * 100
        op_accs[op] = acc
        bar = '█' * int(acc / 5)
        print(f"  {op:4s}: {bar:20s} {acc:.1f}%")
    
    total_acc = sum(total_by_op.values()) - sum(errors_by_op.values())
    total_acc = total_acc / sum(total_by_op.values()) * 100
    
    return {
        'name': name,
        'total_acc': total_acc,
        'op_accs': op_accs,
        'final_purity': avg_purity,
        'train_time': train_time,
        'tile_counts': dict(tile_counts),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("TriX 6502: VANILLA vs GEOMETRIX")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Same data for both
    data = generate_data(n_per_op=1500)
    print(f"Dataset: {len(data):,} samples ({len(OPCODES)} ops × 1500)")
    
    # Vanilla TriX
    vanilla = TriX6502Vanilla(d_model=128, num_tiles=16)
    vanilla_params = sum(p.numel() for p in vanilla.parameters())
    print(f"\nVanilla params: {vanilla_params:,}")
    
    # GeometriX
    geometrix = TriX6502GeometriX(d_model=128, num_tiles=16)
    geometrix_params = sum(p.numel() for p in geometrix.parameters())
    print(f"GeometriX params: {geometrix_params:,}")
    
    # Train both
    results_vanilla = train_and_evaluate(vanilla, data, "Vanilla TriX", device=device)
    results_geometrix = train_and_evaluate(geometrix, data, "GeometriX", device=device)
    
    # Comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'Vanilla':>12} {'GeometriX':>12} {'Delta':>12}")
    print("-" * 60)
    
    print(f"{'Overall Accuracy':<20} {results_vanilla['total_acc']:>11.1f}% {results_geometrix['total_acc']:>11.1f}% {results_geometrix['total_acc']-results_vanilla['total_acc']:>+11.1f}%")
    print(f"{'Tile Purity':<20} {results_vanilla['final_purity']:>12.2f} {results_geometrix['final_purity']:>12.2f} {results_geometrix['final_purity']-results_vanilla['final_purity']:>+12.2f}")
    print(f"{'Train Time (s)':<20} {results_vanilla['train_time']:>12.1f} {results_geometrix['train_time']:>12.1f} {results_geometrix['train_time']-results_vanilla['train_time']:>+12.1f}")
    
    print(f"\n{'Per-Op Accuracy:':<20}")
    print("-" * 60)
    for op in OPCODES:
        v_acc = results_vanilla['op_accs'][op]
        g_acc = results_geometrix['op_accs'][op]
        delta = g_acc - v_acc
        winner = "←" if delta < -1 else ("→" if delta > 1 else "=")
        print(f"  {op:<4} ({OP_TO_CAT[op]:<6}) {v_acc:>10.1f}% {g_acc:>10.1f}% {delta:>+10.1f}% {winner}")
    
    print("\n" + "=" * 60)
    if results_geometrix['total_acc'] > results_vanilla['total_acc'] + 1:
        print("VERDICT: GeometriX wins")
    elif results_vanilla['total_acc'] > results_geometrix['total_acc'] + 1:
        print("VERDICT: Vanilla wins")
    else:
        print("VERDICT: Tie (within 1%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
