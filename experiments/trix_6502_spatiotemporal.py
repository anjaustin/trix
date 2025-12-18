#!/usr/bin/env python3
"""
TriX 6502: SpatioTemporal Test

The key insight: ADC fails because it needs TEMPORAL state (carry flag).
GeometriX adds SPATIAL routing. v4 adds TEMPORAL routing.

Combined: Content × Position × State

This should fix ADC.
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
from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4

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
    """Vanilla TriX (content-only)."""
    
    def __init__(self, d_model=128, num_tiles=16, num_layers=2):
        super().__init__()
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        self.input_proj = nn.Linear(32 + 17, d_model)
        
        self.ffn_layers = nn.ModuleList([
            SparseLookupFFN(
                d_model=d_model,
                num_tiles=num_tiles,
                tiles_per_cluster=4,
            ) for _ in range(num_layers)
        ])
        
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
        
        total_aux = {'total_aux': 0.0}
        info = None
        for ffn in self.ffn_layers:
            x, info, aux = ffn(x)
            total_aux['total_aux'] = total_aux['total_aux'] + aux['total_aux']
        
        result = self.result_head(x.squeeze(1))
        return result, info, total_aux


class TriX6502GeometriX(nn.Module):
    """GeometriX (content + spatial)."""
    
    def __init__(self, d_model=128, num_tiles=16, num_layers=2):
        super().__init__()
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        self.input_proj = nn.Linear(32 + 17, d_model)
        
        self.ffn_layers = nn.ModuleList([
            SparseLookupFFNv3(
                d_model=d_model,
                num_tiles=num_tiles,
                tiles_per_cluster=4,
                max_seq_len=len(OPCODES),
                position_spread=1.5,
                use_gauge=False,
                use_vortex=False,
                track_topology=False,
            ) for _ in range(num_layers)
        ])
        
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
        positions = op_idx.float().unsqueeze(1)
        
        total_aux = {'total_aux': 0.0}
        info = None
        for ffn in self.ffn_layers:
            x, info, aux = ffn(x, positions=positions)
            total_aux['total_aux'] = total_aux['total_aux'] + aux['total_aux']
        
        result = self.result_head(x.squeeze(1))
        return result, info, total_aux


class TriX6502SpatioTemporal(nn.Module):
    """SpatioTemporal (content + spatial + temporal state)."""
    
    def __init__(self, d_model=128, num_tiles=16, num_layers=2):
        super().__init__()
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        self.input_proj = nn.Linear(32 + 17, d_model)
        
        self.ffn_layers = nn.ModuleList([
            SparseLookupFFNv4(
                d_model=d_model,
                num_tiles=num_tiles,
                tiles_per_cluster=4,
                num_states=2,  # carry=0, carry=1
                state_dim=16,
                max_seq_len=len(OPCODES),
                position_spread=1.5,
            ) for _ in range(num_layers)
        ])
        
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
        positions = op_idx.float().unsqueeze(1)
        states = c.unsqueeze(1)  # Carry flag IS the temporal state
        
        total_aux = {'total_aux': 0.0}
        info = None
        for ffn in self.ffn_layers:
            x, info, aux = ffn(x, positions=positions, states=states)
            total_aux['total_aux'] = total_aux['total_aux'] + aux['total_aux']
        
        result = self.result_head(x.squeeze(1))
        return result, info, total_aux


def train_and_evaluate(model, data, name, epochs=40, batch_size=256, device='cuda'):
    """Train and evaluate model."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare tensors
    op_idx = torch.tensor([OP_TO_IDX[d['op']] for d in data], device=device)
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    c = torch.tensor([d['c'] for d in data], device=device)
    result = torch.tensor([d['result'] for d in data], device=device)
    result_bits = torch.stack([(result >> i) & 1 for i in range(8)], dim=1).float()
    
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(data), device=device)
        total_loss, correct = 0, 0
        
        for i in range(0, len(data) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            
            res_pred, info, aux = model(op_idx[idx], a[idx], b[idx], c[idx])
            loss = F.binary_cross_entropy(res_pred, result_bits[idx]) + aux['total_aux']
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pred = sum((res_pred[:, k] > 0.5).long() << k for k in range(8))
            correct += (pred == result[idx]).sum().item()
        
        acc = correct / len(data) * 100
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: loss={total_loss/(len(data)//batch_size):.4f}, acc={acc:.1f}%")
    
    train_time = time.time() - start
    
    # Verification
    model.eval()
    op_results = {}
    
    with torch.no_grad():
        for op in OPCODES:
            correct = 0
            total = 500
            
            # Test both carry states for ADC
            for _ in range(total):
                a_val = np.random.randint(0, 256)
                b_val = np.random.randint(0, 256) if op in ['ADC', 'AND', 'ORA', 'EOR'] else 0
                c_val = np.random.randint(0, 2) if op == 'ADC' else 0
                expected = compute_op(op, a_val, b_val, c_val)
                
                op_t = torch.tensor([OP_TO_IDX[op]], device=device)
                a_t = torch.tensor([a_val], device=device)
                b_t = torch.tensor([b_val], device=device)
                c_t = torch.tensor([c_val], device=device)
                
                res_pred, _, _ = model(op_t, a_t, b_t, c_t)
                pred = sum((res_pred[0, k] > 0.5).long().item() << k for k in range(8))
                
                if pred == expected:
                    correct += 1
            
            op_results[op] = correct / total * 100
    
    # ADC breakdown by carry state
    adc_c0, adc_c1 = 0, 0
    adc_c0_total, adc_c1_total = 0, 0
    
    with torch.no_grad():
        for c_val in [0, 1]:
            for _ in range(250):
                a_val = np.random.randint(0, 256)
                b_val = np.random.randint(0, 256)
                expected = compute_op('ADC', a_val, b_val, c_val)
                
                op_t = torch.tensor([OP_TO_IDX['ADC']], device=device)
                a_t = torch.tensor([a_val], device=device)
                b_t = torch.tensor([b_val], device=device)
                c_t = torch.tensor([c_val], device=device)
                
                res_pred, _, _ = model(op_t, a_t, b_t, c_t)
                pred = sum((res_pred[0, k] > 0.5).long().item() << k for k in range(8))
                
                if c_val == 0:
                    adc_c0_total += 1
                    if pred == expected:
                        adc_c0 += 1
                else:
                    adc_c1_total += 1
                    if pred == expected:
                        adc_c1 += 1
    
    op_results['ADC_C0'] = adc_c0 / adc_c0_total * 100
    op_results['ADC_C1'] = adc_c1 / adc_c1_total * 100
    
    total_acc = sum(op_results[op] for op in OPCODES) / len(OPCODES)
    
    print(f"\nVerification:")
    for op in OPCODES:
        bar = '█' * int(op_results[op] / 5)
        extra = ""
        if op == 'ADC':
            extra = f" [C=0: {op_results['ADC_C0']:.0f}%, C=1: {op_results['ADC_C1']:.0f}%]"
        print(f"  {op:4s}: {bar:20s} {op_results[op]:.1f}%{extra}")
    
    print(f"\nOverall: {total_acc:.1f}%, Time: {train_time:.1f}s")
    
    return {
        'name': name,
        'total_acc': total_acc,
        'op_results': op_results,
        'train_time': train_time,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("TriX 6502: VANILLA vs GEOMETRIX vs SPATIOTEMPORAL")
    print("=" * 60)
    print(f"Device: {device}")
    print("\nThe Test: Can SpatioTemporal fix ADC by routing on carry state?")
    
    # Same data for all
    data = generate_data(n_per_op=1500)
    print(f"Dataset: {len(data):,} samples")
    
    # Models
    num_layers = 3
    models = [
        ("Vanilla (Content)", TriX6502Vanilla),
        ("GeometriX (Content+Spatial)", TriX6502GeometriX),
        ("SpatioTemporal (Content+Spatial+State)", TriX6502SpatioTemporal),
    ]
    
    results = []
    for name, ModelClass in models:
        model = ModelClass(d_model=128, num_tiles=16, num_layers=num_layers)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}: {params:,} params")
        
        result = train_and_evaluate(model, data, name, epochs=40, device=device)
        results.append(result)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Model':<40} {'Overall':>10} {'ADC':>10} {'ADC_C0':>10} {'ADC_C1':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<40} {r['total_acc']:>9.1f}% {r['op_results']['ADC']:>9.1f}% "
              f"{r['op_results']['ADC_C0']:>9.1f}% {r['op_results']['ADC_C1']:>9.1f}%")
    
    print("\n" + "=" * 60)
    
    # Did SpatioTemporal fix ADC?
    vanilla_adc = results[0]['op_results']['ADC']
    spatiotemporal_adc = results[2]['op_results']['ADC']
    
    if spatiotemporal_adc > vanilla_adc + 10:
        print(f"VERDICT: SpatioTemporal FIXES ADC! ({vanilla_adc:.0f}% → {spatiotemporal_adc:.0f}%)")
    elif spatiotemporal_adc > vanilla_adc + 5:
        print(f"VERDICT: SpatioTemporal improves ADC ({vanilla_adc:.0f}% → {spatiotemporal_adc:.0f}%)")
    else:
        print(f"VERDICT: No significant ADC improvement ({vanilla_adc:.0f}% → {spatiotemporal_adc:.0f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
