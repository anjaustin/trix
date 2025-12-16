#!/usr/bin/env python3
"""
TriX 6502 with PROPER Training Data

Uses FLYNNCONCEIVABLE's exhaustive data generation:
- All 131,072 ALU combinations
- Edge case oversampling
- 50% C_in=1 (not random)
- Zero result oversampling
- Soroban encoding for ALU
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
from soroban import encode as soroban_encode, decode as soroban_decode
from training.data import (
    adc_truth, sbc_truth,
    asl_truth, lsr_truth, rol_truth, ror_truth,
    and_truth, ora_truth, eor_truth,
    inc_truth, dec_truth,
)


def generate_exhaustive_alu_data():
    """Generate ALL 131,072 ADC combinations with proper sampling."""
    data = []
    
    print("Generating exhaustive ALU data (131,072 combinations)...")
    
    # All combinations
    for a in range(256):
        for op in range(256):
            for c in [0, 1]:
                d = adc_truth(a, op, c)
                data.append({
                    'op': 'ADC',
                    'a': a, 'b': op, 'c': c,
                    'result': d['result'],
                    'n': d['n'], 'z': d['z'], 'c_out': d['c'], 'v': d['v']
                })
    
    # Zero result oversampling (VGem's fix) - 10x
    zero_cases = [d for d in data if d['result'] == 0]
    print(f"  Zero cases: {len(zero_cases)} (oversampling 10x)")
    data.extend(zero_cases * 9)
    
    print(f"  Total ALU samples: {len(data):,}")
    return data


def generate_exhaustive_logic_data():
    """Generate exhaustive LOGIC data."""
    data = []
    
    print("Generating exhaustive LOGIC data...")
    
    for a in range(256):
        for op in range(256):
            # AND
            d = and_truth(a, op)
            data.append({
                'op': 'AND', 'a': a, 'b': op, 'c': 0,
                'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': 0, 'v': 0
            })
            # ORA
            d = ora_truth(a, op)
            data.append({
                'op': 'ORA', 'a': a, 'b': op, 'c': 0,
                'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': 0, 'v': 0
            })
            # EOR
            d = eor_truth(a, op)
            data.append({
                'op': 'EOR', 'a': a, 'b': op, 'c': 0,
                'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': 0, 'v': 0
            })
    
    print(f"  Total LOGIC samples: {len(data):,}")
    return data


def generate_exhaustive_shift_data():
    """Generate exhaustive SHIFT data (all 1536 combinations)."""
    data = []
    
    print("Generating exhaustive SHIFT data (1,536 combinations)...")
    
    for val in range(256):
        # ASL
        d = asl_truth(val)
        data.append({
            'op': 'ASL', 'a': val, 'b': 0, 'c': 0,
            'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': d['c'], 'v': 0
        })
        # LSR
        d = lsr_truth(val)
        data.append({
            'op': 'LSR', 'a': val, 'b': 0, 'c': 0,
            'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': d['c'], 'v': 0
        })
        # ROL, ROR with carry
        for c in [0, 1]:
            d = rol_truth(val, c)
            data.append({
                'op': 'ROL', 'a': val, 'b': 0, 'c': c,
                'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': d['c'], 'v': 0
            })
            d = ror_truth(val, c)
            data.append({
                'op': 'ROR', 'a': val, 'b': 0, 'c': c,
                'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': d['c'], 'v': 0
            })
    
    print(f"  Total SHIFT samples: {len(data):,}")
    return data


def generate_exhaustive_incdec_data():
    """Generate exhaustive INC/DEC data (512 combinations)."""
    data = []
    
    print("Generating exhaustive INCDEC data (512 combinations)...")
    
    for val in range(256):
        d = inc_truth(val)
        data.append({
            'op': 'INC', 'a': val, 'b': 0, 'c': 0,
            'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': 0, 'v': 0
        })
        d = dec_truth(val)
        data.append({
            'op': 'DEC', 'a': val, 'b': 0, 'c': 0,
            'result': d['result'], 'n': d['n'], 'z': d['z'], 'c_out': 0, 'v': 0
        })
    
    print(f"  Total INCDEC samples: {len(data):,}")
    return data


# Operations
OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC']
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}
OP_TO_CAT = {
    'ADC': 'ALU',
    'AND': 'LOGIC', 'ORA': 'LOGIC', 'EOR': 'LOGIC',
    'ASL': 'SHIFT', 'LSR': 'SHIFT', 'ROL': 'SHIFT', 'ROR': 'SHIFT',
    'INC': 'INCDEC', 'DEC': 'INCDEC',
}


class TriX6502Soroban(nn.Module):
    """
    TriX 6502 with Soroban encoding for ALU operations.
    
    Soroban (thermometer) encoding makes carry propagation visible.
    """
    
    def __init__(self, d_model=128, num_tiles=16):
        super().__init__()
        
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        
        # Input projection
        # Soroban: 32 dims for A, 32 dims for B (4 rods × 8 positions each)
        # Binary: 8 + 8 = 16 dims
        # We'll use hybrid: 32 (soroban A) + 32 (soroban B) + 1 (c) + 32 (op) = 97
        self.input_proj = nn.Linear(32 + 64 + 1, d_model)
        
        self.ffn = SparseLookupFFNv2(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
            ternary_weight=0.01,
            sparsity_weight=0.005,
            diversity_weight=0.01,
        )
        
        # Output: Soroban-encoded result (32 dims) + flags (4 dims)
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
        )
        self.flag_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )
    
    def forward(self, op_idx, a, b, c, labels=None):
        B = op_idx.shape[0]
        device = op_idx.device
        
        op_emb = self.op_embed(op_idx)
        
        # Soroban encode A and B
        a_sor = soroban_encode(a)  # (B, 32)
        b_sor = soroban_encode(b)  # (B, 32)
        
        x = torch.cat([op_emb, a_sor, b_sor, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x).unsqueeze(1)
        
        out, info, aux = self.ffn(x, labels=labels)
        out = out.squeeze(1)
        
        result_sor = self.result_head(out)
        flags = self.flag_head(out)
        
        return result_sor, flags, info, aux


def train(model, data, epochs=50, batch_size=1024, device='cuda'):
    """Train with proper data."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    
    # Prepare tensors
    op_idx = torch.tensor([OP_TO_IDX[d['op']] for d in data], device=device)
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    c = torch.tensor([d['c'] for d in data], device=device)
    result = torch.tensor([d['result'] for d in data], device=device)
    n = torch.tensor([d['n'] for d in data], dtype=torch.float32, device=device)
    z = torch.tensor([d['z'] for d in data], dtype=torch.float32, device=device)
    c_out = torch.tensor([d['c_out'] for d in data], dtype=torch.float32, device=device)
    v = torch.tensor([d['v'] for d in data], dtype=torch.float32, device=device)
    
    # Soroban encode targets
    result_sor = soroban_encode(result)
    flags_t = torch.stack([n, z, c_out, v], dim=1)
    
    ops = [d['op'] for d in data]
    
    print(f"\nTraining on {len(data):,} samples, {epochs} epochs")
    print("-" * 70)
    
    for epoch in range(epochs):
        model.train()
        model.ffn.reset_claim_tracking()
        perm = torch.randperm(len(data), device=device)
        total_loss, correct = 0, 0
        tile_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(0, len(data) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            
            res_sor_pred, flags_pred, info, aux = model(
                op_idx[idx], a[idx], b[idx], c[idx], labels=op_idx[idx]
            )
            
            # Loss with extra weight on Z flag
            result_loss = F.binary_cross_entropy(res_sor_pred, result_sor[idx])
            flag_loss = F.binary_cross_entropy(flags_pred, flags_t[idx])
            z_loss = F.binary_cross_entropy(flags_pred[:, 1], z[idx]) * 3.0  # Extra Z weight
            
            loss = result_loss + flag_loss + z_loss + aux['total_aux']
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            # Decode and check accuracy
            pred_result = soroban_decode(res_sor_pred)
            # Apply Z flag override (VGem's trick)
            pred_result[flags_pred[:, 1] > 0.5] = 0
            correct += (pred_result == result[idx]).sum().item()
            
            # Track tiles
            tiles = info['tile_idx'].squeeze(-1).cpu().numpy()
            for j, t in enumerate(tiles):
                tile_counts[int(t)][ops[idx[j].item()]] += 1
        
        scheduler.step()
        
        acc = correct / len(data) * 100
        avg_loss = total_loss / (len(data) // batch_size)
        
        # Category purity
        purities = []
        for tile, op_counts in tile_counts.items():
            cat_counts = defaultdict(int)
            for op, cnt in op_counts.items():
                cat_counts[OP_TO_CAT[op]] += cnt
            total = sum(cat_counts.values())
            if total > 100:
                purities.append(max(cat_counts.values()) / total)
        avg_purity = np.mean(purities) if purities else 0
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}, acc={acc:.2f}%, purity={avg_purity:.2f}")
    
    return tile_counts


def verify_exhaustive(model, device='cuda'):
    """Verify on ALL combinations (exhaustive test)."""
    print("\n" + "=" * 70)
    print("EXHAUSTIVE VERIFICATION")
    print("=" * 70)
    
    model.eval()
    
    errors_by_op = defaultdict(int)
    total_by_op = defaultdict(int)
    
    with torch.no_grad():
        # ALU: All 131,072
        print("\nTesting ADC (131,072 combinations)...")
        for a in range(256):
            for op in range(256):
                for c in [0, 1]:
                    expected = (a + op + c) & 0xFF
                    
                    op_idx = torch.tensor([OP_TO_IDX['ADC']], device=device)
                    a_t = torch.tensor([a], device=device)
                    b_t = torch.tensor([op], device=device)
                    c_t = torch.tensor([c], device=device)
                    
                    res_sor, flags, _, _ = model(op_idx, a_t, b_t, c_t)
                    pred = soroban_decode(res_sor)[0].item()
                    if flags[0, 1] > 0.5:  # Z flag override
                        pred = 0
                    
                    total_by_op['ADC'] += 1
                    if pred != expected:
                        errors_by_op['ADC'] += 1
            
            if (a + 1) % 64 == 0:
                print(f"  Progress: {a+1}/256")
        
        # SHIFT: All 1,536
        print("\nTesting SHIFT (1,536 combinations)...")
        for val in range(256):
            for op_name, expected in [
                ('ASL', (val << 1) & 0xFF),
                ('LSR', val >> 1),
            ]:
                op_idx = torch.tensor([OP_TO_IDX[op_name]], device=device)
                a_t = torch.tensor([val], device=device)
                b_t = torch.tensor([0], device=device)
                c_t = torch.tensor([0], device=device)
                
                res_sor, flags, _, _ = model(op_idx, a_t, b_t, c_t)
                pred = soroban_decode(res_sor)[0].item()
                if flags[0, 1] > 0.5:
                    pred = 0
                
                total_by_op[op_name] += 1
                if pred != expected:
                    errors_by_op[op_name] += 1
        
        # LOGIC: Sample 10,000
        print("\nTesting LOGIC (10,000 samples)...")
        for _ in range(10000):
            a = np.random.randint(0, 256)
            op = np.random.randint(0, 256)
            
            for op_name, expected in [
                ('AND', a & op),
                ('ORA', a | op),
                ('EOR', a ^ op),
            ]:
                op_idx = torch.tensor([OP_TO_IDX[op_name]], device=device)
                a_t = torch.tensor([a], device=device)
                b_t = torch.tensor([op], device=device)
                c_t = torch.tensor([0], device=device)
                
                res_sor, flags, _, _ = model(op_idx, a_t, b_t, c_t)
                pred = soroban_decode(res_sor)[0].item()
                if flags[0, 1] > 0.5:
                    pred = 0
                
                total_by_op[op_name] += 1
                if pred != expected:
                    errors_by_op[op_name] += 1
        
        # INCDEC: All 512
        print("\nTesting INCDEC (512 combinations)...")
        for val in range(256):
            for op_name, expected in [
                ('INC', (val + 1) & 0xFF),
                ('DEC', (val - 1) & 0xFF),
            ]:
                op_idx = torch.tensor([OP_TO_IDX[op_name]], device=device)
                a_t = torch.tensor([val], device=device)
                b_t = torch.tensor([0], device=device)
                c_t = torch.tensor([0], device=device)
                
                res_sor, flags, _, _ = model(op_idx, a_t, b_t, c_t)
                pred = soroban_decode(res_sor)[0].item()
                if flags[0, 1] > 0.5:
                    pred = 0
                
                total_by_op[op_name] += 1
                if pred != expected:
                    errors_by_op[op_name] += 1
    
    # Results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    
    for op in OPCODES:
        if total_by_op[op] > 0:
            errors = errors_by_op[op]
            total = total_by_op[op]
            acc = (total - errors) / total * 100
            status = "✓" if errors == 0 else f"✗ ({errors} errors)"
            print(f"  {op:4s}: {acc:6.2f}% {status}")
    
    total_errors = sum(errors_by_op.values())
    total_tests = sum(total_by_op.values())
    overall_acc = (total_tests - total_errors) / total_tests * 100
    print(f"\n  OVERALL: {overall_acc:.2f}% ({total_errors} errors / {total_tests:,} tests)")
    
    return errors_by_op


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Generate exhaustive data
    print("\n" + "=" * 70)
    print("GENERATING EXHAUSTIVE DATA")
    print("=" * 70)
    
    data = []
    data.extend(generate_exhaustive_alu_data())
    data.extend(generate_exhaustive_logic_data())
    data.extend(generate_exhaustive_shift_data())
    data.extend(generate_exhaustive_incdec_data())
    
    np.random.shuffle(data)
    print(f"\nTotal dataset: {len(data):,} samples")
    
    # Count by category
    cat_counts = defaultdict(int)
    for d in data:
        cat_counts[OP_TO_CAT[d['op']]] += 1
    print("\nBy category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count:,}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = TriX6502Soroban(d_model=128, num_tiles=16)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    tile_counts = train(model, data, epochs=50, batch_size=2048, device=device)
    
    # Verify
    errors = verify_exhaustive(model, device)
