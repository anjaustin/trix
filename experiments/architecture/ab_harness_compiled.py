#!/usr/bin/env python3
"""
A/B Harness: Dynamic vs Compiled Dispatch

Compare baseline (dynamic routing) against candidate (compiled dispatch)
on the same inputs. Measure:
  1. Agreement rate (do they produce the same output?)
  2. Accuracy delta (if we have labels)
  3. Compiled hit rate (how often compiled path is used)
  4. Worst disagreements (where do they diverge?)

Usage:
    python experiments/ab_harness_compiled.py
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/flynnconceivable')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

from trix.nn import SparseLookupFFNv2, CompiledDispatch
from training.data import adc_truth


# =============================================================================
# Data Generation (6502 operations)
# =============================================================================

OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}

def compute_op(op, a, b, c):
    """Ground truth computation."""
    if op == 'ADC': return (a + b + c) & 0xFF
    elif op == 'AND': return a & b
    elif op == 'ORA': return a | b
    elif op == 'EOR': return a ^ b
    elif op == 'ASL': return (a << 1) & 0xFF
    elif op == 'LSR': return a >> 1
    elif op == 'INC': return (a + 1) & 0xFF
    elif op == 'DEC': return (a - 1) & 0xFF

def generate_test_data(n_per_op=200):
    """Generate balanced test dataset."""
    data = []
    for op in OPCODES:
        for _ in range(n_per_op):
            a = np.random.randint(0, 256)
            b = np.random.randint(0, 256) if op in ['ADC', 'AND', 'ORA', 'EOR'] else 0
            c = np.random.randint(0, 2) if op == 'ADC' else 0
            result = compute_op(op, a, b, c)
            data.append({
                'op': op, 'op_idx': OP_TO_IDX[op],
                'a': a, 'b': b, 'c': c,
                'result': result
            })
    np.random.shuffle(data)
    return data


# =============================================================================
# Model
# =============================================================================

class TriX6502(nn.Module):
    """Simple TriX model for 6502 operations."""
    
    def __init__(self, d_model=64, num_tiles=12):
        super().__init__()
        self.op_embed = nn.Embedding(len(OPCODES), 16)
        self.input_proj = nn.Linear(16 + 17, d_model)
        self.ffn = SparseLookupFFNv2(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
            ternary_weight=0.01,
            sparsity_weight=0.005,
        )
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
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
        
        result_bits = self.result_head(out)
        return result_bits, info, aux
    
    def predict(self, op_idx, a, b, c):
        """Get integer prediction."""
        result_bits, info, _ = self(op_idx, a, b, c)
        pred = sum((result_bits[:, i] > 0.5).long() << i for i in range(8))
        return pred, info


# =============================================================================
# A/B Harness
# =============================================================================

@dataclass
class ABStats:
    """Statistics from A/B comparison."""
    n: int = 0
    agree: int = 0
    compiled_used: int = 0
    correct_dynamic: int = 0
    correct_compiled: int = 0
    disagreements: List[Dict] = None
    
    def __post_init__(self):
        if self.disagreements is None:
            self.disagreements = []
    
    @property
    def agree_rate(self) -> float:
        return self.agree / self.n if self.n > 0 else 0
    
    @property
    def compiled_rate(self) -> float:
        return self.compiled_used / self.n if self.n > 0 else 0
    
    @property
    def accuracy_dynamic(self) -> float:
        return self.correct_dynamic / self.n if self.n > 0 else 0
    
    @property
    def accuracy_compiled(self) -> float:
        return self.correct_compiled / self.n if self.n > 0 else 0
    
    @property
    def accuracy_delta(self) -> float:
        return self.accuracy_compiled - self.accuracy_dynamic


def eval_ab(
    model: TriX6502,
    compiler: CompiledDispatch,
    data: List[Dict],
    device: str = 'cuda',
    confidence: float = 0.9,
    max_disagreements: int = 20,
) -> ABStats:
    """
    A/B comparison: dynamic routing vs compiled dispatch.
    
    Args:
        model: The base TriX model
        compiler: CompiledDispatch wrapper around model.ffn
        data: Test data with op, a, b, c, result
        device: Device to run on
        confidence: Confidence threshold for compiled dispatch
        max_disagreements: Max disagreements to record
    
    Returns:
        ABStats with comparison metrics
    """
    model.eval()
    stats = ABStats()
    
    with torch.no_grad():
        for sample in data:
            op_idx = torch.tensor([sample['op_idx']], device=device)
            a = torch.tensor([sample['a']], device=device)
            b = torch.tensor([sample['b']], device=device)
            c = torch.tensor([sample['c']], device=device)
            target = sample['result']
            
            # Dynamic routing (baseline)
            pred_dyn, info_dyn = model.predict(op_idx, a, b, c)
            pred_dyn = pred_dyn.item()
            
            # Compiled dispatch (candidate)
            # We need to go through the compiler for the FFN part
            op_emb = model.op_embed(op_idx)
            a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
            b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
            x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
            x = model.input_proj(x).unsqueeze(1)
            
            # Use compiler instead of ffn directly
            out_cmp, info_cmp, _ = compiler.forward(
                x, 
                class_hint=sample['op_idx'],
                confidence=confidence
            )
            out_cmp = out_cmp.squeeze(1)
            result_bits_cmp = model.result_head(out_cmp)
            pred_cmp = sum((result_bits_cmp[0, i] > 0.5).long().item() << i for i in range(8))
            
            # Update stats
            stats.n += 1
            
            if pred_dyn == pred_cmp:
                stats.agree += 1
            elif len(stats.disagreements) < max_disagreements:
                stats.disagreements.append({
                    'op': sample['op'],
                    'a': sample['a'],
                    'b': sample['b'],
                    'c': sample['c'],
                    'target': target,
                    'pred_dynamic': pred_dyn,
                    'pred_compiled': pred_cmp,
                    'compiled_used': info_cmp.get('compiled', False),
                })
            
            if info_cmp.get('compiled', False):
                stats.compiled_used += 1
            
            if pred_dyn == target:
                stats.correct_dynamic += 1
            
            if pred_cmp == target:
                stats.correct_compiled += 1
    
    return stats


def print_report(stats: ABStats):
    """Print A/B comparison report."""
    print("\n" + "=" * 70)
    print("A/B COMPARISON: Dynamic vs Compiled")
    print("=" * 70)
    
    print(f"\nSamples tested: {stats.n}")
    print(f"\n--- Agreement ---")
    print(f"  Agreement rate: {stats.agree_rate:.1%} ({stats.agree}/{stats.n})")
    
    print(f"\n--- Compiled Dispatch ---")
    print(f"  Compiled hit rate: {stats.compiled_rate:.1%} ({stats.compiled_used}/{stats.n})")
    
    print(f"\n--- Accuracy ---")
    print(f"  Dynamic accuracy:  {stats.accuracy_dynamic:.1%}")
    print(f"  Compiled accuracy: {stats.accuracy_compiled:.1%}")
    print(f"  Delta:             {stats.accuracy_delta:+.1%}")
    
    if stats.accuracy_delta >= 0:
        print(f"\n  ✓ Compiled matches or improves on dynamic")
    else:
        print(f"\n  ✗ Compiled worse than dynamic by {-stats.accuracy_delta:.1%}")
    
    if stats.disagreements:
        print(f"\n--- Worst Disagreements ({len(stats.disagreements)}) ---")
        for i, d in enumerate(stats.disagreements[:10]):
            compiled_flag = "compiled" if d['compiled_used'] else "dynamic"
            print(f"  {i+1}. {d['op']}({d['a']}, {d['b']}, c={d['c']}) "
                  f"→ target={d['target']}, dyn={d['pred_dynamic']}, "
                  f"cmp={d['pred_compiled']} [{compiled_flag}]")
    
    print("\n" + "=" * 70)
    
    # Verdict
    if stats.agree_rate > 0.95 and stats.accuracy_delta >= -0.01:
        print("VERDICT: ✓ PASS - Compiled dispatch is safe to use")
    elif stats.agree_rate > 0.90:
        print("VERDICT: ~ MARGINAL - Review disagreements before deploying")
    else:
        print("VERDICT: ✗ FAIL - Significant divergence, needs investigation")
    
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Generate training data
    print("\n[1] Generating training data...")
    train_data = generate_test_data(n_per_op=500)
    test_data = generate_test_data(n_per_op=200)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create and train model
    print("\n[2] Training model...")
    model = TriX6502(d_model=64, num_tiles=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Prepare tensors
    op_idx = torch.tensor([d['op_idx'] for d in train_data], device=device)
    a = torch.tensor([d['a'] for d in train_data], device=device)
    b = torch.tensor([d['b'] for d in train_data], device=device)
    c = torch.tensor([d['c'] for d in train_data], device=device)
    result = torch.tensor([d['result'] for d in train_data], device=device)
    result_bits = torch.stack([(result >> i) & 1 for i in range(8)], dim=1).float()
    
    # Train
    for epoch in range(30):
        model.train()
        model.ffn.reset_claim_tracking()
        perm = torch.randperm(len(train_data), device=device)
        total_loss, correct = 0, 0
        
        for i in range(0, len(train_data) - 256, 256):
            idx = perm[i:i+256]
            pred_bits, info, aux = model(op_idx[idx], a[idx], b[idx], c[idx], labels=op_idx[idx])
            loss = F.binary_cross_entropy(pred_bits, result_bits[idx]) + aux['total_aux']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = sum((pred_bits[:, i] > 0.5).long() << i for i in range(8))
            correct += (pred == result[idx]).sum().item()
        
        if (epoch + 1) % 10 == 0:
            acc = correct / len(train_data) * 100
            print(f"  Epoch {epoch+1}: loss={total_loss/(len(train_data)//256):.4f}, acc={acc:.1f}%")
    
    # Create compiler and compile
    print("\n[3] Compiling stable classes...")
    compiler = CompiledDispatch(model.ffn, confidence_threshold=0.5)
    
    # Profile
    profiles = compiler.profile_all(num_classes=len(OPCODES))
    print("\n  Profiles:")
    for class_id, stats in profiles.items():
        op_name = OPCODES[class_id] if class_id < len(OPCODES) else f"class_{class_id}"
        print(f"    {op_name}: tile={stats.mode_tile}, freq={stats.mode_frequency:.0%}, "
              f"compilable={stats.is_compilable(0.4)}")
    
    # Compile
    compiled = compiler.compile_stable(threshold=0.4, min_confidence=0.5)
    print(f"\n  Compiled {len(compiled)} classes: {list(compiled.keys())}")
    
    # Run A/B comparison
    print("\n[4] Running A/B comparison...")
    stats = eval_ab(model, compiler, test_data, device=device, confidence=0.9)
    
    # Print report
    print_report(stats)
    
    # Export results
    results = {
        'n': stats.n,
        'agree_rate': stats.agree_rate,
        'compiled_rate': stats.compiled_rate,
        'accuracy_dynamic': stats.accuracy_dynamic,
        'accuracy_compiled': stats.accuracy_compiled,
        'accuracy_delta': stats.accuracy_delta,
        'num_disagreements': len(stats.disagreements),
    }
    
    print("\n[5] Results JSON:")
    print(json.dumps(results, indent=2))
    
    return stats


if __name__ == "__main__":
    stats = main()
