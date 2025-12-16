#!/usr/bin/env python3
"""
FFT Atom: BUTTERFLY_CORE v2
===========================

Hypothesis: TDSR needs SELECTION to work. Give it multiple operations.

Operations:
  op=0: (a+b, a-b)  - standard butterfly
  op=1: (a-b, a+b)  - reversed
  op=2: (a+b, b-a)  - variant
  op=3: (a*2, b*2)  - doubling (control)

If accuracy improves with operation selection, the hypothesis is confirmed:
TDSR is for SELECTION, not raw COMPUTATION.

CODENAME: ANN WILSON - HEART Atom Test
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from trix.nn import TemporalTileLayer


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'value_range': 16,
    'num_ops': 4,              # 4 different operations
    'd_model': 32,
    'd_state': 16,
    'num_tiles': 8,
    'epochs': 100,
    'batch_size': 64,
    'lr': 0.005,
    'seeds': [42, 123, 456],
}

# Operation definitions
def op_butterfly_std(a, b):
    """Standard butterfly: (a+b, a-b)"""
    return a + b, a - b

def op_butterfly_rev(a, b):
    """Reversed butterfly: (a-b, a+b)"""
    return a - b, a + b

def op_butterfly_var(a, b):
    """Variant: (a+b, b-a)"""
    return a + b, b - a

def op_double(a, b):
    """Doubling: (2a, 2b)"""
    return 2 * a, 2 * b

OPERATIONS = [op_butterfly_std, op_butterfly_rev, op_butterfly_var, op_double]
OP_NAMES = ['std_butterfly', 'rev_butterfly', 'var_butterfly', 'double']


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(value_range=16, num_ops=4):
    """Generate (op, a, b) -> (out1, out2) data."""
    data = []
    
    for op_idx in range(num_ops):
        op_func = OPERATIONS[op_idx]
        for a in range(value_range):
            for b in range(value_range):
                out1, out2 = op_func(a, b)
                data.append({
                    'op': op_idx,
                    'a': a,
                    'b': b,
                    'out1': out1,
                    'out2': out2,
                })
    
    return data


def encode_output(value, value_range):
    """Encode output value to class index (handling negatives)."""
    # Output range: for value_range=16
    # - sums can be 0 to 30
    # - diffs can be -15 to 15
    # - doubles can be 0 to 30
    # Encode with offset
    return value + value_range - 1


def create_datasets(value_range=16, num_ops=4, train_ratio=0.8):
    """Create train/test splits."""
    all_data = generate_data(value_range, num_ops)
    np.random.shuffle(all_data)
    
    split = int(len(all_data) * train_ratio)
    return all_data[:split], all_data[split:]


# =============================================================================
# Model
# =============================================================================

class MultiButterflyPredictor(nn.Module):
    """
    Predict multiple butterfly variants using TDSR.
    
    The key test: Does having multiple operations improve accuracy
    by giving routing a purpose?
    """
    
    def __init__(self, value_range=16, num_ops=4, d_model=32, d_state=16, num_tiles=8):
        super().__init__()
        
        self.value_range = value_range
        self.num_ops = num_ops
        self.output_range = 2 * value_range - 1 + value_range  # Extra room for edge cases
        
        # Embeddings
        self.op_embed = nn.Embedding(num_ops, d_model // 4)
        self.a_embed = nn.Embedding(value_range, d_model // 4 + d_model // 8)
        self.b_embed = nn.Embedding(value_range, d_model // 4 + d_model // 8)
        
        # Project to d_model
        embed_dim = d_model // 4 + 2 * (d_model // 4 + d_model // 8)
        self.proj = nn.Linear(embed_dim, d_model)
        
        # Temporal tile layer
        self.temporal = TemporalTileLayer(
            d_model=d_model,
            d_state=d_state,
            num_tiles=num_tiles,
            routing_temp=0.5,
        )
        
        # Output heads
        self.out1_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.output_range),
        )
        self.out2_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.output_range),
        )
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        
        # Tracking
        self.register_buffer('tile_op_counts', torch.zeros(num_tiles, num_ops))
    
    def forward(self, op, a, b, track=True):
        batch_size = op.shape[0]
        device = op.device
        
        # Embed
        op_emb = self.op_embed(op)
        a_emb = self.a_embed(a)
        b_emb = self.b_embed(b)
        x = torch.cat([op_emb, a_emb, b_emb], dim=-1)
        x = self.proj(x)
        
        # Temporal processing
        state = self.temporal.init_state(batch_size, device)
        output, new_state, info = self.temporal(x, state)
        
        # Predict outputs
        out1_logits = self.out1_head(output)
        out2_logits = self.out2_head(output)
        
        # Track tile-op correspondence
        if track and self.training:
            with torch.no_grad():
                for i in range(batch_size):
                    tile = info['tile_idx'][i].item()
                    o = op[i].item()
                    self.tile_op_counts[tile, o] += 1
        
        return out1_logits, out2_logits, info
    
    def get_tile_op_purity(self):
        """Analyze tile specialization by operation."""
        counts = self.tile_op_counts.cpu().numpy()
        analysis = {}
        
        for tile in range(self.num_tiles):
            total = counts[tile].sum()
            if total > 0:
                dominant_op = counts[tile].argmax()
                purity = counts[tile, dominant_op] / total
                analysis[tile] = {
                    'dominant_op': int(dominant_op),
                    'dominant_op_name': OP_NAMES[dominant_op],
                    'purity': float(purity),
                    'counts': counts[tile].tolist(),
                    'total': int(total),
                }
        
        return analysis
    
    def reset_tracking(self):
        self.tile_op_counts.zero_()
        self.temporal.reset_tracking()


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, data, optimizer, device, value_range):
    model.train()
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    total_loss = 0
    correct_out1 = 0
    correct_out2 = 0
    correct_both = 0
    total = 0
    
    batch_size = CONFIG['batch_size']
    
    for i in range(0, len(data), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = [data[j] for j in batch_idx]
        
        op = torch.tensor([d['op'] for d in batch], device=device)
        a = torch.tensor([d['a'] for d in batch], device=device)
        b = torch.tensor([d['b'] for d in batch], device=device)
        target_out1 = torch.tensor([encode_output(d['out1'], value_range) for d in batch], device=device)
        target_out2 = torch.tensor([encode_output(d['out2'], value_range) for d in batch], device=device)
        
        out1_logits, out2_logits, info = model(op, a, b)
        
        loss = F.cross_entropy(out1_logits, target_out1) + F.cross_entropy(out2_logits, target_out2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pred_out1 = out1_logits.argmax(dim=-1)
        pred_out2 = out2_logits.argmax(dim=-1)
        
        correct_out1 += (pred_out1 == target_out1).sum().item()
        correct_out2 += (pred_out2 == target_out2).sum().item()
        correct_both += ((pred_out1 == target_out1) & (pred_out2 == target_out2)).sum().item()
        total += len(batch)
    
    return {
        'loss': total_loss / (len(data) // batch_size + 1),
        'acc_out1': correct_out1 / total,
        'acc_out2': correct_out2 / total,
        'acc_both': correct_both / total,
    }


def evaluate(model, data, device, value_range):
    model.eval()
    
    correct_both = 0
    total = 0
    
    per_op_correct = defaultdict(int)
    per_op_total = defaultdict(int)
    
    with torch.no_grad():
        for d in data:
            op = torch.tensor([d['op']], device=device)
            a = torch.tensor([d['a']], device=device)
            b = torch.tensor([d['b']], device=device)
            
            out1_logits, out2_logits, info = model(op, a, b, track=False)
            
            pred_out1 = out1_logits.argmax(dim=-1).item() - (value_range - 1)
            pred_out2 = out2_logits.argmax(dim=-1).item() - (value_range - 1)
            
            is_correct = (pred_out1 == d['out1']) and (pred_out2 == d['out2'])
            
            correct_both += is_correct
            total += 1
            
            per_op_correct[d['op']] += is_correct
            per_op_total[d['op']] += 1
    
    per_op_acc = {OP_NAMES[op]: per_op_correct[op] / per_op_total[op] 
                  for op in per_op_total}
    
    return {
        'acc_both': correct_both / total,
        'per_op_accuracy': per_op_acc,
    }


# =============================================================================
# Main
# =============================================================================

def run_single_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data, test_data = create_datasets(CONFIG['value_range'], CONFIG['num_ops'])
    
    model = MultiButterflyPredictor(
        value_range=CONFIG['value_range'],
        num_ops=CONFIG['num_ops'],
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        num_tiles=CONFIG['num_tiles'],
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    best_acc = 0
    
    for epoch in range(CONFIG['epochs']):
        model.reset_tracking()
        
        train_metrics = train_epoch(model, train_data, optimizer, device, CONFIG['value_range'])
        test_metrics = evaluate(model, test_data, device, CONFIG['value_range'])
        
        if test_metrics['acc_both'] > best_acc:
            best_acc = test_metrics['acc_both']
        
        if test_metrics['acc_both'] >= 0.95:
            print(f"  [Seed {seed}] 95%+ at epoch {epoch + 1}!")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"  [Seed {seed}] Epoch {epoch + 1}: "
                  f"loss={train_metrics['loss']:.4f}, "
                  f"both={test_metrics['acc_both']:.1%}")
    
    # Final analysis
    model.reset_tracking()
    model.train()
    for d in train_data[:1000]:
        op = torch.tensor([d['op']], device=device)
        a = torch.tensor([d['a']], device=device)
        b = torch.tensor([d['b']], device=device)
        model(op, a, b, track=True)
    
    final_metrics = evaluate(model, test_data, device, CONFIG['value_range'])
    tile_analysis = model.get_tile_op_purity()
    
    return {
        'seed': seed,
        'accuracy': final_metrics['acc_both'],
        'per_op': final_metrics['per_op_accuracy'],
        'tile_analysis': tile_analysis,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    print("\n[HYPOTHESIS TEST: SELECTION vs COMPUTATION]")
    print("If TDSR needs selection, multiple operations should improve accuracy.")
    print(f"\nOperations: {OP_NAMES}")
    
    print("\n[TRAINING]")
    all_results = []
    
    for seed in CONFIG['seeds']:
        print(f"\nSeed {seed}:")
        result = run_single_seed(seed, device)
        all_results.append(result)
    
    # Aggregate
    accuracies = [r['accuracy'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n" + "=" * 70)
    print("RESULTS: BUTTERFLY v2 (with operation selection)")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"(Compare to v1 without selection: 0%)")
    
    print("\nPer-operation accuracy (best run):")
    best_run = max(all_results, key=lambda r: r['accuracy'])
    for op_name, acc in best_run['per_op'].items():
        print(f"  {op_name}: {acc:.1%}")
    
    print("\nTile-operation specialization:")
    for tile, info in sorted(best_run['tile_analysis'].items()):
        print(f"  Tile {tile}: {info['dominant_op_name']} "
              f"(purity={info['purity']:.0%}, n={info['total']})")
    
    # Verdict
    print("\n" + "=" * 70)
    if mean_acc > 0.5:
        print("HYPOTHESIS CONFIRMED: Selection improves accuracy!")
        print(f"  v1 (no selection): 0%")
        print(f"  v2 (with selection): {mean_acc:.1%}")
        print("\nTDSR is for SELECTION, not raw COMPUTATION.")
    else:
        print("HYPOTHESIS UNCLEAR: Selection didn't help enough.")
    print("=" * 70)
    
    # Save results
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'atom_butterfly_v2_{timestamp}.json'
    
    results_json = {
        'config': CONFIG,
        'timestamp': timestamp,
        'hypothesis': 'TDSR needs selection to work',
        'comparison': {
            'v1_no_selection': 0.0,
            'v2_with_selection': float(mean_acc),
        },
        'aggregate': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'all_accuracies': [float(a) for a in accuracies],
        },
        'best_run': {
            'seed': best_run['seed'],
            'accuracy': float(best_run['accuracy']),
            'per_op': {k: float(v) for k, v in best_run['per_op'].items()},
        },
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results_json


if __name__ == "__main__":
    main()
