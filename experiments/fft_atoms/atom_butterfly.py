#!/usr/bin/env python3
"""
FFT Atom 2: BUTTERFLY_CORE
==========================

The compute primitive of FFT: (a, b) -> (a+b, a-b)

No twiddles, no permutation - just the core butterfly operation.

Tests whether TDSR tiles can represent the compute primitive cleanly.

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
    'value_range': 16,         # Inputs 0-15 (small integers first)
    'd_model': 32,
    'd_state': 16,
    'num_tiles': 8,
    'epochs': 100,
    'batch_size': 64,
    'lr': 0.003,
    'seeds': [42, 123, 456],
}


# =============================================================================
# Data Generation
# =============================================================================

def generate_butterfly_data(value_range=16):
    """
    Generate all (a, b) -> (a+b, a-b) pairs.
    
    For value_range=16:
    - Inputs: a, b ∈ {0, 1, ..., 15}
    - Outputs: sum ∈ {0, ..., 30}, diff ∈ {-15, ..., 15}
    
    We'll encode diff with offset to make it non-negative.
    """
    data = []
    
    for a in range(value_range):
        for b in range(value_range):
            sum_ab = a + b
            diff_ab = a - b
            
            data.append({
                'a': a,
                'b': b,
                'sum': sum_ab,
                'diff': diff_ab,
                'diff_encoded': diff_ab + value_range - 1,  # Shift to [0, 2*range-2]
            })
    
    return data


def create_datasets(value_range=16, train_ratio=0.8):
    """Create train/test splits."""
    all_data = generate_butterfly_data(value_range)
    np.random.shuffle(all_data)
    
    split = int(len(all_data) * train_ratio)
    return all_data[:split], all_data[split:]


# =============================================================================
# Model
# =============================================================================

class ButterflyPredictor(nn.Module):
    """
    Predict butterfly outputs (a+b, a-b) using temporal tiles.
    
    Input: (a, b) encoded
    Output: (sum, diff)
    
    The key question: Can tiles learn the butterfly primitive?
    """
    
    def __init__(self, value_range=16, d_model=32, d_state=16, num_tiles=8):
        super().__init__()
        
        self.value_range = value_range
        self.output_range_sum = 2 * value_range - 1  # 0 to 30 for range=16
        self.output_range_diff = 2 * value_range - 1  # -15 to 15 encoded as 0 to 30
        
        # Embeddings for a and b
        self.a_embed = nn.Embedding(value_range, d_model // 2)
        self.b_embed = nn.Embedding(value_range, d_model // 2)
        
        # Temporal tile layer
        self.temporal = TemporalTileLayer(
            d_model=d_model,
            d_state=d_state,
            num_tiles=num_tiles,
            routing_temp=0.5,
        )
        
        # Output heads: predict sum and diff
        self.sum_head = nn.Linear(d_model, self.output_range_sum)
        self.diff_head = nn.Linear(d_model, self.output_range_diff)
        
        self.d_model = d_model
        self.d_state = d_state
        self.num_tiles = num_tiles
        
        # Tracking: which tiles handle which magnitude regimes
        self.register_buffer('tile_sum_counts', torch.zeros(num_tiles, self.output_range_sum))
    
    def forward(self, a, b, track=True):
        """
        Forward pass.
        
        Args:
            a: (batch,) first input values
            b: (batch,) second input values
            track: whether to track tile usage
        
        Returns:
            sum_logits: (batch, output_range_sum)
            diff_logits: (batch, output_range_diff)
            info: routing information
        """
        batch_size = a.shape[0]
        device = a.device
        
        # Embed inputs
        a_emb = self.a_embed(a)
        b_emb = self.b_embed(b)
        x = torch.cat([a_emb, b_emb], dim=-1)
        
        # Initialize state
        state = self.temporal.init_state(batch_size, device)
        
        # Single-step temporal processing
        output, new_state, info = self.temporal(x, state)
        
        # Predict sum and diff
        sum_logits = self.sum_head(output)
        diff_logits = self.diff_head(output)
        
        return sum_logits, diff_logits, info
    
    def reset_tracking(self):
        self.tile_sum_counts.zero_()
        self.temporal.reset_tracking()


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, data, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    total_loss = 0
    correct_sum = 0
    correct_diff = 0
    correct_both = 0
    total = 0
    
    batch_size = CONFIG['batch_size']
    
    for i in range(0, len(data), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = [data[j] for j in batch_idx]
        
        a = torch.tensor([d['a'] for d in batch], device=device)
        b = torch.tensor([d['b'] for d in batch], device=device)
        target_sum = torch.tensor([d['sum'] for d in batch], device=device)
        target_diff = torch.tensor([d['diff_encoded'] for d in batch], device=device)
        
        sum_logits, diff_logits, info = model(a, b)
        
        loss_sum = F.cross_entropy(sum_logits, target_sum)
        loss_diff = F.cross_entropy(diff_logits, target_diff)
        loss = loss_sum + loss_diff
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pred_sum = sum_logits.argmax(dim=-1)
        pred_diff = diff_logits.argmax(dim=-1)
        
        correct_sum += (pred_sum == target_sum).sum().item()
        correct_diff += (pred_diff == target_diff).sum().item()
        correct_both += ((pred_sum == target_sum) & (pred_diff == target_diff)).sum().item()
        total += len(batch)
    
    n_batches = len(data) // batch_size + 1
    return {
        'loss': total_loss / n_batches,
        'acc_sum': correct_sum / total,
        'acc_diff': correct_diff / total,
        'acc_both': correct_both / total,
    }


def evaluate(model, data, device):
    """Evaluate model."""
    model.eval()
    
    correct_sum = 0
    correct_diff = 0
    correct_both = 0
    total = 0
    
    predictions = []
    
    with torch.no_grad():
        for d in data:
            a = torch.tensor([d['a']], device=device)
            b = torch.tensor([d['b']], device=device)
            
            sum_logits, diff_logits, info = model(a, b, track=False)
            
            pred_sum = sum_logits.argmax(dim=-1).item()
            pred_diff = diff_logits.argmax(dim=-1).item() - (model.value_range - 1)  # Decode
            
            tile = info['tile_idx'][0].item()
            
            is_sum_correct = pred_sum == d['sum']
            is_diff_correct = pred_diff == d['diff']
            is_both_correct = is_sum_correct and is_diff_correct
            
            correct_sum += is_sum_correct
            correct_diff += is_diff_correct
            correct_both += is_both_correct
            total += 1
            
            predictions.append({
                'a': d['a'],
                'b': d['b'],
                'sum_true': d['sum'],
                'sum_pred': pred_sum,
                'diff_true': d['diff'],
                'diff_pred': pred_diff,
                'correct_sum': is_sum_correct,
                'correct_diff': is_diff_correct,
                'correct_both': is_both_correct,
                'tile': tile,
            })
    
    return {
        'acc_sum': correct_sum / total,
        'acc_diff': correct_diff / total,
        'acc_both': correct_both / total,
        'predictions': predictions,
    }


# =============================================================================
# Analysis
# =============================================================================

def analyze_tiles(predictions, num_tiles):
    """Analyze tile behavior."""
    
    # Tile usage by input magnitude
    tile_magnitude = defaultdict(list)
    tile_correct = defaultdict(list)
    
    for p in predictions:
        tile = p['tile']
        magnitude = p['a'] + p['b']  # Sum as proxy for magnitude
        tile_magnitude[tile].append(magnitude)
        tile_correct[tile].append(p['correct_both'])
    
    analysis = {}
    for tile in range(num_tiles):
        if tile_magnitude[tile]:
            mags = tile_magnitude[tile]
            correct = tile_correct[tile]
            analysis[tile] = {
                'count': len(mags),
                'mean_magnitude': np.mean(mags),
                'std_magnitude': np.std(mags),
                'accuracy': np.mean(correct),
            }
    
    return analysis


def print_report(metrics, tile_analysis):
    """Print human-readable report."""
    
    print("\n" + "=" * 70)
    print("BUTTERFLY CORE REPORT")
    print("=" * 70)
    
    print(f"\n[ACCURACY]")
    print(f"  Sum (a+b):  {metrics['acc_sum']:.1%}")
    print(f"  Diff (a-b): {metrics['acc_diff']:.1%}")
    print(f"  Both:       {metrics['acc_both']:.1%}")
    
    print(f"\n[TILE ANALYSIS]")
    for tile, info in sorted(tile_analysis.items()):
        print(f"  Tile {tile}: n={info['count']}, "
              f"mean_mag={info['mean_magnitude']:.1f}, "
              f"acc={info['accuracy']:.0%}")
    
    print("=" * 70)


# =============================================================================
# Main Runner
# =============================================================================

def run_single_seed(seed, device):
    """Run experiment with single seed."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data
    train_data, test_data = create_datasets(CONFIG['value_range'])
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create model
    model = ButterflyPredictor(
        value_range=CONFIG['value_range'],
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        num_tiles=CONFIG['num_tiles'],
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Training loop
    history = []
    
    for epoch in range(CONFIG['epochs']):
        model.reset_tracking()
        
        train_metrics = train_epoch(model, train_data, optimizer, device)
        test_metrics = evaluate(model, test_data, device)
        
        history.append({
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items() if k != 'predictions'},
        })
        
        # Early stopping on high accuracy
        if test_metrics['acc_both'] >= 0.99:
            print(f"  [Seed {seed}] High accuracy at epoch {epoch + 1}!")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"  [Seed {seed}] Epoch {epoch + 1}: "
                  f"loss={train_metrics['loss']:.4f}, "
                  f"both={test_metrics['acc_both']:.1%}")
    
    # Final evaluation
    final_metrics = evaluate(model, test_data, device)
    tile_analysis = analyze_tiles(final_metrics['predictions'], CONFIG['num_tiles'])
    
    return {
        'seed': seed,
        'final_metrics': {k: v for k, v in final_metrics.items() if k != 'predictions'},
        'tile_analysis': tile_analysis,
        'history': history,
    }


def main():
    """Main experiment runner."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Value range: 0-{CONFIG['value_range']-1}")
    
    # Show the task
    print("\n[TASK: BUTTERFLY_CORE]")
    print("Learn (a, b) -> (a+b, a-b)")
    print("\nExamples:")
    for a, b in [(3, 5), (7, 2), (0, 8), (15, 15)]:
        print(f"  ({a}, {b}) -> ({a+b}, {a-b})")
    
    # Run multiple seeds
    print(f"\n[TRAINING]")
    all_results = []
    
    for seed in CONFIG['seeds']:
        print(f"\nSeed {seed}:")
        result = run_single_seed(seed, device)
        all_results.append(result)
    
    # Aggregate results
    accuracies = [r['final_metrics']['acc_both'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # Best run for detailed analysis
    best_run = max(all_results, key=lambda r: r['final_metrics']['acc_both'])
    
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"\nBoth Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"Seeds: {accuracies}")
    
    print_report(best_run['final_metrics'], best_run['tile_analysis'])
    
    # Pass/Fail
    print("\n[VERDICT]")
    passed = mean_acc >= 0.90
    
    if passed:
        print(f"✓ ACCURACY: {mean_acc:.1%} ≥ 90%")
        print("\nATOM BUTTERFLY_CORE: ✓ PASS")
        print("TDSR can learn the butterfly compute primitive!")
    else:
        print(f"✗ ACCURACY: {mean_acc:.1%} < 90%")
        print("\nATOM BUTTERFLY_CORE: ✗ FAIL")
        print("TDSR struggles with arithmetic - may need richer tiles.")
    
    print("=" * 70)
    
    # Save results
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'atom_butterfly_{timestamp}.json'
    
    results_json = {
        'config': CONFIG,
        'timestamp': timestamp,
        'device': device,
        'aggregate': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'all_accuracies': [float(a) for a in accuracies],
            'passed': bool(passed),
        },
        'best_run': {
            'seed': best_run['seed'],
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in best_run['final_metrics'].items()},
            'tile_analysis': {str(k): {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                       for kk, vv in v.items()}
                             for k, v in best_run['tile_analysis'].items()},
        },
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results_json


if __name__ == "__main__":
    results = main()
