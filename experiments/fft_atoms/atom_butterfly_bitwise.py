#!/usr/bin/env python3
"""
FFT Atom: BUTTERFLY with Bit-Level Output
==========================================

PURE TRIX. No symbolic organs.

The 6502 proved bit-level prediction works (95%+ accuracy).
Apply the same approach to butterfly: (a, b) → (a+b, a-b)

Output encoding:
- Sum (0-30): 5 bits unsigned
- Diff (-15 to 15): 5 bits magnitude + 1 sign bit

Total: 11 bits, sigmoid + BCE loss

CODENAME: ANN WILSON - PURE HEART
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

from trix.nn import SparseLookupFFNv2


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'value_range': 16,      # 0-15
    'd_model': 128,         # Larger model
    'num_tiles': 16,        # More tiles
    'num_freqs': 8,         # Fourier frequencies
    'epochs': 200,          # Proof of concept
    'batch_size': 64,
    'lr': 0.005,
    'seeds': [42],          # Single seed for speed
}


# =============================================================================
# Bit Encoding/Decoding
# =============================================================================

def encode_sum_bits(value, num_bits=5):
    """Encode unsigned sum (0-30) as bits."""
    bits = []
    for i in range(num_bits):
        bits.append((value >> i) & 1)
    return bits


def decode_sum_bits(bits):
    """Decode bits to unsigned sum."""
    value = 0
    for i, b in enumerate(bits):
        value += int(b > 0.5) << i
    return value


def encode_diff_bits(value, num_bits=5):
    """Encode signed diff (-15 to 15) as magnitude + sign."""
    sign = 1 if value < 0 else 0
    mag = abs(value)
    bits = [sign]
    for i in range(num_bits):
        bits.append((mag >> i) & 1)
    return bits  # 6 bits total


def decode_diff_bits(bits):
    """Decode bits to signed diff."""
    sign = bits[0]
    mag = 0
    for i, b in enumerate(bits[1:]):
        mag += int(b > 0.5) << i
    return -mag if sign > 0.5 else mag


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(value_range=16):
    """Generate all (a, b) -> (sum_bits, diff_bits) pairs."""
    data = []
    
    for a in range(value_range):
        for b in range(value_range):
            sum_val = a + b
            diff_val = a - b
            
            sum_bits = encode_sum_bits(sum_val, num_bits=5)
            diff_bits = encode_diff_bits(diff_val, num_bits=5)
            
            data.append({
                'a': a,
                'b': b,
                'sum': sum_val,
                'diff': diff_val,
                'sum_bits': sum_bits,      # 5 bits
                'diff_bits': diff_bits,    # 6 bits (sign + 5 mag)
            })
    
    return data


# =============================================================================
# Model: Pure TriX with Bit-Level Output
# =============================================================================

def fourier_features(x, num_freqs=8, max_val=16):
    """
    Encode integers with Fourier features.
    
    Arithmetic has natural frequency structure.
    This is an architectural prior, not symbolic computation.
    """
    # Normalize to [0, 2π]
    x_norm = x.float().unsqueeze(-1) * (2 * np.pi / max_val)
    
    # Frequencies: 1, 2, 4, 8, ...
    freqs = (2 ** torch.arange(num_freqs, device=x.device, dtype=torch.float)).unsqueeze(0)
    
    angles = x_norm * freqs  # (batch, num_freqs)
    
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch, 2*num_freqs)


class PureTriXButterfly(nn.Module):
    """
    Pure TriX butterfly using stacked SparseLookupFFNv2.
    
    No symbolic organs. Everything learned.
    Bit-level output like 6502.
    Fourier features for input encoding.
    Multiple TriX layers for more expressive power.
    """
    
    def __init__(self, value_range=16, d_model=128, num_tiles=16, num_freqs=8, num_layers=3):
        super().__init__()
        
        self.value_range = value_range
        self.num_freqs = num_freqs
        
        # Input: Fourier features for a and b
        fourier_dim = 2 * num_freqs * 2  # a and b
        
        # Project Fourier features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Stack of TriX layers with residual connections
        self.trix_layers = nn.ModuleList([
            SparseLookupFFNv2(
                d_model=d_model,
                num_tiles=num_tiles,
                tiles_per_cluster=max(2, num_tiles // 4),
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Bit-level output heads with more capacity
        self.sum_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 5),
        )
        self.diff_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 6),
        )
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_layers = num_layers
    
    def forward(self, a, b):
        """
        Args:
            a, b: (batch,) integer values [0, value_range)
        
        Returns:
            sum_logits: (batch, 5) logits for sum bits
            diff_logits: (batch, 6) logits for diff bits
            routing_info: from TriX
        """
        # Fourier feature encoding
        a_feat = fourier_features(a, self.num_freqs, self.value_range)
        b_feat = fourier_features(b, self.num_freqs, self.value_range)
        x = torch.cat([a_feat, b_feat], dim=-1)  # (batch, 4*num_freqs)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, d_model)
        
        # Add sequence dimension for TriX
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Stack of TriX layers with residual connections
        all_aux = []
        for i, (trix_layer, ln) in enumerate(zip(self.trix_layers, self.layer_norms)):
            out, routing_info, aux = trix_layer(x)
            x = ln(x + out)  # Residual + LayerNorm
            all_aux.append(aux)
        
        # Combine aux losses
        combined_aux = {'total_aux': sum(a.get('total_aux', 0) for a in all_aux)}
        
        # Remove sequence dimension
        out = x.squeeze(1)  # (batch, d_model)
        
        # Bit predictions
        sum_logits = self.sum_head(out)
        diff_logits = self.diff_head(out)
        
        return sum_logits, diff_logits, routing_info, combined_aux
    
    def predict(self, a, b):
        """Get integer predictions."""
        sum_logits, diff_logits, _, _ = self.forward(a, b)
        
        sum_probs = torch.sigmoid(sum_logits)
        diff_probs = torch.sigmoid(diff_logits)
        
        # Decode
        batch_size = a.shape[0]
        sum_preds = []
        diff_preds = []
        
        for i in range(batch_size):
            sum_preds.append(decode_sum_bits(sum_probs[i].tolist()))
            diff_preds.append(decode_diff_bits(diff_probs[i].tolist()))
        
        return sum_preds, diff_preds


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, data, optimizer, device):
    model.train()
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    total_loss = 0
    total_sum_bits_correct = 0
    total_diff_bits_correct = 0
    total_bits = 0
    
    batch_size = CONFIG['batch_size']
    
    for i in range(0, len(data), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = [data[j] for j in batch_idx]
        
        a = torch.tensor([d['a'] for d in batch], device=device)
        b = torch.tensor([d['b'] for d in batch], device=device)
        
        # Target bits
        sum_bits = torch.tensor([d['sum_bits'] for d in batch], device=device, dtype=torch.float)
        diff_bits = torch.tensor([d['diff_bits'] for d in batch], device=device, dtype=torch.float)
        
        # Forward
        sum_logits, diff_logits, routing_info, aux = model(a, b)
        
        # BCE loss on bits
        loss_sum = F.binary_cross_entropy_with_logits(sum_logits, sum_bits)
        loss_diff = F.binary_cross_entropy_with_logits(diff_logits, diff_bits)
        loss = loss_sum + loss_diff
        
        # Add aux losses
        if 'total_aux' in aux:
            loss = loss + 0.01 * aux['total_aux']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Bit accuracy
        sum_pred_bits = (torch.sigmoid(sum_logits) > 0.5).float()
        diff_pred_bits = (torch.sigmoid(diff_logits) > 0.5).float()
        
        total_sum_bits_correct += (sum_pred_bits == sum_bits).sum().item()
        total_diff_bits_correct += (diff_pred_bits == diff_bits).sum().item()
        total_bits += sum_bits.numel() + diff_bits.numel()
    
    return {
        'loss': total_loss / (len(data) // batch_size + 1),
        'bit_accuracy': (total_sum_bits_correct + total_diff_bits_correct) / total_bits,
    }


def evaluate(model, data, device):
    model.eval()
    
    correct_sum = 0
    correct_diff = 0
    correct_both = 0
    total = 0
    
    # Track per-tile accuracy
    tile_correct = defaultdict(int)
    tile_total = defaultdict(int)
    
    with torch.no_grad():
        for d in data:
            a = torch.tensor([d['a']], device=device)
            b = torch.tensor([d['b']], device=device)
            
            sum_preds, diff_preds = model.predict(a, b)
            
            sum_pred = sum_preds[0]
            diff_pred = diff_preds[0]
            
            sum_correct = (sum_pred == d['sum'])
            diff_correct = (diff_pred == d['diff'])
            
            correct_sum += sum_correct
            correct_diff += diff_correct
            correct_both += (sum_correct and diff_correct)
            total += 1
    
    return {
        'acc_sum': correct_sum / total,
        'acc_diff': correct_diff / total,
        'acc_both': correct_both / total,
    }


# =============================================================================
# Main
# =============================================================================

def run_single_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data = generate_data(CONFIG['value_range'])
    np.random.shuffle(data)
    
    # Use all data for both train and test (proof of concept)
    train_data = data
    test_data = data
    
    print(f"  All data: {len(data)} (proof of concept - can TriX learn this?)")
    
    model = PureTriXButterfly(
        value_range=CONFIG['value_range'],
        d_model=CONFIG['d_model'],
        num_tiles=CONFIG['num_tiles'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    best_acc = 0
    
    for epoch in range(CONFIG['epochs']):
        train_metrics = train_epoch(model, train_data, optimizer, device)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            test_metrics = evaluate(model, test_data, device)
            
            if test_metrics['acc_both'] > best_acc:
                best_acc = test_metrics['acc_both']
            
            print(f"  [Seed {seed}] Epoch {epoch + 1}: "
                  f"loss={train_metrics['loss']:.4f}, "
                  f"bits={train_metrics['bit_accuracy']:.1%}, "
                  f"both={test_metrics['acc_both']:.1%}")
            
            if test_metrics['acc_both'] >= 0.99:
                print(f"  [Seed {seed}] 99%+ achieved!")
                break
    
    # Final evaluation
    final_metrics = evaluate(model, test_data, device)
    
    return {
        'seed': seed,
        'final_metrics': final_metrics,
        'best_acc': best_acc,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Value range: 0-{CONFIG['value_range']-1}")
    
    print("\n" + "=" * 70)
    print("PURE TRIX BUTTERFLY")
    print("Bit-level output, no symbolic organs")
    print("=" * 70)
    
    print("\n[TRAINING]")
    all_results = []
    
    for seed in CONFIG['seeds']:
        print(f"\nSeed {seed}:")
        result = run_single_seed(seed, device)
        all_results.append(result)
    
    # Aggregate
    accuracies = [r['final_metrics']['acc_both'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n" + "=" * 70)
    print("RESULTS: PURE TRIX BUTTERFLY")
    print("=" * 70)
    
    print(f"\nBoth Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"Seeds: {[f'{a:.1%}' for a in accuracies]}")
    
    best_run = max(all_results, key=lambda r: r['final_metrics']['acc_both'])
    print(f"\nBest run (seed {best_run['seed']}):")
    print(f"  Sum:  {best_run['final_metrics']['acc_sum']:.1%}")
    print(f"  Diff: {best_run['final_metrics']['acc_diff']:.1%}")
    print(f"  Both: {best_run['final_metrics']['acc_both']:.1%}")
    
    # Verdict
    print("\n" + "=" * 70)
    if mean_acc >= 0.95:
        print("✓ PURE TRIX BUTTERFLY: SUCCESS")
        print("  Bit-level output enables learned arithmetic")
        print("  No symbolic organs needed")
    elif mean_acc >= 0.80:
        print("◐ PURE TRIX BUTTERFLY: PARTIAL SUCCESS")
        print(f"  {mean_acc:.1%} accuracy - promising but needs work")
    else:
        print("✗ PURE TRIX BUTTERFLY: NEEDS MORE WORK")
        print(f"  {mean_acc:.1%} accuracy")
    print("=" * 70)
    
    # Save results
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'pure_trix_butterfly_{timestamp}.json'
    
    results_json = {
        'config': CONFIG,
        'timestamp': timestamp,
        'approach': 'bit-level output, BCE loss',
        'aggregate': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'all_accuracies': [float(a) for a in accuracies],
        },
        'best_run': {
            'seed': best_run['seed'],
            'metrics': {k: float(v) for k, v in best_run['final_metrics'].items()},
        },
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return mean_acc


if __name__ == "__main__":
    main()
