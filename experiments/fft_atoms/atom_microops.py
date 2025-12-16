#!/usr/bin/env python3
"""
FFT Micro-Ops: Pure TriX with Meaningful Routing
=================================================

The 6502 worked because opcodes gave routing meaning.
FFT needs the same: a vocabulary of micro-ops.

MICRO-OPS:
  ADD: a + b
  SUB: a - b
  
Later (for full FFT):
  SWAP: exchange values
  TWIDDLE_0: multiply by 1 (identity)
  TWIDDLE_Q: multiply by i (quarter turn)

With operation type as input, routing has a job:
  - Tile 0 learns ADD
  - Tile 1 learns SUB
  - Router selects based on opcode

This is PURE TRIX. No symbolic organs.
The tiles LEARN the operations. Routing SELECTS.

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
from trix import SparseLookupFFN  # v1 for comparison


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'value_range': 16,      # 0-15
    'd_model': 128,         # More capacity
    'num_tiles': 8,         # Enough tiles for ops to specialize
    'num_freqs': 8,         # More frequencies
    'num_layers': 2,        # Stack TriX layers
    'epochs': 100,          # Proof of concept
    'batch_size': 32,       # Smaller batches
    'lr': 0.005,
    'seeds': [1122911624],  # The Second Star Constant
    'use_v1': True,         # v1 works better
}

# Micro-op definitions
OPS = {
    0: ('ADD', lambda a, b: a + b),
    1: ('SUB', lambda a, b: a - b),
}
NUM_OPS = len(OPS)


# =============================================================================
# Bit Encoding
# =============================================================================

def encode_bits(value, num_bits=6, signed=True):
    """Encode value as bits. Handles signed values."""
    if signed:
        sign = 1 if value < 0 else 0
        mag = abs(value)
        bits = [sign]
        for i in range(num_bits - 1):
            bits.append((mag >> i) & 1)
    else:
        bits = []
        for i in range(num_bits):
            bits.append((value >> i) & 1)
    return bits


def decode_bits(bits, signed=True):
    """Decode bits to integer."""
    if signed:
        sign = bits[0]
        mag = 0
        for i, b in enumerate(bits[1:]):
            mag += int(b > 0.5) << i
        return -mag if sign > 0.5 else mag
    else:
        value = 0
        for i, b in enumerate(bits):
            value += int(b > 0.5) << i
        return value


# =============================================================================
# Fourier Features
# =============================================================================

def fourier_features(x, num_freqs=6, max_val=16):
    """Encode integers with Fourier features."""
    x_norm = x.float().unsqueeze(-1) * (2 * np.pi / max_val)
    freqs = (2 ** torch.arange(num_freqs, device=x.device, dtype=torch.float)).unsqueeze(0)
    angles = x_norm * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(value_range=16):
    """Generate (op, a, b) -> result data for all micro-ops."""
    data = []
    
    for op_id, (op_name, op_func) in OPS.items():
        for a in range(value_range):
            for b in range(value_range):
                result = op_func(a, b)
                
                # Determine output encoding
                # ADD: 0-30, unsigned, 5 bits
                # SUB: -15 to 15, signed, 6 bits (1 sign + 5 mag)
                if op_name == 'ADD':
                    result_bits = encode_bits(result, num_bits=5, signed=False)
                else:  # SUB
                    result_bits = encode_bits(result, num_bits=6, signed=True)
                
                data.append({
                    'op': op_id,
                    'op_name': op_name,
                    'a': a,
                    'b': b,
                    'result': result,
                    'result_bits': result_bits,
                })
    
    return data


# =============================================================================
# Model: Pure TriX with Micro-Ops
# =============================================================================

class PureTriXMicroOps(nn.Module):
    """
    Pure TriX for FFT micro-operations.
    
    Key insight: operation type is an INPUT.
    This gives routing a meaningful job.
    
    Architecture:
    - Fourier features for a, b
    - Embedding for operation type
    - TriX FFN routes based on (op, a, b)
    - Bit-level output
    """
    
    def __init__(self, value_range=16, num_ops=2, d_model=64, num_tiles=8, num_freqs=6, 
                 num_layers=1, use_v1=False):
        super().__init__()
        
        self.value_range = value_range
        self.num_ops = num_ops
        self.num_freqs = num_freqs
        self.num_tiles = num_tiles
        self.num_layers = num_layers
        self.use_v1 = use_v1
        
        # Operation embedding (critical - this is what routing keys on)
        self.op_embed = nn.Embedding(num_ops, d_model // 4)
        
        # Fourier features for a and b
        fourier_dim = 2 * num_freqs  # per value
        
        # Project all inputs to d_model
        input_dim = d_model // 4 + 2 * fourier_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Core TriX FFN(s) - this is where routing happens
        if use_v1:
            print("  Using SparseLookupFFN (v1)")
            self.trix_layers = nn.ModuleList([
                SparseLookupFFN(
                    d_model=d_model,
                    num_tiles=num_tiles,
                    tiles_per_cluster=max(2, num_tiles // 4),
                )
                for _ in range(num_layers)
            ])
        else:
            print("  Using SparseLookupFFNv2")
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
        
        # Bit-level output (max 6 bits for signed SUB)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 6),
        )
        
        # Track tile-operation correspondence
        self.register_buffer('tile_op_counts', torch.zeros(num_tiles, num_ops))
        
        self.d_model = d_model
    
    def forward(self, op, a, b, track=True):
        """
        Args:
            op: (batch,) operation indices
            a, b: (batch,) integer values
            track: whether to track tile-op correspondence
        
        Returns:
            logits: (batch, 6) bit logits
            routing_info: from TriX
            aux: auxiliary losses
        """
        batch_size = op.shape[0]
        device = op.device
        
        # Encode inputs
        op_emb = self.op_embed(op)  # (batch, d_model//4)
        a_feat = fourier_features(a, self.num_freqs, self.value_range)  # (batch, 2*num_freqs)
        b_feat = fourier_features(b, self.num_freqs, self.value_range)
        
        # Concatenate and project
        x = torch.cat([op_emb, a_feat, b_feat], dim=-1)
        x = self.input_proj(x)  # (batch, d_model)
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Stack of TriX layers
        all_routing_info = []
        all_aux = []
        
        for trix_layer, ln in zip(self.trix_layers, self.layer_norms):
            out, routing_info, aux = trix_layer(x)
            x = ln(x + out)  # Residual + LayerNorm
            all_routing_info.append(routing_info)
            all_aux.append(aux)
        
        # Use first layer's routing for tracking (most interpretable)
        routing_info = all_routing_info[0]
        
        # Track tile-operation correspondence
        if track and self.training:
            with torch.no_grad():
                # Get winning tiles from routing info
                if 'tile_idx' in routing_info:
                    tile_idx = routing_info['tile_idx'].squeeze(-1)  # (batch,)
                    for i in range(batch_size):
                        t = tile_idx[i].item() if tile_idx.dim() > 0 else tile_idx.item()
                        o = op[i].item()
                        self.tile_op_counts[t, o] += 1
        
        # Combine aux losses
        combined_aux = {'total_aux': sum(a.get('total_aux', 0) for a in all_aux)}
        
        # Output
        out = x.squeeze(1)  # (batch, d_model)
        logits = self.output_head(out)
        
        return logits, routing_info, combined_aux
    
    def get_tile_specialization(self):
        """Analyze which tiles specialize to which operations."""
        counts = self.tile_op_counts.cpu().numpy()
        analysis = {}
        
        for tile in range(self.num_tiles):
            total = counts[tile].sum()
            if total > 0:
                dominant_op = counts[tile].argmax()
                purity = counts[tile, dominant_op] / total
                analysis[tile] = {
                    'dominant_op': int(dominant_op),
                    'dominant_op_name': OPS[dominant_op][0],
                    'purity': float(purity),
                    'counts': {OPS[i][0]: int(counts[tile, i]) for i in range(NUM_OPS)},
                    'total': int(total),
                }
        
        return analysis
    
    def reset_tracking(self):
        self.tile_op_counts.zero_()


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, data, optimizer, device, routing_weight=0.1):
    model.train()
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    total_loss = 0
    total_bits_correct = 0
    total_bits = 0
    
    batch_size = CONFIG['batch_size']
    
    for i in range(0, len(data), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = [data[j] for j in batch_idx]
        
        op = torch.tensor([d['op'] for d in batch], device=device)
        a = torch.tensor([d['a'] for d in batch], device=device)
        b = torch.tensor([d['b'] for d in batch], device=device)
        
        # Pad result_bits to 6 bits
        target_bits = []
        for d in batch:
            bits = d['result_bits']
            # Pad to 6 bits
            while len(bits) < 6:
                bits = bits + [0]
            target_bits.append(bits)
        target_bits = torch.tensor(target_bits, device=device, dtype=torch.float)
        
        # Forward
        logits, routing_info, aux = model(op, a, b)
        
        # BCE loss on bits
        loss = F.binary_cross_entropy_with_logits(logits, target_bits)
        
        # Routing consistency loss: encourage same op -> same tile
        # Use tile_idx to predict op (if routing is clean, this should be easy)
        if 'tile_idx' in routing_info and routing_weight > 0:
            tile_idx = routing_info['tile_idx'].squeeze(-1)  # (batch,)
            # One-hot encode tiles and use as features to predict op
            # If tiles specialize, this prediction should be easy
            tile_onehot = F.one_hot(tile_idx, model.num_tiles).float()
            op_pred = tile_onehot @ model.tile_op_counts.float()  # Soft prediction
            # Don't backprop through counts, just through the routing
            # Alternative: add a small classifier head
        
        # Add aux losses
        if 'total_aux' in aux:
            loss = loss + 0.01 * aux['total_aux']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Bit accuracy
        pred_bits = (torch.sigmoid(logits) > 0.5).float()
        total_bits_correct += (pred_bits == target_bits).sum().item()
        total_bits += target_bits.numel()
    
    return {
        'loss': total_loss / (len(data) // batch_size + 1),
        'bit_accuracy': total_bits_correct / total_bits,
    }


def evaluate(model, data, device):
    model.eval()
    
    correct = 0
    total = 0
    
    per_op_correct = defaultdict(int)
    per_op_total = defaultdict(int)
    
    with torch.no_grad():
        for d in data:
            op = torch.tensor([d['op']], device=device)
            a = torch.tensor([d['a']], device=device)
            b = torch.tensor([d['b']], device=device)
            
            logits, _, _ = model(op, a, b, track=False)
            pred_bits = (torch.sigmoid(logits[0]) > 0.5).tolist()
            
            # Decode based on operation
            if d['op_name'] == 'ADD':
                pred_result = decode_bits(pred_bits[:5], signed=False)
            else:
                pred_result = decode_bits(pred_bits[:6], signed=True)
            
            is_correct = (pred_result == d['result'])
            correct += is_correct
            total += 1
            
            per_op_correct[d['op_name']] += is_correct
            per_op_total[d['op_name']] += 1
    
    per_op_acc = {op: per_op_correct[op] / per_op_total[op] for op in per_op_total}
    
    return {
        'accuracy': correct / total,
        'per_op_accuracy': per_op_acc,
    }


# =============================================================================
# Main
# =============================================================================

def run_single_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data = generate_data(CONFIG['value_range'])
    np.random.shuffle(data)
    
    # Proof of concept: use all data
    train_data = data
    test_data = data
    
    print(f"  All data: {len(data)} (proof of concept)")
    print(f"  Ops: {[OPS[i][0] for i in range(NUM_OPS)]}")
    
    model = PureTriXMicroOps(
        value_range=CONFIG['value_range'],
        num_ops=NUM_OPS,
        d_model=CONFIG['d_model'],
        num_tiles=CONFIG['num_tiles'],
        num_freqs=CONFIG['num_freqs'],
        num_layers=CONFIG.get('num_layers', 1),
        use_v1=CONFIG.get('use_v1', False),
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    best_acc = 0
    
    for epoch in range(CONFIG['epochs']):
        model.reset_tracking()
        train_metrics = train_epoch(model, train_data, optimizer, device)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            test_metrics = evaluate(model, test_data, device)
            
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
            
            print(f"  [Seed {seed}] Epoch {epoch + 1}: "
                  f"loss={train_metrics['loss']:.4f}, "
                  f"bits={train_metrics['bit_accuracy']:.1%}, "
                  f"acc={test_metrics['accuracy']:.1%} "
                  f"({', '.join(f'{k}:{v:.0%}' for k,v in test_metrics['per_op_accuracy'].items())})")
            
            if test_metrics['accuracy'] >= 0.99:
                print(f"  [Seed {seed}] 99%+ achieved!")
                break
    
    # Final evaluation and tile analysis
    model.reset_tracking()
    model.train()
    for d in train_data:
        op = torch.tensor([d['op']], device=device)
        a = torch.tensor([d['a']], device=device)
        b = torch.tensor([d['b']], device=device)
        model(op, a, b, track=True)
    
    tile_analysis = model.get_tile_specialization()
    final_metrics = evaluate(model, test_data, device)
    
    return {
        'seed': seed,
        'final_metrics': final_metrics,
        'tile_analysis': tile_analysis,
        'best_acc': best_acc,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    print("\n" + "=" * 70)
    print("PURE TRIX MICRO-OPS")
    print("Operations: ADD, SUB")
    print("Routing has meaning: select which op to apply")
    print("=" * 70)
    
    print("\n[TRAINING]")
    all_results = []
    
    for seed in CONFIG['seeds']:
        print(f"\nSeed {seed}:")
        result = run_single_seed(seed, device)
        all_results.append(result)
    
    # Aggregate
    accuracies = [r['final_metrics']['accuracy'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n" + "=" * 70)
    print("RESULTS: PURE TRIX MICRO-OPS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    
    # Per-operation accuracy
    best_run = max(all_results, key=lambda r: r['final_metrics']['accuracy'])
    print(f"\nPer-operation accuracy (best run, seed {best_run['seed']}):")
    for op_name, acc in best_run['final_metrics']['per_op_accuracy'].items():
        print(f"  {op_name}: {acc:.1%}")
    
    # Tile specialization - THE KEY METRIC
    print("\n" + "-" * 50)
    print("TILE SPECIALIZATION (the key metric)")
    print("-" * 50)
    
    tile_analysis = best_run['tile_analysis']
    
    # Count tiles that specialize to each operation
    op_specialists = defaultdict(list)
    high_purity_tiles = 0
    
    for tile, info in sorted(tile_analysis.items()):
        purity = info['purity']
        dom_op = info['dominant_op_name']
        
        if purity >= 0.8:
            high_purity_tiles += 1
            op_specialists[dom_op].append((tile, purity))
        
        purity_bar = "█" * int(purity * 20)
        print(f"  Tile {tile}: {dom_op:4s} purity={purity:.0%} {purity_bar} "
              f"(n={info['total']})")
    
    print(f"\nHigh-purity tiles (≥80%): {high_purity_tiles}/{len(tile_analysis)}")
    
    for op_name in OPS.values():
        op_name = op_name[0]
        specialists = op_specialists.get(op_name, [])
        if specialists:
            print(f"  {op_name} specialists: {[f'T{t}({p:.0%})' for t, p in specialists]}")
    
    # Verdict
    print("\n" + "=" * 70)
    if mean_acc >= 0.95 and high_purity_tiles >= 2:
        print("✓ PURE TRIX MICRO-OPS: SUCCESS")
        print(f"  Accuracy: {mean_acc:.1%}")
        print(f"  Tile specialization: {high_purity_tiles} tiles at ≥80% purity")
        print("  ROUTING HAS MEANING - tiles specialize to operations")
    elif mean_acc >= 0.90:
        print("◐ PARTIAL SUCCESS")
        print(f"  Accuracy: {mean_acc:.1%}")
        print(f"  Tile specialization: {high_purity_tiles} tiles at ≥80% purity")
    else:
        print("✗ NEEDS MORE WORK")
        print(f"  Accuracy: {mean_acc:.1%}")
    print("=" * 70)
    
    # Save results
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'pure_trix_microops_{timestamp}.json'
    
    results_json = {
        'config': CONFIG,
        'ops': {str(k): v[0] for k, v in OPS.items()},
        'timestamp': timestamp,
        'aggregate': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'all_accuracies': [float(a) for a in accuracies],
        },
        'best_run': {
            'seed': best_run['seed'],
            'accuracy': float(best_run['final_metrics']['accuracy']),
            'per_op': {k: float(v) for k, v in best_run['final_metrics']['per_op_accuracy'].items()},
            'tile_analysis': {str(k): {
                'dominant_op': v['dominant_op_name'],
                'purity': v['purity'],
                'total': v['total'],
            } for k, v in best_run['tile_analysis'].items()},
        },
        'high_purity_tiles': high_purity_tiles,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return mean_acc, high_purity_tiles


if __name__ == "__main__":
    main()
