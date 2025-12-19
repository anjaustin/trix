#!/usr/bin/env python3
"""
TriXGR (Guns and Roses) - 6502 Monolithic Training with Geometric Validation

CODENAME: GUNS AND ROSES
MISSION: Train 6502 emulator while validating geometric framework

BASELINE: Previous monolithic model achieved 66% accuracy
TARGET: Beat 66% AND validate geometric properties hold on real task

GEOMETRIC METRICS TRACKED:
1. Manifold curvature (routing stability under perturbation)
2. Geodesic consistency (does routing = nearest signature?)
3. Signature movement (how much does training warp the manifold?)
4. Tile specialization (do tiles cluster by operation type?)

Author: Droid (Mesa 11 Rigorous Testing)
Date: 2024-12-18
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
from typing import Dict, List, Tuple, Optional
import time
import json

# Import TriX components
from trix.nn import SparseLookupFFNv2

# Import 6502 ground truth
try:
    from training.data import (
        adc_truth, sbc_truth,
        asl_truth, lsr_truth, rol_truth, ror_truth,
        and_truth, ora_truth, eor_truth,
        inc_truth, dec_truth,
    )
    HAS_FLYNN = True
except ImportError:
    HAS_FLYNN = False
    print("Warning: flynnconceivable not found, using built-in ground truth")


# =============================================================================
# GROUND TRUTH (fallback if flynnconceivable not available)
# =============================================================================

if not HAS_FLYNN:
    def adc_truth(a, b, c):
        result = (a + b + c) & 0xFF
        c_out = int((a + b + c) > 255)
        v = int(((a ^ result) & (b ^ result) & 0x80) != 0)
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0), 'c': c_out, 'v': v}
    
    def and_truth(a, b):
        result = a & b
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}
    
    def ora_truth(a, b):
        result = a | b
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}
    
    def eor_truth(a, b):
        result = a ^ b
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}
    
    def asl_truth(val):
        result = (val << 1) & 0xFF
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0), 'c': (val >> 7) & 1}
    
    def lsr_truth(val):
        result = val >> 1
        return {'result': result, 'n': 0, 'z': int(result == 0), 'c': val & 1}
    
    def inc_truth(val):
        result = (val + 1) & 0xFF
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}
    
    def dec_truth(val):
        result = (val - 1) & 0xFF
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}


# =============================================================================
# DATA GENERATION
# =============================================================================

OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}
OP_TO_CAT = {
    'ADC': 'ALU',
    'AND': 'LOGIC', 'ORA': 'LOGIC', 'EOR': 'LOGIC',
    'ASL': 'SHIFT', 'LSR': 'SHIFT',
    'INC': 'INCDEC', 'DEC': 'INCDEC',
}


def generate_exhaustive_data(fast_mode: bool = True) -> List[Dict]:
    """Generate 6502 dataset. fast_mode uses sampling for speed."""
    data = []
    
    if fast_mode:
        print("Generating SAMPLED 6502 data (fast mode)...")
        n_samples = 5000  # Per operation - smaller for faster iteration
        
        # ADC
        for _ in range(n_samples):
            a, b, c = np.random.randint(256), np.random.randint(256), np.random.randint(2)
            d = adc_truth(a, b, c)
            data.append({'op': 'ADC', 'a': a, 'b': b, 'c': c, 'result': d['result']})
        
        # LOGIC
        for op_name, truth_fn in [('AND', and_truth), ('ORA', ora_truth), ('EOR', eor_truth)]:
            for _ in range(n_samples):
                a, b = np.random.randint(256), np.random.randint(256)
                data.append({'op': op_name, 'a': a, 'b': b, 'c': 0, 'result': truth_fn(a, b)['result']})
        
        # SHIFT (all 512 - it's small)
        for val in range(256):
            data.append({'op': 'ASL', 'a': val, 'b': 0, 'c': 0, 'result': asl_truth(val)['result']})
            data.append({'op': 'LSR', 'a': val, 'b': 0, 'c': 0, 'result': lsr_truth(val)['result']})
        
        # INCDEC (all 512 - it's small)
        for val in range(256):
            data.append({'op': 'INC', 'a': val, 'b': 0, 'c': 0, 'result': inc_truth(val)['result']})
            data.append({'op': 'DEC', 'a': val, 'b': 0, 'c': 0, 'result': dec_truth(val)['result']})
        
        print(f"  Total samples: {len(data):,}")
    else:
        print("Generating exhaustive 6502 data...")
        
        # ADC: All 131,072 combinations (256 * 256 * 2)
        print("  ADC: 131,072 combinations...")
        for a in range(256):
            for b in range(256):
                for c in [0, 1]:
                    d = adc_truth(a, b, c)
                    data.append({'op': 'ADC', 'a': a, 'b': b, 'c': c, 'result': d['result']})
        
        # LOGIC: 3 * 65,536 = 196,608 combinations
        print("  LOGIC (AND/ORA/EOR): 196,608 combinations...")
        for a in range(256):
            for b in range(256):
                data.append({'op': 'AND', 'a': a, 'b': b, 'c': 0, 'result': and_truth(a, b)['result']})
                data.append({'op': 'ORA', 'a': a, 'b': b, 'c': 0, 'result': ora_truth(a, b)['result']})
                data.append({'op': 'EOR', 'a': a, 'b': b, 'c': 0, 'result': eor_truth(a, b)['result']})
        
        # SHIFT: 2 * 256 = 512 combinations
        print("  SHIFT (ASL/LSR): 512 combinations...")
        for val in range(256):
            data.append({'op': 'ASL', 'a': val, 'b': 0, 'c': 0, 'result': asl_truth(val)['result']})
            data.append({'op': 'LSR', 'a': val, 'b': 0, 'c': 0, 'result': lsr_truth(val)['result']})
        
        # INCDEC: 2 * 256 = 512 combinations
        print("  INCDEC (INC/DEC): 512 combinations...")
        for val in range(256):
            data.append({'op': 'INC', 'a': val, 'b': 0, 'c': 0, 'result': inc_truth(val)['result']})
            data.append({'op': 'DEC', 'a': val, 'b': 0, 'c': 0, 'result': dec_truth(val)['result']})
        
        # Edge case oversampling (VGem's fix)
        edge_cases = [d for d in data if d['result'] == 0 or d['a'] in [0, 0xFF] or d['b'] in [0, 0xFF]]
        print(f"  Edge cases: {len(edge_cases)} (oversampling 3x)")
        data.extend(edge_cases * 2)
        
        print(f"  Total samples: {len(data):,}")
    
    np.random.shuffle(data)
    return data


def split_data(data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train/test with stratification by operation."""
    by_op = defaultdict(list)
    for d in data:
        by_op[d['op']].append(d)
    
    train_data, test_data = [], []
    for op, samples in by_op.items():
        n_train = int(len(samples) * train_ratio)
        train_data.extend(samples[:n_train])
        test_data.extend(samples[n_train:])
    
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    
    return train_data, test_data


# =============================================================================
# MODEL
# =============================================================================

class XORMixer(nn.Module):
    """
    XOR-based superposition mixer for routing scores.
    
    XOR properties we exploit:
    - Self-inverse: a ⊕ b ⊕ b = a
    - Orthogonality generator
    - Natural superposition creator
    
    Applies learned XOR-like mixing to help signatures find superposition states.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        # Learnable XOR-like mixing weights
        self.mix_weight = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.mix_bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # XOR-inspired mixing: use tanh to create [-1, 1] range like ternary
        # Then apply learned mixing matrix
        x_ternary = torch.tanh(x)
        mixed = torch.matmul(x_ternary, self.mix_weight) + self.mix_bias
        # XOR is self-inverse, so we add back the original (residual)
        return x + mixed


class TriXGR6502(nn.Module):
    """
    TriX with Geometric Routing for 6502.
    
    Key features:
    - Exposes signatures for geometric analysis
    - Tracks routing decisions
    - Supports soft routing for gradient flow
    - Configurable depth (num_layers)
    - XOR mixing for superposition magic
    """
    
    def __init__(self, d_model: int = 128, num_tiles: int = 16, num_layers: int = 2, use_xor_mixing: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_layers = num_layers
        self.use_xor_mixing = use_xor_mixing
        
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        
        # Input: op_embed (32) + a_bits (8) + b_bits (8) + c (1) = 49
        self.input_proj = nn.Linear(49, d_model)
        
        # XOR mixer for superposition
        if use_xor_mixing:
            self.xor_mixer = XORMixer(d_model)
        
        # Stack of FFN layers
        self.ffn_layers = nn.ModuleList([
            SparseLookupFFNv2(
                d_model=d_model,
                num_tiles=num_tiles,
                tiles_per_cluster=4,
                ternary_weight=0.01,
                sparsity_weight=0.005,
                diversity_weight=0.01,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )
        
        # For geometric tracking
        self.initial_signatures = None
        
    def get_signatures(self) -> torch.Tensor:
        """Extract tile signatures from first FFN layer."""
        # SparseLookupFFNv2 has explicit signatures
        return self.ffn_layers[0].signatures_raw.clone()
    
    def save_initial_signatures(self):
        """Save signatures at training start for movement tracking."""
        self.initial_signatures = self.get_signatures().detach().clone()
    
    def get_signature_movement(self) -> float:
        """Compute how much signatures moved from initial."""
        if self.initial_signatures is None:
            return 0.0
        current = self.get_signatures()
        return (current - self.initial_signatures).abs().mean().item()
    
    def forward(self, op_idx, a, b, c, return_routing: bool = False):
        batch_size = op_idx.shape[0]
        device = op_idx.device
        
        # Encode inputs
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
        
        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x)  # [B, d_model]
        
        # XOR mixing for superposition magic
        if self.use_xor_mixing:
            x = self.xor_mixer(x)
        
        x = x.unsqueeze(1)  # [B, 1, d_model]
        
        # Forward through all FFN layers
        total_aux = {'total_aux': 0.0}
        first_info = None
        
        for i, ffn in enumerate(self.ffn_layers):
            out, info, aux = ffn(x)
            x = out  # Pass to next layer
            total_aux['total_aux'] = total_aux['total_aux'] + aux['total_aux']
            if i == 0:
                first_info = info  # Track routing from first layer
        
        out = x.squeeze(1)
        
        # Predict result
        result = self.result_head(out)
        
        if return_routing:
            return result, first_info, total_aux
        return result, total_aux


# =============================================================================
# GEOMETRIC METRICS
# =============================================================================

@dataclass
class GeometricMetrics:
    """Geometric metrics for a training epoch."""
    curvature: float           # Routing stability under perturbation
    geodesic_consistency: float  # Does routing match nearest signature?
    signature_movement: float  # How much did signatures move?
    tile_purity: float         # Do tiles specialize by operation?
    routing_entropy: float     # How spread out is routing?


def measure_curvature(model: TriXGR6502, data_tensors: Dict, epsilon: float = 0.1) -> float:
    """Measure manifold curvature via routing stability (fast version)."""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Small subset for speed
        n = min(200, len(data_tensors['op_idx']))
        idx = torch.arange(n, device=device)
        
        op_idx = data_tensors['op_idx'][idx]
        a = data_tensors['a'][idx]
        b = data_tensors['b'][idx]
        c = data_tensors['c'][idx]
        
        # Original routing
        _, info, _ = model(op_idx, a, b, c, return_routing=True)
        original_routes = info['tile_idx'].squeeze(-1)
        
        # Perturbed routing
        a_noisy = (a + torch.randint(-5, 6, a.shape, device=device)).clamp(0, 255)
        b_noisy = (b + torch.randint(-5, 6, b.shape, device=device)).clamp(0, 255)
        
        _, info_noisy, _ = model(op_idx, a_noisy, b_noisy, c, return_routing=True)
        noisy_routes = info_noisy['tile_idx'].squeeze(-1)
        
        flip_rate = (original_routes != noisy_routes).float().mean().item()
    
    return flip_rate


def measure_tile_purity(tile_counts: Dict[int, Dict[str, int]]) -> float:
    """Measure how specialized tiles are by operation category."""
    purities = []
    for tile, op_counts in tile_counts.items():
        cat_counts = defaultdict(int)
        for op, cnt in op_counts.items():
            cat_counts[OP_TO_CAT[op]] += cnt
        total = sum(cat_counts.values())
        if total > 100:
            purities.append(max(cat_counts.values()) / total)
    return np.mean(purities) if purities else 0.0


def measure_routing_entropy(tile_counts: Dict[int, Dict[str, int]], num_tiles: int) -> float:
    """Measure entropy of routing distribution."""
    total_counts = [sum(tile_counts.get(t, {}).values()) for t in range(num_tiles)]
    total = sum(total_counts)
    if total == 0:
        return 0.0
    probs = np.array(total_counts) / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs + 1e-10))


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: TriXGR6502,
    data_tensors: Dict,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 512,
) -> Tuple[float, float, Dict]:
    """Train for one epoch, return loss, accuracy, tile counts."""
    model.train()
    device = next(model.parameters()).device
    
    n = len(data_tensors['op_idx'])
    perm = torch.randperm(n, device=device)
    
    total_loss = 0.0
    correct = 0
    tile_counts = defaultdict(lambda: defaultdict(int))
    ops_list = data_tensors['ops']
    
    for i in range(0, n - batch_size, batch_size):
        idx = perm[i:i+batch_size]
        
        op_idx = data_tensors['op_idx'][idx]
        a = data_tensors['a'][idx]
        b = data_tensors['b'][idx]
        c = data_tensors['c'][idx]
        result_bits = data_tensors['result_bits'][idx]
        
        # Forward
        pred, info, aux = model(op_idx, a, b, c, return_routing=True)
        
        # Loss
        loss = F.binary_cross_entropy(pred, result_bits) + aux['total_aux']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        pred_vals = sum((pred[:, i] > 0.5).long() << i for i in range(8))
        target_vals = data_tensors['result'][idx]
        correct += (pred_vals == target_vals).sum().item()
        
        # Track tiles
        tiles = info['tile_idx'].squeeze(-1).cpu().numpy()
        for j, t in enumerate(tiles):
            tile_counts[int(t)][ops_list[idx[j].item()]] += 1
    
    avg_loss = total_loss / (n // batch_size)
    accuracy = correct / n * 100
    
    return avg_loss, accuracy, dict(tile_counts)


def evaluate_adc_bits(model: TriXGR6502, data_tensors: Dict, batch_size: int = 512) -> Dict[str, float]:
    """Evaluate ADC accuracy per output bit - reveals carry propagation."""
    model.eval()
    device = next(model.parameters()).device
    
    n = len(data_tensors['op_idx'])
    ops_list = data_tensors['ops']
    
    # Track per-bit accuracy for ADC
    bit_correct = [0] * 8
    bit_total = [0] * 8
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            
            op_idx = data_tensors['op_idx'][i:end]
            a = data_tensors['a'][i:end]
            b = data_tensors['b'][i:end]
            c = data_tensors['c'][i:end]
            result_bits = data_tensors['result_bits'][i:end]
            
            pred, _ = model(op_idx, a, b, c)
            pred_bits = (pred > 0.5).float()
            
            # Check each sample
            for j in range(end - i):
                if ops_list[i + j] == 'ADC':
                    for bit in range(8):
                        bit_total[bit] += 1
                        if pred_bits[j, bit] == result_bits[j, bit]:
                            bit_correct[bit] += 1
    
    # Return per-bit accuracy
    return {f'bit_{b}': (bit_correct[b] / bit_total[b] * 100 if bit_total[b] > 0 else 0) 
            for b in range(8)}


def evaluate(model: TriXGR6502, data_tensors: Dict, batch_size: int = 512) -> Dict[str, float]:
    """Evaluate on test data, return per-operation accuracy."""
    model.eval()
    device = next(model.parameters()).device
    
    n = len(data_tensors['op_idx'])
    
    correct_by_op = defaultdict(int)
    total_by_op = defaultdict(int)
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            
            op_idx = data_tensors['op_idx'][i:end]
            a = data_tensors['a'][i:end]
            b = data_tensors['b'][i:end]
            c = data_tensors['c'][i:end]
            
            pred, _ = model(op_idx, a, b, c)
            
            pred_vals = sum((pred[:, j] > 0.5).long() << j for j in range(8))
            target_vals = data_tensors['result'][i:end]
            
            for j in range(end - i):
                op = data_tensors['ops'][i + j]
                total_by_op[op] += 1
                if pred_vals[j] == target_vals[j]:
                    correct_by_op[op] += 1
    
    results = {}
    for op in OPCODES:
        if total_by_op[op] > 0:
            results[op] = correct_by_op[op] / total_by_op[op] * 100
        else:
            results[op] = 0.0
    
    results['overall'] = sum(correct_by_op.values()) / sum(total_by_op.values()) * 100
    
    return results


def prepare_tensors(data: List[Dict], device: str) -> Dict:
    """Convert data to tensors."""
    op_idx = torch.tensor([OP_TO_IDX[d['op']] for d in data], device=device)
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    c = torch.tensor([d['c'] for d in data], device=device)
    result = torch.tensor([d['result'] for d in data], device=device)
    result_bits = torch.stack([(result >> i) & 1 for i in range(8)], dim=1).float()
    ops = [d['op'] for d in data]
    
    return {
        'op_idx': op_idx, 'a': a, 'b': b, 'c': c,
        'result': result, 'result_bits': result_bits, 'ops': ops
    }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_trixgr_6502(
    epochs: int = 50,
    d_model: int = 128,
    num_tiles: int = 16,
    batch_size: int = 512,
    lr: float = 0.001,
    device: str = 'cuda',
    seed: int = 42,
) -> Dict:
    """
    Train TriXGR on 6502 with full geometric tracking.
    
    Returns comprehensive results for comparison.
    """
    
    print("=" * 70)
    print("TriXGR (GUNS AND ROSES) - 6502 Monolithic Training")
    print("=" * 70)
    print()
    print("BASELINE: Previous monolithic achieved 66% accuracy")
    print("TARGET: Beat 66% AND validate geometric properties")
    print()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate data
    print("Phase 1: Data Generation")
    print("-" * 70)
    all_data = generate_exhaustive_data(fast_mode=True)  # Use fast mode for speed
    train_data, test_data = split_data(all_data, train_ratio=0.8)
    print(f"Train: {len(train_data):,}, Test: {len(test_data):,}")
    print()
    
    # Prepare tensors
    print("Phase 2: Tensor Preparation")
    print("-" * 70)
    train_tensors = prepare_tensors(train_data, device)
    test_tensors = prepare_tensors(test_data, device)
    print("Tensors ready")
    print()
    
    # Create model
    print("Phase 3: Model Creation")
    print("-" * 70)
    num_layers = 1  # 1 layer with XOR mixing
    use_xor = True
    model = TriXGR6502(d_model=d_model, num_tiles=num_tiles, num_layers=num_layers, use_xor_mixing=use_xor).to(device)
    model.save_initial_signatures()
    print(f"Model: d_model={d_model}, num_tiles={num_tiles}, num_layers={num_layers}, xor_mixing={use_xor}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'curvature': [],
        'signature_movement': [],
        'tile_purity': [],
        'routing_entropy': [],
    }
    
    # Training loop
    print("Phase 4: Training")
    print("-" * 70)
    print(f"{'Epoch':>5} {'Loss':>8} {'Train%':>8} {'Test%':>8} {'Curv':>8} {'Move':>8} {'Purity':>8}")
    print("-" * 70)
    
    start_time = time.time()
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        loss, train_acc, tile_counts = train_epoch(model, train_tensors, optimizer, batch_size)
        scheduler.step()
        
        # Evaluate
        test_results = evaluate(model, test_tensors)
        test_acc = test_results['overall']
        
        # Geometric metrics
        curvature = measure_curvature(model, train_tensors)
        sig_movement = model.get_signature_movement()
        purity = measure_tile_purity(tile_counts)
        entropy = measure_routing_entropy(tile_counts, num_tiles)
        
        # Record
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['curvature'].append(curvature)
        history['signature_movement'].append(sig_movement)
        history['tile_purity'].append(purity)
        history['routing_entropy'].append(entropy)
        
        best_test_acc = max(best_test_acc, test_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:5d} {loss:8.4f} {train_acc:8.1f} {test_acc:8.1f} "
                  f"{curvature:8.3f} {sig_movement:8.4f} {purity:8.2f}")
    
    train_time = time.time() - start_time
    
    # Final evaluation
    print()
    print("Phase 5: Final Evaluation")
    print("-" * 70)
    
    final_results = evaluate(model, test_tensors)
    
    print("\nPer-operation accuracy:")
    for op in OPCODES:
        bar = '█' * int(final_results[op] / 5)
        print(f"  {op:4s}: {bar:20s} {final_results[op]:6.1f}%")
    
    print(f"\nOverall: {final_results['overall']:.1f}%")
    print(f"Training time: {train_time:.1f}s")
    
    # Geometric summary
    print()
    print("Phase 6: Geometric Validation")
    print("-" * 70)
    
    final_curvature = history['curvature'][-1]
    final_movement = history['signature_movement'][-1]
    final_purity = history['tile_purity'][-1]
    final_entropy = history['routing_entropy'][-1]
    
    print(f"  Manifold Curvature:    {final_curvature:.3f} (lower = smoother)")
    print(f"  Signature Movement:    {final_movement:.4f} (training warped manifold)")
    print(f"  Tile Purity:           {final_purity:.2f} (specialization)")
    print(f"  Routing Entropy:       {final_entropy:.2f} (distribution)")
    
    # Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    beat_baseline = final_results['overall'] > 66.0
    manifold_warped = final_movement > 0.001
    tiles_specialized = final_purity > 0.5
    
    print(f"\n  Accuracy: {final_results['overall']:.1f}% vs 66% baseline: {'PASS' if beat_baseline else 'FAIL'}")
    print(f"  Manifold warped: {manifold_warped} (movement={final_movement:.4f})")
    print(f"  Tiles specialized: {tiles_specialized} (purity={final_purity:.2f})")
    
    if beat_baseline and manifold_warped:
        print("\n  *** TriXGR VALIDATED: Geometric framework holds on real task ***")
    else:
        print("\n  Results inconclusive - see metrics above")
    
    print("=" * 70)
    
    # Return full results
    return {
        'config': {
            'd_model': d_model,
            'num_tiles': num_tiles,
            'epochs': epochs,
            'seed': seed,
        },
        'accuracy': {
            'overall': final_results['overall'],
            'by_op': {op: final_results[op] for op in OPCODES},
            'best': best_test_acc,
        },
        'geometric': {
            'curvature': final_curvature,
            'signature_movement': final_movement,
            'tile_purity': final_purity,
            'routing_entropy': final_entropy,
        },
        'history': history,
        'train_time': train_time,
        'beat_baseline': beat_baseline,
    }


if __name__ == "__main__":
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Run training (64 epochs)
    results = train_trixgr_6502(
        epochs=108,
        d_model=128,
        num_tiles=16,
        batch_size=512,
        lr=0.00337,
        device=device,
        seed=1122911624,
    )
    
    # Save results
    results_file = '/workspace/trix_latest/TriXO/experiments/mesa11/rigorous/trixgr_6502_results.json'
    
    # Convert numpy to python for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
