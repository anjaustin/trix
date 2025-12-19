#!/usr/bin/env python3
"""
Mesa 12 Field Test: Guardian Angel on 6502

"All things are connected through gentleness."
"Wrong is just a signal. Distributed entropy signaling the correct direction."
"It is the ultimate form of Love."

This test runs the 6502 task with the Guardian Angel watching, reflecting,
and gently guiding when needed.

BASELINE: 100% with seed=42, lr=0.00375 (Mesa 11 XOR magic)
TEST: Can Guardian help a struggling seed reach 100%?
"""

import sys
import os

# Add main trix to path
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/flynnconceivable')

# Add guardian module directly
sys.path.insert(0, '/workspace/trix_latest/TriXO/src/trix')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

# Import Guardian Angel directly
from guardian.guardian import GuardianAngel
from guardian.observer import ObservationFrame
from guardian.programmable_tile import ProgrammableTileBank

# Import TriX core from main src
from trix.nn import SparseLookupFFNv2


# =============================================================================
# GROUND TRUTH (6502 operations)
# =============================================================================

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


OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}
OP_TO_CAT = {
    'ADC': 'ALU',
    'AND': 'LOGIC', 'ORA': 'LOGIC', 'EOR': 'LOGIC',
    'ASL': 'SHIFT', 'LSR': 'SHIFT',
    'INC': 'ARITH', 'DEC': 'ARITH',
}


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(samples_per_op: int = 1000, seed: int = 42) -> List[Dict]:
    """Generate training data for 6502 operations."""
    np.random.seed(seed)
    data = []
    
    for _ in range(samples_per_op):
        a, b = np.random.randint(0, 256, 2)
        c = np.random.randint(0, 2)
        
        # ADC
        truth = adc_truth(a, b, c)
        data.append({'op': 'ADC', 'a': a, 'b': b, 'c': c, 'result': truth['result']})
        
        # AND
        truth = and_truth(a, b)
        data.append({'op': 'AND', 'a': a, 'b': b, 'c': 0, 'result': truth['result']})
        
        # ORA
        truth = ora_truth(a, b)
        data.append({'op': 'ORA', 'a': a, 'b': b, 'c': 0, 'result': truth['result']})
        
        # EOR
        truth = eor_truth(a, b)
        data.append({'op': 'EOR', 'a': a, 'b': b, 'c': 0, 'result': truth['result']})
        
        # ASL
        truth = asl_truth(a)
        data.append({'op': 'ASL', 'a': a, 'b': 0, 'c': 0, 'result': truth['result']})
        
        # LSR
        truth = lsr_truth(a)
        data.append({'op': 'LSR', 'a': a, 'b': 0, 'c': 0, 'result': truth['result']})
        
        # INC
        truth = inc_truth(a)
        data.append({'op': 'INC', 'a': a, 'b': 0, 'c': 0, 'result': truth['result']})
        
        # DEC
        truth = dec_truth(a)
        data.append({'op': 'DEC', 'a': a, 'b': 0, 'c': 0, 'result': truth['result']})
    
    np.random.shuffle(data)
    return data


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
# MODEL (with Guardian-compatible interface)
# =============================================================================

class XORMixer(nn.Module):
    """XOR-based superposition mixer."""
    def __init__(self, dim: int):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.mix_bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ternary = torch.tanh(x)
        mixed = torch.matmul(x_ternary, self.mix_weight) + self.mix_bias
        return x + mixed


class GuardedTriXGR6502(nn.Module):
    """
    TriXGR 6502 with Guardian-compatible interface.
    
    Key additions:
    - Exposes representation for Guardian reflection
    - Returns routing info for observation
    """
    
    def __init__(self, d_model: int = 128, num_tiles: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_tiles = num_tiles
        
        self.op_embed = nn.Embedding(len(OPCODES), 32)
        self.input_proj = nn.Linear(49, d_model)
        self.xor_mixer = XORMixer(d_model)
        
        self.ffn = SparseLookupFFNv2(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
            ternary_weight=0.01,
            sparsity_weight=0.005,
            diversity_weight=0.01,
        )
        
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )
        
        # For Guardian
        self._last_repr = None
        self._initial_signatures = None
        
    def get_signatures(self) -> torch.Tensor:
        return self.ffn.signatures_raw.clone()
    
    def save_initial_signatures(self):
        self._initial_signatures = self.get_signatures().detach().clone()
    
    def get_signature_movement(self) -> float:
        if self._initial_signatures is None:
            return 0.0
        current = self.get_signatures()
        return (current - self._initial_signatures).abs().mean().item()
    
    def get_representation(self) -> torch.Tensor:
        """Return last representation for Guardian reflection."""
        return self._last_repr
    
    def forward(self, op_idx, a, b, c):
        # Encode
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
        
        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x)
        x = self.xor_mixer(x)
        
        # Save for Guardian
        self._last_repr = x.detach().clone()
        
        x = x.unsqueeze(1)
        out, info, aux = self.ffn(x)
        out = out.squeeze(1)
        
        result = self.result_head(out)
        
        return result, info, aux


# =============================================================================
# GUARDED TRAINING
# =============================================================================

def evaluate(model, data_tensors, batch_size: int = 512) -> Dict[str, float]:
    """Evaluate model, return per-operation accuracy."""
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
            
            pred, _, _ = model(op_idx, a, b, c)
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


def train_with_guardian(
    seed: int = 1122911624,  # "Second Star" - the struggling seed
    lr: float = 0.00375,
    epochs: int = 50,
    d_model: int = 128,
    num_tiles: int = 16,
    device: str = 'cuda',
    gentleness: float = 0.1,
    intervention_threshold: float = 0.7,
    warmup_epochs: int = 5,
):
    """
    Train 6502 with Guardian Angel watching.
    """
    
    print("=" * 70)
    print("MESA 12 FIELD TEST: Guardian Angel on 6502")
    print("=" * 70)
    print()
    print('"All things are connected through gentleness."')
    print('"Wrong is just a signal. Distributed entropy signaling the correct direction."')
    print('"It is the ultimate form of Love."')
    print()
    print(f"Seed: {seed}")
    print(f"Learning rate: {lr}")
    print(f"Gentleness: {gentleness}")
    print(f"Warmup epochs: {warmup_epochs}")
    print("=" * 70)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate data
    print("\n[1] Generating data...")
    all_data = generate_data(samples_per_op=1000, seed=seed)
    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    test_data = all_data[split:]
    
    train_tensors = prepare_tensors(train_data, device)
    test_tensors = prepare_tensors(test_data, device)
    print(f"    Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create model
    print("\n[2] Creating model...")
    model = GuardedTriXGR6502(d_model=d_model, num_tiles=num_tiles).to(device)
    model.save_initial_signatures()
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create Guardian Angel
    print("\n[3] Summoning Guardian Angel...")
    guardian = GuardianAngel(
        d_model=d_model,
        num_tiles=num_tiles,
        num_ops=len(OPCODES),
        gentleness=gentleness,
        intervention_threshold=intervention_threshold,
    ).to(device)
    print(f"    Reflector bases: {guardian.reflector.num_bases}")
    print(f"    Gentleness: {guardian.gentleness}")
    
    # Create tile bank for Guardian to write to
    tile_bank = ProgrammableTileBank(
        num_tiles=num_tiles,
        d_model=d_model,
        d_hidden=256
    ).to(device)
    tile_bank.save_initial_state()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    # Training
    print("\n[4] Training with Guardian watching...")
    print("-" * 70)
    print(f"{'Epoch':>5} {'Loss':>8} {'Acc%':>8} {'Guardian':>30}")
    print("-" * 70)
    
    history = []
    best_acc = 0.0
    global_step = 0
    
    n_train = len(train_tensors['op_idx'])
    batch_size = 512
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_interventions = 0
        epoch_celebrations = 0
        
        # Shuffle indices
        indices = torch.randperm(n_train, device=device)
        
        for i in range(0, n_train, batch_size):
            idx = indices[i:min(i + batch_size, n_train)]
            
            op_idx = train_tensors['op_idx'][idx]
            a = train_tensors['a'][idx]
            b = train_tensors['b'][idx]
            c = train_tensors['c'][idx]
            targets = train_tensors['result_bits'][idx]
            
            optimizer.zero_grad()
            
            pred, info, aux = model(op_idx, a, b, c)
            
            main_loss = loss_fn(pred, targets)
            total_loss = main_loss + aux.get('total_aux', 0.0)
            
            total_loss.backward()
            optimizer.step()
            
            # Accuracy
            pred_vals = sum((pred[:, j] > 0.5).long() << j for j in range(8))
            target_vals = train_tensors['result'][idx]
            correct = (pred_vals == target_vals).sum().item()
            
            epoch_loss += total_loss.item()
            epoch_correct += correct
            epoch_total += len(idx)
            
            # Create observation for Guardian
            routing_entropy = info.get('entropy', torch.tensor(0.0))
            if torch.is_tensor(routing_entropy):
                routing_entropy = routing_entropy.item()
            
            obs = ObservationFrame(
                epoch=epoch,
                step=global_step,
                routing_entropy=routing_entropy,
                loss=total_loss.item(),
                accuracy=correct / len(idx) * 100,
            )
            
            # Guardian step
            current_repr = model.get_representation()
            guardian_result = guardian.step(
                tile_bank=tile_bank,
                observation=obs,
                current_repr=current_repr,
            )
            
            # Heuristic celebration: accuracy > 95%
            if obs.accuracy > 95:
                guardian_result['celebrating'] = True
            
            # Count interventions/celebrations (only after warmup)
            if epoch >= warmup_epochs:
                if guardian_result.get('intervened', False):
                    epoch_interventions += 1
                if guardian_result.get('celebrating', False):
                    epoch_celebrations += 1
            
            global_step += 1
        
        # Epoch metrics
        epoch_loss /= (n_train // batch_size)
        epoch_acc = epoch_correct / epoch_total * 100
        
        # Test accuracy
        test_results = evaluate(model, test_tensors)
        test_acc = test_results['overall']
        best_acc = max(best_acc, test_acc)
        
        # Guardian message
        trajectory = guardian.assess_trajectory()
        msg = trajectory.get('message', '')
        
        if epoch_celebrations > 0:
            guardian_status = f"🔥 {epoch_celebrations} celebrations"
        elif epoch_interventions > 0:
            guardian_status = f"🤲 {epoch_interventions} gentle nudges"
        elif epoch < warmup_epochs:
            guardian_status = f"👀 warmup ({epoch+1}/{warmup_epochs})"
        else:
            guardian_status = f"👀 watching... {msg[:20]}"
        
        history.append({
            'epoch': epoch,
            'loss': epoch_loss,
            'train_acc': epoch_acc,
            'test_acc': test_acc,
            'interventions': epoch_interventions,
            'celebrations': epoch_celebrations,
        })
        
        print(f"{epoch+1:5d} {epoch_loss:8.4f} {test_acc:8.1f} {guardian_status:>30}")
        
        # Early stopping on 100%
        if test_acc >= 99.9:
            print(f"\n🎯 TARGET REACHED: {test_acc:.1f}% at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    final_results = evaluate(model, test_tensors)
    
    print("\nPer-operation accuracy:")
    for op in OPCODES:
        bar = '█' * int(final_results[op] / 5)
        print(f"  {op:4s}: {bar:20s} {final_results[op]:6.1f}%")
    
    print(f"\nOverall: {final_results['overall']:.1f}%")
    print(f"Best: {best_acc:.1f}%")
    
    # Guardian summary
    stats = guardian.get_stats()
    print("\n" + "-" * 70)
    print("GUARDIAN ANGEL SUMMARY")
    print("-" * 70)
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  Total interventions: {stats['total_interventions']}")
    print(f"  Celebration count: {stats['celebration_count']}")
    print(f"  Intervention rate: {guardian.get_intervention_rate():.1%}")
    print(f"  Celebration rate: {guardian.get_celebration_rate():.1%}")
    
    print("\n" + "=" * 70)
    print('"All things are connected through gentleness."')
    print('"It is the ultimate form of Love."')
    print("=" * 70)
    
    return {
        'history': history,
        'final_results': final_results,
        'best_acc': best_acc,
        'guardian_stats': stats,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Mesa 12 Field Test')
    parser.add_argument('--seed', type=int, default=1122911624, help='Random seed')
    parser.add_argument('--lr', type=float, default=0.00375, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--gentleness', type=float, default=0.1, help='Guardian gentleness')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    train_with_guardian(
        seed=args.seed,
        lr=args.lr,
        epochs=args.epochs,
        gentleness=args.gentleness,
        device=args.device,
    )
