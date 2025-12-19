#!/usr/bin/env python3
"""
HALO Pipeline Field Test - The Four Phases of Mastery

Phase 1: EXPLORATION      - Entropic harmony, find where energy pools
Phase 2: EXPEDITION       - Map the nodes of interest  
Phase 3: CONVERGENCE      - Train for accuracy with informed init
Phase 4: MASTERY          - HALO ACTIVATES with full journey context

"I don't know how to help you yet.
 Let me watch you explore.  
 Let me see who you're becoming.
 THEN I'll know exactly what you need."

RLHF is dead. Long live HALO.
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/trix_latest/TriXO/src/trix')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import HALO components
from guardian.pipeline import HALOPipeline, Phase
from guardian.guardian import GuardianAngel
from guardian.programmable_tile import ProgrammableTileBank

# Import TriX
from trix.nn import SparseLookupFFNv2


# =============================================================================
# 6502 Ground Truth
# =============================================================================

def adc_truth(a, b, c):
    result = (a + b + c) & 0xFF
    return result

def and_truth(a, b): return a & b
def ora_truth(a, b): return a | b
def eor_truth(a, b): return a ^ b
def asl_truth(val): return (val << 1) & 0xFF
def lsr_truth(val): return val >> 1
def inc_truth(val): return (val + 1) & 0xFF
def dec_truth(val): return (val - 1) & 0xFF

OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(samples_per_op: int = 1000, seed: int = 42):
    np.random.seed(seed)
    data = []
    
    for _ in range(samples_per_op):
        a, b = np.random.randint(0, 256, 2)
        c = np.random.randint(0, 2)
        
        data.append({'op': 'ADC', 'a': a, 'b': b, 'c': c, 'result': adc_truth(a, b, c)})
        data.append({'op': 'AND', 'a': a, 'b': b, 'c': 0, 'result': and_truth(a, b)})
        data.append({'op': 'ORA', 'a': a, 'b': b, 'c': 0, 'result': ora_truth(a, b)})
        data.append({'op': 'EOR', 'a': a, 'b': b, 'c': 0, 'result': eor_truth(a, b)})
        data.append({'op': 'ASL', 'a': a, 'b': 0, 'c': 0, 'result': asl_truth(a)})
        data.append({'op': 'LSR', 'a': a, 'b': 0, 'c': 0, 'result': lsr_truth(a)})
        data.append({'op': 'INC', 'a': a, 'b': 0, 'c': 0, 'result': inc_truth(a)})
        data.append({'op': 'DEC', 'a': a, 'b': 0, 'c': 0, 'result': dec_truth(a)})
    
    np.random.shuffle(data)
    return data


def make_dataloader(data, device, batch_size=256):
    op_idx = torch.tensor([OP_TO_IDX[d['op']] for d in data], device=device)
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    c = torch.tensor([d['c'] for d in data], device=device)
    result = torch.tensor([d['result'] for d in data], device=device)
    result_bits = torch.stack([(result >> i) & 1 for i in range(8)], dim=1).float()
    
    dataset = TensorDataset(op_idx, a, b, c, result_bits)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =============================================================================
# Model
# =============================================================================

class XORMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.mix_bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        x_ternary = torch.tanh(x)
        mixed = torch.matmul(x_ternary, self.mix_weight) + self.mix_bias
        return x + mixed


class HALO6502Model(nn.Module):
    """Model compatible with HALO pipeline."""
    
    def __init__(self, d_model=128, num_tiles=16):
        super().__init__()
        self.d_model = d_model
        
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
        
    def forward(self, op_idx, a, b, c):
        # Encode
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()
        
        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x)
        x = self.xor_mixer(x)
        
        x = x.unsqueeze(1)
        out, info, aux = self.ffn(x)
        out = out.squeeze(1)
        
        result = self.result_head(out)
        
        return result, info, aux


# =============================================================================
# Main
# =============================================================================

def run_halo_pipeline(
    seed: int = 42,
    total_epochs: int = 128,
    phase_schedule: tuple = (32, 32, 32, 32),
    lr_schedule: tuple = (0.005, 0.003, 0.002, 0.001),
    device: str = 'cuda',
):
    """Run the complete HALO pipeline on 6502."""
    
    print("\n" + "🔥"*35)
    print("       HALO PIPELINE FIELD TEST: 6502 CPU EMULATION")
    print("🔥"*35 + "\n")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data
    print("[1] Generating data...")
    all_data = generate_data(samples_per_op=1000, seed=seed)
    split = int(len(all_data) * 0.8)
    train_loader = make_dataloader(all_data[:split], device)
    eval_loader = make_dataloader(all_data[split:], device)
    print(f"    Train: {split}, Test: {len(all_data) - split}")
    
    # Model
    print("\n[2] Creating model...")
    d_model, num_tiles = 128, 16
    model = HALO6502Model(d_model=d_model, num_tiles=num_tiles)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # HALO components
    print("\n[3] Initializing HALO...")
    tile_bank = ProgrammableTileBank(num_tiles=num_tiles, d_model=d_model, d_hidden=256)
    guardian = GuardianAngel(
        d_model=d_model,
        num_tiles=num_tiles,
        num_ops=len(OPCODES),
        gentleness=0.1,
        intervention_threshold=0.6,
    )
    
    # Pipeline
    pipeline = HALOPipeline(
        model=model,
        tile_bank=tile_bank,
        guardian=guardian,
        optimizer_fn=lambda params, lr: torch.optim.Adam(params, lr=lr),
        task_loss_fn=nn.BCELoss(),
        device=device,
        total_epochs=total_epochs,
        phase_schedule=phase_schedule,
        verbose=True,
    )
    
    # RUN IT
    print("\n[4] Launching pipeline...")
    results = pipeline.run(
        train_loader=train_loader,
        eval_loader=eval_loader,
        lr_schedule=lr_schedule,
    )
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='HALO Pipeline Field Test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true', help='Quick test with 32 epochs')
    
    args = parser.parse_args()
    
    if args.quick:
        schedule = (8, 8, 8, 8)
        total = 32
    else:
        schedule = (32, 32, 32, 32)
        total = args.epochs
    
    run_halo_pipeline(
        seed=args.seed,
        total_epochs=total,
        phase_schedule=schedule,
        device=args.device,
    )
