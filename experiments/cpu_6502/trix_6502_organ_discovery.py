#!/usr/bin/env python3
"""
TriX 6502 Organ Discovery Experiment

The Question: Can TriX tiles learn to BE CPU organs without being told which is which?

Experiment Design:
1. Generate ALL 6502 operations mixed together
2. Train TriX to predict (result, flags) from (opcode, A, operand, C_in)
3. Measure tile specialization - do tiles become organs?

This tests the core thesis: semantic geometry naturally carves at operation boundaries.
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/flynnconceivable')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import json
from datetime import datetime

# TriX imports
from trix.nn import SparseLookupFFN, SparseLookupFFNv2

# FLYNNCONCEIVABLE data generators
from training.data import (
    adc_truth, sbc_truth,
    asl_truth, lsr_truth, rol_truth, ror_truth,
    and_truth, ora_truth, eor_truth,
    inc_truth, dec_truth,
    cmp_truth,
)


# =============================================================================
# Data Generation
# =============================================================================

# Operation categories (for measuring specialization, NOT for training)
OP_CATEGORIES = {
    'ALU': ['ADC', 'SBC'],
    'LOGIC': ['AND', 'ORA', 'EOR'],
    'SHIFT': ['ASL', 'LSR', 'ROL', 'ROR'],
    'INCDEC': ['INC', 'DEC'],
    'COMPARE': ['CMP'],
}

# Reverse mapping
OP_TO_CATEGORY = {}
for cat, ops in OP_CATEGORIES.items():
    for op in ops:
        OP_TO_CATEGORY[op] = cat

# Opcode encoding (one-hot style, but as integer for embedding)
OPCODES = ['ADC', 'SBC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC', 'CMP']
OPCODE_TO_IDX = {op: i for i, op in enumerate(OPCODES)}


def generate_all_operations() -> List[Dict]:
    """Generate exhaustive dataset of all 6502 operations."""
    data = []
    
    # ALU: ADC, SBC (256 * 256 * 2 = 131,072 each, but sample for tractability)
    print("Generating ALU operations...")
    for a in range(256):
        for op in range(0, 256, 4):  # Sample every 4th for speed
            for c in [0, 1]:
                d = adc_truth(a, op, c)
                data.append({
                    'opcode': 'ADC',
                    'a': a, 'operand': op, 'c_in': c,
                    'result': d['result'],
                    'n': d['n'], 'z': d['z'], 'c': d['c'], 'v': d['v']
                })
                d = sbc_truth(a, op, c)
                data.append({
                    'opcode': 'SBC',
                    'a': a, 'operand': op, 'c_in': c,
                    'result': d['result'],
                    'n': d['n'], 'z': d['z'], 'c': d['c'], 'v': d['v']
                })
    
    # LOGIC: AND, ORA, EOR (256 * 256 = 65,536 each)
    print("Generating LOGIC operations...")
    for a in range(256):
        for op in range(0, 256, 4):  # Sample
            for func, name in [(and_truth, 'AND'), (ora_truth, 'ORA'), (eor_truth, 'EOR')]:
                d = func(a, op)
                data.append({
                    'opcode': name,
                    'a': a, 'operand': op, 'c_in': 0,
                    'result': d['result'],
                    'n': d['n'], 'z': d['z'], 'c': 0, 'v': 0
                })
    
    # SHIFT: ASL, LSR, ROL, ROR
    print("Generating SHIFT operations...")
    for val in range(256):
        # ASL, LSR (no carry input)
        d = asl_truth(val)
        data.append({
            'opcode': 'ASL',
            'a': val, 'operand': 0, 'c_in': 0,
            'result': d['result'],
            'n': d['n'], 'z': d['z'], 'c': d['c'], 'v': 0
        })
        d = lsr_truth(val)
        data.append({
            'opcode': 'LSR',
            'a': val, 'operand': 0, 'c_in': 0,
            'result': d['result'],
            'n': d['n'], 'z': d['z'], 'c': d['c'], 'v': 0
        })
        # ROL, ROR (with carry input)
        for c in [0, 1]:
            d = rol_truth(val, c)
            data.append({
                'opcode': 'ROL',
                'a': val, 'operand': 0, 'c_in': c,
                'result': d['result'],
                'n': d['n'], 'z': d['z'], 'c': d['c'], 'v': 0
            })
            d = ror_truth(val, c)
            data.append({
                'opcode': 'ROR',
                'a': val, 'operand': 0, 'c_in': c,
                'result': d['result'],
                'n': d['n'], 'z': d['z'], 'c': d['c'], 'v': 0
            })
    
    # INCDEC: INC, DEC
    print("Generating INCDEC operations...")
    for val in range(256):
        d = inc_truth(val)
        data.append({
            'opcode': 'INC',
            'a': val, 'operand': 0, 'c_in': 0,
            'result': d['result'],
            'n': d['n'], 'z': d['z'], 'c': 0, 'v': 0
        })
        d = dec_truth(val)
        data.append({
            'opcode': 'DEC',
            'a': val, 'operand': 0, 'c_in': 0,
            'result': d['result'],
            'n': d['n'], 'z': d['z'], 'c': 0, 'v': 0
        })
    
    # COMPARE: CMP
    print("Generating COMPARE operations...")
    for a in range(256):
        for op in range(0, 256, 4):  # Sample
            d = cmp_truth(a, op)
            data.append({
                'opcode': 'CMP',
                'a': a, 'operand': op, 'c_in': 0,
                'result': (a - op) & 0xFF,  # CMP computes but doesn't store
                'n': d['n'], 'z': d['z'], 'c': d['c'], 'v': 0
            })
    
    print(f"Total samples: {len(data):,}")
    return data


# =============================================================================
# Model: TriX-based 6502 Processor
# =============================================================================

class TriX6502(nn.Module):
    """
    TriX-based 6502 operation processor.
    
    Input: (opcode, A, operand, C_in) encoded as embedding + binary
    Output: (result, N, Z, C, V) flags
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_tiles: int = 16,
        tiles_per_cluster: int = 4,
        use_v2: bool = False,
        **v2_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_v2 = use_v2
        
        # Input encoding
        self.opcode_embed = nn.Embedding(len(OPCODES), d_model // 4)
        self.input_proj = nn.Linear(d_model // 4 + 17, d_model)  # 17 = 8 + 8 + 1 (A bits, op bits, c_in)
        
        # TriX FFN (the core)
        if use_v2:
            self.ffn = SparseLookupFFNv2(
                d_model=d_model,
                num_tiles=num_tiles,
                tiles_per_cluster=tiles_per_cluster,
                **v2_kwargs
            )
        else:
            self.ffn = SparseLookupFFN(
                d_model=d_model,
                num_tiles=num_tiles,
                tiles_per_cluster=tiles_per_cluster,
            )
        
        # Output heads
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )
        self.flag_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )
    
    def encode_input(self, opcode_idx, a, operand, c_in):
        """Encode inputs as d_model vector."""
        batch_size = opcode_idx.shape[0]
        
        # Opcode embedding
        op_emb = self.opcode_embed(opcode_idx)  # (B, d_model//4)
        
        # Binary encode A and operand
        a_bits = torch.zeros(batch_size, 8, device=opcode_idx.device)
        op_bits = torch.zeros(batch_size, 8, device=opcode_idx.device)
        for i in range(8):
            a_bits[:, i] = ((a >> i) & 1).float()
            op_bits[:, i] = ((operand >> i) & 1).float()
        
        # Concatenate
        x = torch.cat([op_emb, a_bits, op_bits, c_in.float().unsqueeze(1)], dim=1)
        x = self.input_proj(x)
        
        return x
    
    def forward(self, opcode_idx, a, operand, c_in, labels=None):
        """
        Forward pass.
        
        Returns:
            result_bits: (B, 8) predicted result bits
            flags: (B, 4) predicted flags (N, Z, C, V)
            routing_info: dict with tile assignments
            aux_losses: dict with auxiliary losses
        """
        # Encode input
        x = self.encode_input(opcode_idx, a, operand, c_in)
        x = x.unsqueeze(1)  # (B, 1, d_model) for FFN
        
        # TriX routing + computation
        if self.use_v2:
            out, routing_info, aux_losses = self.ffn(x, labels=labels)
        else:
            out, routing_info, aux_losses = self.ffn(x)
        
        out = out.squeeze(1)  # (B, d_model)
        
        # Decode outputs
        result_bits = self.result_head(out)
        flags = self.flag_head(out)
        
        return result_bits, flags, routing_info, aux_losses
    
    def predict(self, opcode_idx, a, operand, c_in):
        """Predict with decoding."""
        self.eval()
        with torch.no_grad():
            result_bits, flags, routing_info, _ = self(opcode_idx, a, operand, c_in)
            
            # Decode result
            result = torch.zeros(result_bits.shape[0], dtype=torch.long, device=result_bits.device)
            for i in range(8):
                result += ((result_bits[:, i] > 0.5).long() << i)
            
            # Decode flags
            n = (flags[:, 0] > 0.5).long()
            z = (flags[:, 1] > 0.5).long()
            c = (flags[:, 2] > 0.5).long()
            v = (flags[:, 3] > 0.5).long()
            
            return result, n, z, c, v, routing_info


# =============================================================================
# Training
# =============================================================================

def train_trix_6502(
    model: TriX6502,
    data: List[Dict],
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 0.001,
    device: str = 'cuda',
    track_specialization: bool = True,
) -> Dict:
    """Train TriX on 6502 operations."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Prepare tensors
    opcode_idx = torch.tensor([OPCODE_TO_IDX[d['opcode']] for d in data], device=device)
    a_t = torch.tensor([d['a'] for d in data], device=device)
    op_t = torch.tensor([d['operand'] for d in data], device=device)
    c_t = torch.tensor([d['c_in'] for d in data], device=device)
    result_t = torch.tensor([d['result'] for d in data], device=device)
    n_t = torch.tensor([d['n'] for d in data], dtype=torch.float32, device=device)
    z_t = torch.tensor([d['z'] for d in data], dtype=torch.float32, device=device)
    c_out_t = torch.tensor([d['c'] for d in data], dtype=torch.float32, device=device)
    v_t = torch.tensor([d['v'] for d in data], dtype=torch.float32, device=device)
    
    # Target bits
    result_bits_t = torch.zeros(len(data), 8, device=device)
    for i in range(8):
        result_bits_t[:, i] = ((result_t >> i) & 1).float()
    
    flags_t = torch.stack([n_t, z_t, c_out_t, v_t], dim=1)
    
    # For specialization tracking
    opcodes = [d['opcode'] for d in data]
    
    history = {
        'loss': [],
        'accuracy': [],
        'tile_purity': [],
    }
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(data), device=device)
        total_loss = 0
        correct = 0
        
        # Track tile assignments per opcode
        tile_assignments = defaultdict(lambda: defaultdict(int))
        
        for i in range(0, len(data) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            
            # Forward
            labels = opcode_idx[idx] if model.use_v2 else None
            result_bits, flags, routing_info, aux_losses = model(
                opcode_idx[idx], a_t[idx], op_t[idx], c_t[idx], labels=labels
            )
            
            # Loss
            result_loss = F.binary_cross_entropy(result_bits, result_bits_t[idx])
            flag_loss = F.binary_cross_entropy(flags, flags_t[idx])
            # Extra weight on Z flag (it's tricky)
            z_loss = F.binary_cross_entropy(flags[:, 1], z_t[idx]) * 2.0
            
            loss = result_loss + flag_loss + z_loss + aux_losses['total_aux']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy (result only)
            pred_result = torch.zeros(len(idx), dtype=torch.long, device=device)
            for b in range(8):
                pred_result += ((result_bits[:, b] > 0.5).long() << b)
            correct += (pred_result == result_t[idx]).sum().item()
            
            # Track tile assignments for specialization analysis
            if track_specialization:
                tiles = routing_info['tile_idx'].squeeze(-1).cpu().numpy()
                for j, tile in enumerate(tiles):
                    op = opcodes[idx[j].item()]
                    tile_assignments[int(tile)][op] += 1
        
        acc = correct / len(data) * 100
        avg_loss = total_loss / (len(data) // batch_size)
        
        # Compute tile purity
        purity = compute_tile_purity(tile_assignments)
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        history['tile_purity'].append(purity)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: loss={avg_loss:.4f}, acc={acc:.2f}%, purity={purity:.2f}")
    
    return history, tile_assignments


def compute_tile_purity(tile_assignments: Dict) -> float:
    """
    Compute average tile purity.
    
    Purity = for each tile, what fraction of assignments are the dominant operation?
    """
    purities = []
    for tile, op_counts in tile_assignments.items():
        if not op_counts:
            continue
        total = sum(op_counts.values())
        dominant = max(op_counts.values())
        purities.append(dominant / total)
    
    return np.mean(purities) if purities else 0.0


def analyze_specialization(tile_assignments: Dict) -> Dict:
    """
    Analyze which tiles specialize to which operation categories.
    
    Returns mapping of tile -> dominant category with purity.
    """
    analysis = {}
    
    for tile, op_counts in tile_assignments.items():
        if not op_counts:
            continue
        
        # Aggregate by category
        cat_counts = defaultdict(int)
        for op, count in op_counts.items():
            cat = OP_TO_CATEGORY.get(op, 'OTHER')
            cat_counts[cat] += count
        
        total = sum(cat_counts.values())
        dominant_cat = max(cat_counts, key=cat_counts.get)
        purity = cat_counts[dominant_cat] / total
        
        # Top operations
        sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:3]
        
        analysis[tile] = {
            'dominant_category': dominant_cat,
            'category_purity': purity,
            'top_ops': sorted_ops,
            'total_count': total,
        }
    
    return analysis


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(use_v2: bool = False, num_tiles: int = 16):
    """Run the organ discovery experiment."""
    
    print("=" * 70)
    print(f"TRIX 6502 ORGAN DISCOVERY EXPERIMENT")
    print(f"Version: {'v2 (surgery + regularizers)' if use_v2 else 'v1 (baseline)'}")
    print(f"Tiles: {num_tiles}")
    print("=" * 70)
    
    # Generate data
    print("\n[1] Generating 6502 operations...")
    data = generate_all_operations()
    
    # Count by category
    cat_counts = defaultdict(int)
    for d in data:
        cat = OP_TO_CATEGORY.get(d['opcode'], 'OTHER')
        cat_counts[cat] += 1
    print("\nSamples by category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count:,}")
    
    # Create model
    print("\n[2] Creating TriX model...")
    v2_kwargs = {
        'ternary_weight': 0.01,
        'sparsity_weight': 0.01,
        'diversity_weight': 0.01,
    } if use_v2 else {}
    
    model = TriX6502(
        d_model=64,
        num_tiles=num_tiles,
        tiles_per_cluster=4,
        use_v2=use_v2,
        **v2_kwargs
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Train
    print("\n[3] Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    history, tile_assignments = train_trix_6502(
        model, data,
        epochs=30,
        batch_size=512,
        lr=0.001,
        device=device,
    )
    
    # Analyze specialization
    print("\n[4] Analyzing tile specialization...")
    analysis = analyze_specialization(tile_assignments)
    
    print("\nTile Specialization:")
    print("-" * 70)
    for tile in sorted(analysis.keys()):
        info = analysis[tile]
        top_ops = ', '.join([f"{op}({c})" for op, c in info['top_ops']])
        print(f"  Tile {tile:2d}: {info['dominant_category']:8s} (purity={info['category_purity']:.2f}) | {top_ops}")
    
    # Summary statistics
    print("\n[5] Summary Statistics")
    print("-" * 70)
    
    # How many tiles per category?
    cat_tiles = defaultdict(list)
    for tile, info in analysis.items():
        if info['category_purity'] > 0.5:  # Majority specialized
            cat_tiles[info['dominant_category']].append(tile)
    
    print("\nTiles per category (purity > 50%):")
    for cat in OP_CATEGORIES:
        tiles = cat_tiles.get(cat, [])
        print(f"  {cat}: {len(tiles)} tiles {tiles}")
    
    # Overall purity
    avg_purity = np.mean([info['category_purity'] for info in analysis.values()])
    print(f"\nAverage category purity: {avg_purity:.2f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.2f}%")
    
    # Verdict
    print("\n" + "=" * 70)
    if avg_purity > 0.6:
        print("RESULT: TILES SPECIALIZED TO OPERATION CATEGORIES")
        print("The geometry thesis holds: semantic boundaries emerge from training.")
    elif avg_purity > 0.4:
        print("RESULT: PARTIAL SPECIALIZATION")
        print("Some structure emerged, but tiles share multiple operations.")
    else:
        print("RESULT: NO CLEAR SPECIALIZATION")
        print("Tiles did not carve at semantic boundaries.")
    print("=" * 70)
    
    return model, history, analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="Use SparseLookupFFNv2")
    parser.add_argument("--tiles", type=int, default=16, help="Number of tiles")
    args = parser.parse_args()
    
    model, history, analysis = run_experiment(use_v2=args.v2, num_tiles=args.tiles)
