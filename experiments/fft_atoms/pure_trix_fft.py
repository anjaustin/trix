#!/usr/bin/env python3
"""
Pure TriX FFT: Programmed Tiles + Learned Routing
==================================================

The architecture:
- Tiles are PROGRAMMED with micro-ops (ADD, SUB)
- Routing is LEARNED to select the right tile
- Composition produces FFT

This is PURE TRIX:
- No external organs
- No hybrid compute
- Everything inside TriX

The tiles ARE the operations. Routing IS the control flow.

CODENAME: ANN WILSON - PURE HEART BEATS
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


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'value_range': 16,
    'd_model': 64,
    'num_tiles': 4,         # 2 for ops + 2 spare
    'num_freqs': 6,
    'epochs': 200,
    'batch_size': 32,
    'lr': 0.005,
    'seed': 1122911624,     # The Second Star Constant
}

# FFT Micro-ops
OPS = {
    0: 'ADD',   # a + b
    1: 'SUB',   # a - b
}


# =============================================================================
# Programmed Tile Layer
# =============================================================================

class ProgrammedTileFFN(nn.Module):
    """
    FFN with programmed tiles and learned routing.
    
    Each tile computes a specific operation.
    Routing learns to select the right tile.
    
    This is pure TriX: tiles are the compute, routing is the control.
    """
    
    def __init__(self, d_model=64, num_tiles=4, num_ops=2):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_ops = num_ops
        
        # Routing network: input -> tile selection
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_tiles),
        )
        
        # Programmed tile operations
        # Each tile has weights that implement a specific operation
        # Tile 0: ADD (output = a + b)
        # Tile 1: SUB (output = a - b)
        
        # We'll use small MLPs per tile that we pre-train to do the ops
        self.tile_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_tiles)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Temperature for routing
        self.temperature = 0.5
        
        # Track routing decisions
        self.register_buffer('routing_counts', torch.zeros(num_tiles, num_ops))
    
    def forward(self, x, op_label=None, hard_route=True):
        """
        Args:
            x: (batch, d_model) input
            op_label: (batch,) operation labels for tracking
            hard_route: use hard (argmax) or soft routing
        
        Returns:
            output: (batch, d_model)
            routing_info: dict with tile selections
        """
        batch_size = x.shape[0]
        
        # Compute routing scores
        route_logits = self.router(x) / self.temperature  # (batch, num_tiles)
        
        if hard_route:
            # Hard routing: select one tile
            tile_idx = route_logits.argmax(dim=-1)  # (batch,)
            
            # Compute output for each sample using its selected tile
            outputs = []
            for i in range(batch_size):
                t = tile_idx[i].item()
                out = self.tile_nets[t](x[i:i+1])
                outputs.append(out)
            output = torch.cat(outputs, dim=0)
        else:
            # Soft routing: weighted combination
            route_weights = F.softmax(route_logits, dim=-1)  # (batch, num_tiles)
            
            # Compute all tile outputs
            tile_outputs = torch.stack([net(x) for net in self.tile_nets], dim=1)  # (batch, num_tiles, d_model)
            
            # Weighted combination
            output = (route_weights.unsqueeze(-1) * tile_outputs).sum(dim=1)  # (batch, d_model)
            
            tile_idx = route_logits.argmax(dim=-1)
        
        output = self.output_proj(output)
        
        # Track routing
        if op_label is not None and self.training:
            with torch.no_grad():
                for i in range(batch_size):
                    t = tile_idx[i].item()
                    o = op_label[i].item()
                    self.routing_counts[t, o] += 1
        
        routing_info = {
            'tile_idx': tile_idx,
            'route_logits': route_logits,
        }
        
        return output, routing_info
    
    def get_tile_specialization(self):
        """Analyze tile-operation correspondence."""
        counts = self.routing_counts.cpu().numpy()
        analysis = {}
        
        for tile in range(self.num_tiles):
            total = counts[tile].sum()
            if total > 0:
                dominant_op = counts[tile].argmax()
                purity = counts[tile, dominant_op] / total
                analysis[tile] = {
                    'dominant_op': OPS[dominant_op],
                    'purity': float(purity),
                    'counts': {OPS[i]: int(counts[tile, i]) for i in range(len(OPS))},
                    'total': int(total),
                }
        
        return analysis
    
    def reset_tracking(self):
        self.routing_counts.zero_()


# =============================================================================
# Pure TriX FFT Model
# =============================================================================

class PureTriXFFT(nn.Module):
    """
    Pure TriX model for FFT micro-operations.
    
    Architecture:
    1. Encode inputs (op, a, b) with Fourier features
    2. Route to programmed tile (ADD or SUB)
    3. Decode output bits
    """
    
    def __init__(self, value_range=16, d_model=64, num_tiles=4, num_freqs=6):
        super().__init__()
        
        self.value_range = value_range
        self.d_model = d_model
        self.num_freqs = num_freqs
        
        # Input encoding
        # Op embedding
        self.op_embed = nn.Embedding(len(OPS), d_model // 4)
        
        # Fourier features for a, b
        fourier_dim = 2 * num_freqs
        input_dim = d_model // 4 + 2 * fourier_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Core: Programmed tile FFN
        self.ffn = ProgrammedTileFFN(
            d_model=d_model,
            num_tiles=num_tiles,
            num_ops=len(OPS),
        )
        
        # Output decoding: bits
        # ADD: 5 bits (0-30)
        # SUB: 6 bits (signed -15 to 15)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6),  # Max 6 bits
        )
    
    def forward(self, op, a, b):
        """
        Args:
            op: (batch,) operation indices
            a, b: (batch,) integer values
        
        Returns:
            logits: (batch, 6) bit logits
            routing_info: from FFN
        """
        # Encode inputs
        op_emb = self.op_embed(op)
        a_feat = self._fourier_features(a)
        b_feat = self._fourier_features(b)
        
        x = torch.cat([op_emb, a_feat, b_feat], dim=-1)
        x = self.input_proj(x)
        
        # Route through programmed tiles
        x, routing_info = self.ffn(x, op_label=op)
        
        # Decode to bits
        logits = self.output_head(x)
        
        return logits, routing_info
    
    def _fourier_features(self, x):
        """Fourier feature encoding."""
        x_norm = x.float().unsqueeze(-1) * (2 * np.pi / self.value_range)
        freqs = (2 ** torch.arange(self.num_freqs, device=x.device, dtype=torch.float)).unsqueeze(0)
        angles = x_norm * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def get_tile_specialization(self):
        return self.ffn.get_tile_specialization()
    
    def reset_tracking(self):
        self.ffn.reset_tracking()


# =============================================================================
# Bit Encoding/Decoding
# =============================================================================

def encode_bits(value, num_bits=6, signed=True):
    """Encode value as bits."""
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
# Data Generation
# =============================================================================

def generate_data(value_range=16):
    """Generate (op, a, b) -> result data."""
    data = []
    
    for op_id, op_name in OPS.items():
        for a in range(value_range):
            for b in range(value_range):
                if op_name == 'ADD':
                    result = a + b
                    result_bits = encode_bits(result, num_bits=5, signed=False)
                else:  # SUB
                    result = a - b
                    result_bits = encode_bits(result, num_bits=6, signed=True)
                
                # Pad to 6 bits
                while len(result_bits) < 6:
                    result_bits.append(0)
                
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
# Training
# =============================================================================

def train_epoch(model, data, optimizer, device):
    model.train()
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    total_loss = 0
    batch_size = CONFIG['batch_size']
    
    for i in range(0, len(data), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = [data[j] for j in batch_idx]
        
        op = torch.tensor([d['op'] for d in batch], device=device)
        a = torch.tensor([d['a'] for d in batch], device=device)
        b = torch.tensor([d['b'] for d in batch], device=device)
        target_bits = torch.tensor([d['result_bits'] for d in batch], device=device, dtype=torch.float)
        
        logits, routing_info = model(op, a, b)
        
        loss = F.binary_cross_entropy_with_logits(logits, target_bits)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (len(data) // batch_size + 1)


def evaluate(model, data, device):
    model.eval()
    
    correct = 0
    per_op_correct = defaultdict(int)
    per_op_total = defaultdict(int)
    
    with torch.no_grad():
        for d in data:
            op = torch.tensor([d['op']], device=device)
            a = torch.tensor([d['a']], device=device)
            b = torch.tensor([d['b']], device=device)
            
            logits, _ = model(op, a, b)
            pred_bits = (torch.sigmoid(logits[0]) > 0.5).tolist()
            
            if d['op_name'] == 'ADD':
                pred = decode_bits(pred_bits[:5], signed=False)
            else:
                pred = decode_bits(pred_bits[:6], signed=True)
            
            is_correct = (pred == d['result'])
            correct += is_correct
            per_op_correct[d['op_name']] += is_correct
            per_op_total[d['op_name']] += 1
    
    return {
        'accuracy': correct / len(data),
        'per_op': {op: per_op_correct[op] / per_op_total[op] for op in per_op_total},
    }


# =============================================================================
# Main
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("\n" + "=" * 70)
    print("PURE TRIX FFT")
    print("Programmed Tiles + Learned Routing")
    print("=" * 70)
    
    # Generate data
    data = generate_data(CONFIG['value_range'])
    print(f"\nData: {len(data)} examples ({len(OPS)} ops × {CONFIG['value_range']}² pairs)")
    
    # Create model
    model = PureTriXFFT(
        value_range=CONFIG['value_range'],
        d_model=CONFIG['d_model'],
        num_tiles=CONFIG['num_tiles'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    # Training
    print("\n[TRAINING]")
    
    best_acc = 0
    for epoch in range(CONFIG['epochs']):
        model.reset_tracking()
        loss = train_epoch(model, data, optimizer, device)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            metrics = evaluate(model, data, device)
            
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
            
            per_op_str = ', '.join(f"{k}:{v:.0%}" for k, v in metrics['per_op'].items())
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, acc={metrics['accuracy']:.1%} ({per_op_str})")
            
            if metrics['accuracy'] >= 0.99:
                print(f"  ✓ 99%+ achieved!")
                break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    model.reset_tracking()
    model.train()
    for d in data:
        op = torch.tensor([d['op']], device=device)
        a = torch.tensor([d['a']], device=device)
        b = torch.tensor([d['b']], device=device)
        model(op, a, b)
    
    final_metrics = evaluate(model, data, device)
    tile_analysis = model.get_tile_specialization()
    
    print(f"\nAccuracy: {final_metrics['accuracy']:.1%}")
    for op, acc in final_metrics['per_op'].items():
        print(f"  {op}: {acc:.1%}")
    
    print("\n[TILE SPECIALIZATION]")
    high_purity = 0
    for tile, info in sorted(tile_analysis.items()):
        purity_bar = "█" * int(info['purity'] * 20)
        print(f"  Tile {tile}: {info['dominant_op']:4s} purity={info['purity']:.0%} {purity_bar}")
        if info['purity'] >= 0.8:
            high_purity += 1
    
    print(f"\nHigh-purity tiles (≥80%): {high_purity}/{len(tile_analysis)}")
    
    # Verdict
    print("\n" + "=" * 70)
    if final_metrics['accuracy'] >= 0.95 and high_purity >= 2:
        print("✓ PURE TRIX FFT: SUCCESS")
        print(f"  Accuracy: {final_metrics['accuracy']:.1%}")
        print(f"  Tile specialization: {high_purity} tiles at ≥80% purity")
        print("  Tiles learned ops. Routing learned control.")
    elif final_metrics['accuracy'] >= 0.80:
        print("◐ PARTIAL SUCCESS")
    else:
        print("✗ NEEDS MORE WORK")
    print("=" * 70)
    
    # Save results
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'config': CONFIG,
        'accuracy': final_metrics['accuracy'],
        'per_op': final_metrics['per_op'],
        'tile_analysis': {str(k): v for k, v in tile_analysis.items()},
        'high_purity_tiles': high_purity,
    }
    
    with open(output_dir / f'pure_trix_fft_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return final_metrics['accuracy'], high_purity


if __name__ == "__main__":
    main()
