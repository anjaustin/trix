#!/usr/bin/env python3
"""
Pure TriX Butterfly: The Complete Unit
=======================================

Single butterfly: (a, b) -> (a+b, a-b)

This is the atomic unit of FFT.
If we nail this with pure TriX, composition is straightforward.

Uses the proven approach:
- Tiles learn ADD and SUB
- Routing selects which tile
- Both outputs in one forward pass

CODENAME: ANN WILSON - BUTTERFLY
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


CONFIG = {
    'value_range': 16,
    'd_model': 64,
    'num_tiles': 4,
    'num_freqs': 6,
    'epochs': 150,
    'batch_size': 64,
    'lr': 0.005,
    'seed': 1122911624,
}


class PureTriXButterfly(nn.Module):
    """
    Complete butterfly using pure TriX.
    
    Input: (a, b)
    Output: (a+b, a-b)
    
    Architecture:
    - Shared encoder for a, b
    - Two routing decisions (one for sum, one for diff)
    - Tiles compute the operations
    """
    
    def __init__(self, value_range=16, d_model=64, num_tiles=4, num_freqs=6):
        super().__init__()
        
        self.value_range = value_range
        self.d_model = d_model
        self.num_freqs = num_freqs
        self.num_tiles = num_tiles
        
        # Input encoding
        fourier_dim = 2 * num_freqs
        self.input_proj = nn.Sequential(
            nn.Linear(fourier_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Two routers: one for sum path, one for diff path
        self.router_sum = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_tiles),
        )
        self.router_diff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_tiles),
        )
        
        # Tiles - each learns to be an operation
        self.tile_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_tiles)
        ])
        
        # Output decoders
        self.sum_head = nn.Linear(d_model, 1)
        self.diff_head = nn.Linear(d_model, 1)
        
        self.temperature = 0.5
        
        # Tracking
        self.register_buffer('sum_tile_counts', torch.zeros(num_tiles))
        self.register_buffer('diff_tile_counts', torch.zeros(num_tiles))
    
    def _fourier_features(self, x):
        x_norm = x.float().unsqueeze(-1) * (2 * np.pi / self.value_range)
        freqs = (2 ** torch.arange(self.num_freqs, device=x.device, dtype=torch.float)).unsqueeze(0)
        angles = x_norm * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, a, b, track=True):
        """
        Args:
            a, b: (batch,) input values
        
        Returns:
            sum_pred: (batch,) predicted a+b
            diff_pred: (batch,) predicted a-b
        """
        batch_size = a.shape[0]
        
        # Encode
        a_feat = self._fourier_features(a)
        b_feat = self._fourier_features(b)
        x = torch.cat([a_feat, b_feat], dim=-1)
        x = self.input_proj(x)
        
        # Route for sum
        sum_logits = self.router_sum(x) / self.temperature
        sum_tile = sum_logits.argmax(dim=-1)
        
        # Route for diff
        diff_logits = self.router_diff(x) / self.temperature
        diff_tile = diff_logits.argmax(dim=-1)
        
        # Compute through selected tiles
        sum_outputs = []
        diff_outputs = []
        
        for i in range(batch_size):
            st = sum_tile[i].item()
            dt = diff_tile[i].item()
            
            sum_out = self.tile_nets[st](x[i:i+1])
            diff_out = self.tile_nets[dt](x[i:i+1])
            
            sum_outputs.append(sum_out)
            diff_outputs.append(diff_out)
        
        sum_hidden = torch.cat(sum_outputs, dim=0)
        diff_hidden = torch.cat(diff_outputs, dim=0)
        
        # Decode
        sum_pred = self.sum_head(sum_hidden).squeeze(-1)
        diff_pred = self.diff_head(diff_hidden).squeeze(-1)
        
        # Track
        if track and self.training:
            with torch.no_grad():
                for i in range(batch_size):
                    self.sum_tile_counts[sum_tile[i]] += 1
                    self.diff_tile_counts[diff_tile[i]] += 1
        
        return sum_pred, diff_pred
    
    def get_routing_analysis(self):
        sum_counts = self.sum_tile_counts.cpu().numpy()
        diff_counts = self.diff_tile_counts.cpu().numpy()
        
        analysis = {}
        for t in range(self.num_tiles):
            total = sum_counts[t] + diff_counts[t]
            if total > 0:
                sum_frac = sum_counts[t] / total
                analysis[t] = {
                    'sum_count': int(sum_counts[t]),
                    'diff_count': int(diff_counts[t]),
                    'sum_fraction': float(sum_frac),
                    'specialization': 'SUM' if sum_frac > 0.6 else ('DIFF' if sum_frac < 0.4 else 'MIXED'),
                }
        return analysis
    
    def reset_tracking(self):
        self.sum_tile_counts.zero_()
        self.diff_tile_counts.zero_()


def generate_data(value_range=16):
    """Generate all (a, b) -> (a+b, a-b) pairs."""
    data = []
    for a in range(value_range):
        for b in range(value_range):
            data.append({
                'a': a,
                'b': b,
                'sum': a + b,
                'diff': a - b,
            })
    return data


def train_epoch(model, data, optimizer, device):
    model.train()
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    total_loss = 0
    batch_size = CONFIG['batch_size']
    
    for i in range(0, len(data), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = [data[j] for j in batch_idx]
        
        a = torch.tensor([d['a'] for d in batch], device=device)
        b = torch.tensor([d['b'] for d in batch], device=device)
        target_sum = torch.tensor([d['sum'] for d in batch], device=device, dtype=torch.float)
        target_diff = torch.tensor([d['diff'] for d in batch], device=device, dtype=torch.float)
        
        pred_sum, pred_diff = model(a, b)
        
        loss = F.mse_loss(pred_sum, target_sum) + F.mse_loss(pred_diff, target_diff)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (len(data) // batch_size + 1)


def evaluate(model, data, device):
    model.eval()
    
    correct_sum = 0
    correct_diff = 0
    correct_both = 0
    
    with torch.no_grad():
        for d in data:
            a = torch.tensor([d['a']], device=device)
            b = torch.tensor([d['b']], device=device)
            
            pred_sum, pred_diff = model(a, b, track=False)
            
            ps = pred_sum.round().item()
            pd = pred_diff.round().item()
            
            sum_ok = (ps == d['sum'])
            diff_ok = (pd == d['diff'])
            
            correct_sum += sum_ok
            correct_diff += diff_ok
            correct_both += (sum_ok and diff_ok)
    
    n = len(data)
    return {
        'sum': correct_sum / n,
        'diff': correct_diff / n,
        'both': correct_both / n,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("\n" + "=" * 70)
    print("PURE TRIX BUTTERFLY")
    print("(a, b) -> (a+b, a-b)")
    print("Tiles learn ops. Routing selects.")
    print("=" * 70)
    
    data = generate_data(CONFIG['value_range'])
    print(f"\nData: {len(data)} pairs")
    
    model = PureTriXButterfly(
        value_range=CONFIG['value_range'],
        d_model=CONFIG['d_model'],
        num_tiles=CONFIG['num_tiles'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    print("\n[TRAINING]")
    
    for epoch in range(CONFIG['epochs']):
        model.reset_tracking()
        loss = train_epoch(model, data, optimizer, device)
        scheduler.step()
        
        if (epoch + 1) % 15 == 0:
            metrics = evaluate(model, data, device)
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, "
                  f"sum={metrics['sum']:.1%}, diff={metrics['diff']:.1%}, both={metrics['both']:.1%}")
            
            if metrics['both'] >= 0.99:
                print("  ✓ 99%+ achieved!")
                break
    
    # Final
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    model.reset_tracking()
    model.train()
    for d in data:
        a = torch.tensor([d['a']], device=device)
        b = torch.tensor([d['b']], device=device)
        model(a, b)
    
    final = evaluate(model, data, device)
    routing = model.get_routing_analysis()
    
    print(f"\nAccuracy:")
    print(f"  Sum (a+b):  {final['sum']:.1%}")
    print(f"  Diff (a-b): {final['diff']:.1%}")
    print(f"  Both:       {final['both']:.1%}")
    
    print(f"\n[TILE ROUTING]")
    for t, info in sorted(routing.items()):
        print(f"  Tile {t}: {info['specialization']:4s} "
              f"(sum={info['sum_count']}, diff={info['diff_count']})")
    
    # Count specialists
    sum_specialists = sum(1 for info in routing.values() if info['specialization'] == 'SUM')
    diff_specialists = sum(1 for info in routing.values() if info['specialization'] == 'DIFF')
    
    print(f"\nSpecialists: {sum_specialists} SUM, {diff_specialists} DIFF")
    
    print("\n" + "=" * 70)
    if final['both'] >= 0.99 and sum_specialists >= 1 and diff_specialists >= 1:
        print("✓ PURE TRIX BUTTERFLY: SUCCESS")
        print(f"  100% accuracy on (a,b) -> (a+b, a-b)")
        print(f"  Tiles specialized: {sum_specialists} SUM, {diff_specialists} DIFF")
        print("  This is the FFT atom. Composition is next.")
    elif final['both'] >= 0.90:
        print(f"◐ PARTIAL: {final['both']:.1%}")
    else:
        print(f"✗ NEEDS WORK: {final['both']:.1%}")
    print("=" * 70)
    
    # Save
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'config': CONFIG,
        'accuracy': final,
        'routing': routing,
        'sum_specialists': sum_specialists,
        'diff_specialists': diff_specialists,
    }
    
    with open(output_dir / f'pure_trix_butterfly_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return final['both']


if __name__ == "__main__":
    main()
