#!/usr/bin/env python3
"""
Pure TriX FFT: Linear-Residual Architecture
============================================

Key insight: butterfly is LINEAR (a+b, a-b).
Neural nets struggle with arithmetic extrapolation.

Solution: Skip connections that make linear ops easy.
Tiles learn WHEN to apply which linear combination.
Routing IS the computation - it selects coefficients.

CODENAME: ANN WILSON - THESE DREAMS
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pathlib import Path


CONFIG = {
    'N': 8,
    'value_range': 16,
    'd_model': 64,
    'num_tiles': 4,
    'num_freqs': 8,
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.005,
    'seed': 1122911624,
}


class LinearResidualButterfly(nn.Module):
    """
    Butterfly with linear residual structure.
    
    Key: output = coeff_a * a + coeff_b * b
    For sum: coeff_a=1, coeff_b=1
    For diff: coeff_a=1, coeff_b=-1
    
    Tiles learn to produce the right coefficients.
    Routing selects which tile based on context.
    """
    
    def __init__(self, d_model=64, num_tiles=4, num_freqs=8):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_freqs = num_freqs
        
        # Context encoding (just need to distinguish sum vs diff)
        # Using Fourier features of normalized values
        fourier_dim = 2 * num_freqs
        self.context_proj = nn.Sequential(
            nn.Linear(fourier_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Router: context -> tile selection
        self.router = nn.Linear(d_model, num_tiles)
        
        # Each tile outputs 2 coefficients (for a and b)
        # Initialized near identity for sum, negative for diff
        self.tile_coeffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),  # (coeff_a, coeff_b)
            )
            for _ in range(num_tiles)
        ])
        
        # Initialize tiles to different starting points
        with torch.no_grad():
            for i, tile in enumerate(self.tile_coeffs):
                # Bias last layer toward different operations
                if i % 2 == 0:
                    tile[-1].bias.data = torch.tensor([1.0, 1.0])  # sum-like
                else:
                    tile[-1].bias.data = torch.tensor([1.0, -1.0])  # diff-like
        
        self.temperature = 1.0
    
    def _fourier_features(self, x, scale=32.0):
        """Fourier encode (scale-invariant context)."""
        # Normalize to [-1, 1] range for context
        x_norm = x.unsqueeze(-1) / scale
        freqs = torch.arange(1, self.num_freqs + 1, device=x.device, dtype=torch.float)
        angles = x_norm * freqs * np.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, a, b, op_type=None):
        """
        Args:
            a, b: (batch,) input values (any range)
            op_type: (batch,) 0=sum, 1=diff (optional, for training signal)
        
        Returns:
            sum_pred, diff_pred: (batch,) outputs
        """
        batch_size = a.shape[0]
        
        # Build context from normalized values (scale-invariant)
        a_ctx = self._fourier_features(a)
        b_ctx = self._fourier_features(b)
        ctx = torch.cat([a_ctx, b_ctx], dim=-1)
        ctx = self.context_proj(ctx)
        
        # Soft routing
        route_logits = self.router(ctx) / self.temperature
        route_weights = F.softmax(route_logits, dim=-1)  # (batch, num_tiles)
        
        # Get coefficients from all tiles
        all_coeffs = torch.stack([tile(ctx) for tile in self.tile_coeffs], dim=1)  # (batch, num_tiles, 2)
        
        # Weighted combination of coefficients
        coeffs = (route_weights.unsqueeze(-1) * all_coeffs).sum(dim=1)  # (batch, 2)
        
        coeff_a = coeffs[:, 0]
        coeff_b = coeffs[:, 1]
        
        # Linear combination - THIS IS THE KEY
        # The arithmetic is explicit, tiles just learn the coefficients
        output = coeff_a * a + coeff_b * b
        
        return output, coeffs
    
    def forward_both(self, a, b):
        """Compute both sum and diff outputs."""
        batch_size = a.shape[0]
        
        # Context
        a_ctx = self._fourier_features(a)
        b_ctx = self._fourier_features(b)
        ctx = torch.cat([a_ctx, b_ctx], dim=-1)
        ctx = self.context_proj(ctx)
        
        # For sum: we want coeff_a=1, coeff_b=1
        # For diff: we want coeff_a=1, coeff_b=-1
        # Train two separate routing paths
        
        route_logits = self.router(ctx) / self.temperature
        route_weights = F.softmax(route_logits, dim=-1)
        
        all_coeffs = torch.stack([tile(ctx) for tile in self.tile_coeffs], dim=1)
        coeffs = (route_weights.unsqueeze(-1) * all_coeffs).sum(dim=1)
        
        # We need to output BOTH sum and diff
        # Approach: coeffs learned should be for the "current" operation
        # We'll train with op_type to distinguish
        
        coeff_a = coeffs[:, 0]
        coeff_b = coeffs[:, 1]
        output = coeff_a * a + coeff_b * b
        
        return output, coeffs


class DualPathButterfly(nn.Module):
    """
    Two routing paths: one for sum, one for diff.
    Each path learns its own coefficient selection.
    """
    
    def __init__(self, d_model=64, num_tiles=4, num_freqs=8):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_freqs = num_freqs
        
        fourier_dim = 2 * num_freqs
        self.context_proj = nn.Sequential(
            nn.Linear(fourier_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Separate routers for sum and diff
        self.router_sum = nn.Linear(d_model, num_tiles)
        self.router_diff = nn.Linear(d_model, num_tiles)
        
        # Shared tiles that output coefficients
        self.tiles = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),
            )
            for _ in range(num_tiles)
        ])
        
        # Initialize: half for sum, half for diff
        with torch.no_grad():
            for i, tile in enumerate(self.tiles):
                if i < num_tiles // 2:
                    tile[-1].bias.data = torch.tensor([1.0, 1.0])
                else:
                    tile[-1].bias.data = torch.tensor([1.0, -1.0])
        
        self.temperature = 0.5
    
    def _fourier_features(self, x, scale=32.0):
        x_norm = x.unsqueeze(-1) / scale
        freqs = torch.arange(1, self.num_freqs + 1, device=x.device, dtype=torch.float)
        angles = x_norm * freqs * np.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, a, b):
        """Returns (sum_pred, diff_pred)."""
        # Context
        a_ctx = self._fourier_features(a)
        b_ctx = self._fourier_features(b)
        ctx = torch.cat([a_ctx, b_ctx], dim=-1)
        ctx = self.context_proj(ctx)
        
        # Get all tile outputs
        all_coeffs = torch.stack([tile(ctx) for tile in self.tiles], dim=1)  # (batch, num_tiles, 2)
        
        # Route for sum
        sum_weights = F.softmax(self.router_sum(ctx) / self.temperature, dim=-1)
        sum_coeffs = (sum_weights.unsqueeze(-1) * all_coeffs).sum(dim=1)
        sum_pred = sum_coeffs[:, 0] * a + sum_coeffs[:, 1] * b
        
        # Route for diff
        diff_weights = F.softmax(self.router_diff(ctx) / self.temperature, dim=-1)
        diff_coeffs = (diff_weights.unsqueeze(-1) * all_coeffs).sum(dim=1)
        diff_pred = diff_coeffs[:, 0] * a + diff_coeffs[:, 1] * b
        
        return sum_pred, diff_pred, sum_coeffs, diff_coeffs


def train_and_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("\n" + "=" * 70)
    print("PURE TRIX FFT - LINEAR RESIDUAL ARCHITECTURE")
    print("Tiles learn coefficients, arithmetic is explicit")
    print("=" * 70)
    
    model = DualPathButterfly(
        d_model=CONFIG['d_model'],
        num_tiles=CONFIG['num_tiles'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    # Training data: small range (will generalize via linear structure)
    train_data = []
    vr = CONFIG['value_range']
    for a in range(vr):
        for b in range(vr):
            train_data.append({'a': a, 'b': b, 'sum': a + b, 'diff': a - b})
    
    print(f"Training on {len(train_data)} pairs (range 0-{vr-1})")
    
    batch_size = CONFIG['batch_size']
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        np.random.shuffle(train_data)
        
        total_loss = 0
        total_coeff_loss = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            a = torch.tensor([d['a'] for d in batch], device=device, dtype=torch.float)
            b = torch.tensor([d['b'] for d in batch], device=device, dtype=torch.float)
            target_sum = torch.tensor([d['sum'] for d in batch], device=device, dtype=torch.float)
            target_diff = torch.tensor([d['diff'] for d in batch], device=device, dtype=torch.float)
            
            sum_pred, diff_pred, sum_coeffs, diff_coeffs = model(a, b)
            
            # Output loss
            out_loss = F.mse_loss(sum_pred, target_sum) + F.mse_loss(diff_pred, target_diff)
            
            # Coefficient regularization: encourage (1,1) for sum, (1,-1) for diff
            target_sum_coeffs = torch.tensor([[1.0, 1.0]], device=device).expand(len(batch), -1)
            target_diff_coeffs = torch.tensor([[1.0, -1.0]], device=device).expand(len(batch), -1)
            
            coeff_loss = F.mse_loss(sum_coeffs, target_sum_coeffs) + F.mse_loss(diff_coeffs, target_diff_coeffs)
            
            loss = out_loss + 0.1 * coeff_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += out_loss.item()
            total_coeff_loss += coeff_loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for d in train_data:
                    a = torch.tensor([d['a']], device=device, dtype=torch.float)
                    b = torch.tensor([d['b']], device=device, dtype=torch.float)
                    ps, pd, _, _ = model(a, b)
                    if round(ps.item()) == d['sum'] and round(pd.item()) == d['diff']:
                        correct += 1
            
            acc = correct / len(train_data)
            print(f"  Epoch {epoch+1:3d}: loss={total_loss:.4f}, coeff_loss={total_coeff_loss:.4f}, acc={acc:.1%}")
            
            if acc >= 0.99:
                print("  ✓ Trained!")
                break
    
    # Test generalization to larger values
    print("\n[GENERALIZATION TEST]")
    
    model.eval()
    test_ranges = [
        ("Training range", 0, vr),
        ("2x range", 0, vr * 2),
        ("FFT range", -vr * 8, vr * 8),
    ]
    
    for name, lo, hi in test_ranges:
        correct = 0
        total = 200
        
        with torch.no_grad():
            for _ in range(total):
                av = np.random.randint(lo, hi)
                bv = np.random.randint(lo, hi)
                
                a = torch.tensor([av], device=device, dtype=torch.float)
                b = torch.tensor([bv], device=device, dtype=torch.float)
                
                ps, pd, sc, dc = model(a, b)
                
                if round(ps.item()) == av + bv and round(pd.item()) == av - bv:
                    correct += 1
        
        print(f"  {name}: {correct}/{total} = {correct/total:.1%}")
    
    # Check learned coefficients
    print("\n[LEARNED COEFFICIENTS]")
    with torch.no_grad():
        a = torch.tensor([5.0], device=device)
        b = torch.tensor([3.0], device=device)
        _, _, sum_c, diff_c = model(a, b)
        print(f"  Sum coeffs:  ({sum_c[0,0]:.3f}, {sum_c[0,1]:.3f}) - target (1, 1)")
        print(f"  Diff coeffs: ({diff_c[0,0]:.3f}, {diff_c[0,1]:.3f}) - target (1, -1)")
    
    # Full FFT test
    print("\n[FULL FFT TEST]")
    
    N = CONFIG['N']
    
    def run_fft(x):
        values = torch.tensor(x, device=device, dtype=torch.float)
        
        for stage in range(int(np.log2(N))):
            stride = 2 ** stage
            new_values = values.clone()
            
            for i in range(N):
                partner = i ^ stride
                if i < partner:
                    a = values[i:i+1]
                    b = values[partner:partner+1]
                    ps, pd, _, _ = model(a, b)
                    new_values[i] = ps.round()
                    new_values[partner] = pd.round()
            
            values = new_values
        
        return [int(v.item()) for v in values]
    
    def reference_fft(x):
        result = list(x)
        for stage in range(int(np.log2(N))):
            stride = 2 ** stage
            new_result = result.copy()
            for i in range(N):
                partner = i ^ stride
                if i < partner:
                    a, b = result[i], result[partner]
                    new_result[i] = a + b
                    new_result[partner] = a - b
            result = new_result
        return result
    
    correct = 0
    total = 100
    
    for _ in range(total):
        x = [np.random.randint(0, vr) for _ in range(N)]
        expected = reference_fft(x)
        predicted = run_fft(x)
        if expected == predicted:
            correct += 1
    
    fft_acc = correct / total
    print(f"  FFT Accuracy: {correct}/{total} = {fft_acc:.1%}")
    
    # Show examples
    print("\nExamples:")
    for _ in range(3):
        x = [np.random.randint(0, vr) for _ in range(N)]
        expected = reference_fft(x)
        predicted = run_fft(x)
        match = "✓" if expected == predicted else "✗"
        print(f"  {x} → {predicted} {match}")
    
    # Verdict
    print("\n" + "=" * 70)
    if fft_acc >= 0.90:
        print(f"✓ PURE TRIX FFT N={N}: SUCCESS")
        print(f"  Linear residual architecture works!")
        print(f"  Tiles learn coefficients, arithmetic generalizes.")
    else:
        print(f"✗ FFT: {fft_acc:.1%}")
    print("=" * 70)
    
    return fft_acc


if __name__ == "__main__":
    train_and_test()
