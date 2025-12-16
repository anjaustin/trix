#!/usr/bin/env python3
"""
Pure TriX FFT: Composition Test
===============================

We PROVED butterfly works (100%).
Now test: can we COMPOSE it into full FFT?

Strategy:
1. Train butterfly on small fixed range
2. Apply it iteratively for full FFT
3. See if composition holds

CODENAME: ANN WILSON - BARRACUDA
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
    'd_model': 128,
    'num_tiles': 8,
    'num_freqs': 12,
    'epochs': 200,
    'batch_size': 64,
    'lr': 0.002,
    'seed': 1122911624,
}


class FastPureTriXButterfly(nn.Module):
    """
    Vectorized butterfly - no per-sample loops.
    Trained on small range, applied to FFT.
    """
    
    def __init__(self, d_model=64, num_tiles=4, num_freqs=8):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_freqs = num_freqs
        
        # Fourier feature projection
        fourier_dim = 2 * num_freqs * 2  # a and b
        self.input_proj = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Soft routing (differentiable)
        self.router = nn.Linear(d_model, num_tiles)
        
        # Tile networks
        self.tile_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_tiles)
        ])
        
        # Separate heads for sum and diff
        self.sum_head = nn.Linear(d_model, 1)
        self.diff_head = nn.Linear(d_model, 1)
        
        self.temperature = 1.0
    
    def _fourier_features(self, x, scale=256.0):
        """Fourier encode with adjustable scale."""
        x_norm = x.unsqueeze(-1) * (2 * np.pi / scale)
        freqs = (2 ** torch.arange(self.num_freqs, device=x.device, dtype=torch.float))
        angles = x_norm * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, a, b, scale=256.0):
        """
        Vectorized butterfly.
        
        Args:
            a, b: (batch,) float values
            scale: normalization scale for Fourier features
        
        Returns:
            sum_pred, diff_pred: (batch,) predictions
        """
        # Encode
        a_feat = self._fourier_features(a, scale)
        b_feat = self._fourier_features(b, scale)
        x = torch.cat([a_feat, b_feat], dim=-1)
        x = self.input_proj(x)  # (batch, d_model)
        
        # Soft routing for training
        route_weights = F.softmax(self.router(x) / self.temperature, dim=-1)  # (batch, num_tiles)
        
        # Compute all tiles
        tile_outputs = torch.stack([net(x) for net in self.tile_nets], dim=1)  # (batch, num_tiles, d_model)
        
        # Weighted combination
        combined = (route_weights.unsqueeze(-1) * tile_outputs).sum(dim=1)  # (batch, d_model)
        
        # Decode
        sum_pred = self.sum_head(combined).squeeze(-1)
        diff_pred = self.diff_head(combined).squeeze(-1)
        
        return sum_pred, diff_pred


def generate_butterfly_data(value_range=16, num_samples=5000):
    """Generate butterfly training data."""
    data = []
    for _ in range(num_samples):
        a = np.random.randint(0, value_range)
        b = np.random.randint(0, value_range)
        data.append({'a': a, 'b': b, 'sum': a + b, 'diff': a - b})
    return data


def train_butterfly(model, device, value_range=16, epochs=100):
    """Train butterfly on FFT range (includes intermediate values)."""
    print("\n[TRAINING BUTTERFLY]")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # FFT intermediate range: for N=8 with inputs 0-15
    # Final values can be roughly -128 to 128
    fft_range = value_range * CONFIG['N']  # Conservative estimate
    
    # Generate training data across full FFT range
    all_data = []
    # Small values (original inputs)
    for a in range(value_range):
        for b in range(value_range):
            all_data.append({'a': a, 'b': b, 'sum': a + b, 'diff': a - b})
    
    # Intermediate values (what FFT produces)
    np.random.seed(CONFIG['seed'])
    for _ in range(2000):
        a = np.random.randint(-fft_range, fft_range + 1)
        b = np.random.randint(-fft_range, fft_range + 1)
        all_data.append({'a': a, 'b': b, 'sum': a + b, 'diff': a - b})
    
    print(f"  Training on {len(all_data)} pairs (range: -{fft_range} to {fft_range})")
    
    batch_size = CONFIG['batch_size']
    
    for epoch in range(epochs):
        model.train()
        
        np.random.shuffle(all_data)
        total_loss = 0
        
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i+batch_size]
            
            a = torch.tensor([d['a'] for d in batch], device=device, dtype=torch.float)
            b = torch.tensor([d['b'] for d in batch], device=device, dtype=torch.float)
            target_sum = torch.tensor([d['sum'] for d in batch], device=device, dtype=torch.float)
            target_diff = torch.tensor([d['diff'] for d in batch], device=device, dtype=torch.float)
            
            pred_sum, pred_diff = model(a, b, scale=fft_range * 2)
            
            loss = F.mse_loss(pred_sum, target_sum) + F.mse_loss(pred_diff, target_diff)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            # Evaluate
            model.eval()
            correct = 0
            with torch.no_grad():
                for d in all_data:
                    a = torch.tensor([d['a']], device=device, dtype=torch.float)
                    b = torch.tensor([d['b']], device=device, dtype=torch.float)
                    ps, pd = model(a, b, scale=fft_range * 2)
                    if round(ps.item()) == d['sum'] and round(pd.item()) == d['diff']:
                        correct += 1
            
            acc = correct / len(all_data)
            print(f"  Epoch {epoch+1:3d}: loss={total_loss:.4f}, acc={acc:.1%}")
            
            if acc >= 0.99:
                print("  ✓ Butterfly trained!")
                return True
    
    return False


def run_fft(model, x, device, fft_range):
    """
    Run full FFT using trained butterfly.
    """
    model.eval()
    N = len(x)
    num_stages = int(np.log2(N))
    
    values = torch.tensor(x, device=device, dtype=torch.float)
    
    with torch.no_grad():
        for stage in range(num_stages):
            stride = 2 ** stage
            new_values = values.clone()
            
            for i in range(N):
                partner = i ^ stride
                if i < partner:
                    a = values[i:i+1]
                    b = values[partner:partner+1]
                    
                    # Use same scale as training
                    ps, pd = model(a, b, scale=fft_range * 2)
                    
                    new_values[i] = ps.round()
                    new_values[partner] = pd.round()
            
            values = new_values
    
    return [int(v.item()) for v in values]


def reference_fft(x):
    """Ground truth."""
    N = len(x)
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    N = CONFIG['N']
    
    print("\n" + "=" * 70)
    print(f"PURE TRIX FFT - COMPOSITION TEST (N={N})")
    print("Train butterfly, compose into FFT")
    print("=" * 70)
    
    # Create and train butterfly
    model = FastPureTriXButterfly(
        d_model=CONFIG['d_model'],
        num_tiles=CONFIG['num_tiles'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")
    
    success = train_butterfly(model, device, CONFIG['value_range'], CONFIG['epochs'])
    
    if not success:
        print("\n✗ Butterfly training incomplete")
        return 0.0
    
    # Test FFT composition
    print("\n[FFT COMPOSITION TEST]")
    
    num_correct = 0
    num_tests = 100
    
    for i in range(num_tests):
        x = [np.random.randint(0, CONFIG['value_range']) for _ in range(N)]
        
        expected = reference_fft(x)
        predicted = run_fft(model, x, device)
        
        if expected == predicted:
            num_correct += 1
        elif i < 5:  # Show first few failures
            print(f"  ✗ In: {x}")
            print(f"    Exp: {expected}")
            print(f"    Got: {predicted}")
    
    fft_accuracy = num_correct / num_tests
    
    print(f"\nFFT Accuracy: {num_correct}/{num_tests} = {fft_accuracy:.1%}")
    
    # Show successes
    print("\nSuccessful examples:")
    shown = 0
    for _ in range(20):
        x = [np.random.randint(0, CONFIG['value_range']) for _ in range(N)]
        expected = reference_fft(x)
        predicted = run_fft(model, x, device)
        
        if expected == predicted and shown < 3:
            print(f"  ✓ {x} → {predicted}")
            shown += 1
    
    # Verdict
    print("\n" + "=" * 70)
    if fft_accuracy >= 0.90:
        print(f"✓ PURE TRIX FFT N={N}: SUCCESS")
        print(f"  FFT Accuracy: {fft_accuracy:.1%}")
        print("  Butterfly composes into full FFT!")
    elif fft_accuracy >= 0.50:
        print(f"◐ PARTIAL: {fft_accuracy:.1%}")
        print("  Butterfly works but composition has errors")
    else:
        print(f"✗ COMPOSITION FAILS: {fft_accuracy:.1%}")
        print("  Butterfly trained but doesn't compose")
    print("=" * 70)
    
    return fft_accuracy


if __name__ == "__main__":
    main()
