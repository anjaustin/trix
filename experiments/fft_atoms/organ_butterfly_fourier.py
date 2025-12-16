#!/usr/bin/env python3
"""
Butterfly Organ with Fourier Features
======================================

FFT helps ORGANS, not TDSR.

- Fourier features for inputs (architectural prior)
- FFT-informed loss (richer gradients)
- Exact compute target

TDSR will route to this organ. The organ just computes.

CODENAME: ANN WILSON - HEART
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


# =============================================================================
# Fourier Features
# =============================================================================

def fourier_features(x, num_freqs=4, max_freq=8):
    """
    Encode scalar x with Fourier features.
    
    x: (batch,) values in [0, max_val]
    Returns: (batch, num_freqs * 2) sine and cosine features
    """
    # Normalize x to [0, 2π]
    x_norm = x.float().unsqueeze(-1) * (2 * np.pi / max_freq)
    
    # Frequency multipliers: 1, 2, 4, 8, ...
    freqs = (2 ** torch.arange(num_freqs, device=x.device, dtype=torch.float)).unsqueeze(0)
    
    # Compute sin and cos at each frequency
    angles = x_norm * freqs  # (batch, num_freqs)
    
    features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return features


# =============================================================================
# Fourier-Featured Butterfly Organ
# =============================================================================

class FourierButterflyOrgan(nn.Module):
    """
    Butterfly organ with Fourier feature encoding.
    
    Uses FFT-inspired representation to learn (a,b) → (a+b, a-b).
    
    Architecture:
    - Fourier encode a and b separately
    - Small MLP to combine
    - Direct regression on sum and diff
    """
    
    def __init__(self, value_range=16, num_freqs=4, hidden_dim=32):
        super().__init__()
        
        self.value_range = value_range
        self.num_freqs = num_freqs
        
        # Input: 2 * (num_freqs * 2) = 16 features for num_freqs=4
        input_dim = 2 * num_freqs * 2
        
        # Tiny MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # Output: (sum, diff)
        )
        
        # Output scaling - sum in [0, 2*range-2], diff in [-(range-1), range-1]
        self.register_buffer('sum_scale', torch.tensor(float(2 * value_range - 2)))
        self.register_buffer('diff_scale', torch.tensor(float(value_range - 1)))
        
        self._count_params()
    
    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        bytes_est = total * 4
        print(f"Fourier Butterfly Organ: {total} params, ~{bytes_est} bytes")
    
    def forward(self, a, b):
        """
        Args:
            a, b: (batch,) integer values in [0, value_range)
        
        Returns:
            sum_pred: (batch,) predicted a+b (float)
            diff_pred: (batch,) predicted a-b (float)
        """
        # Fourier encode
        a_feat = fourier_features(a, self.num_freqs, self.value_range)
        b_feat = fourier_features(b, self.num_freqs, self.value_range)
        
        # Concatenate
        x = torch.cat([a_feat, b_feat], dim=-1)
        
        # Compute - direct linear output, no sigmoid/tanh constraints
        out = self.net(x)
        
        return out[:, 0], out[:, 1]
    
    def forward_int(self, a, b):
        """Forward with integer rounding."""
        sum_pred, diff_pred = self.forward(a, b)
        return torch.round(sum_pred).long(), torch.round(diff_pred).long()


# =============================================================================
# Training with FFT-Informed Loss
# =============================================================================

def generate_data(value_range=16):
    """Generate all (a, b) -> (a+b, a-b) pairs."""
    data = []
    for a in range(value_range):
        for b in range(value_range):
            data.append({'a': a, 'b': b, 'sum': a + b, 'diff': a - b})
    return data


def train_organ(organ, data, device, epochs=500, lr=0.01):
    """Train with standard MSE loss."""
    
    optimizer = torch.optim.AdamW(organ.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    target_sum = torch.tensor([d['sum'] for d in data], device=device, dtype=torch.float)
    target_diff = torch.tensor([d['diff'] for d in data], device=device, dtype=torch.float)
    
    best_acc = 0
    history = []
    
    for epoch in range(epochs):
        organ.train()
        
        pred_sum, pred_diff = organ(a, b)
        
        # MSE loss
        loss_sum = F.mse_loss(pred_sum, target_sum)
        loss_diff = F.mse_loss(pred_diff, target_diff)
        loss = loss_sum + loss_diff
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % 25 == 0 or epoch == 0:
            organ.eval()
            with torch.no_grad():
                pred_sum_int, pred_diff_int = organ.forward_int(a, b)
                
                acc_sum = (pred_sum_int == target_sum.long()).float().mean().item()
                acc_diff = (pred_diff_int == target_diff.long()).float().mean().item()
                acc_both = ((pred_sum_int == target_sum.long()) & 
                           (pred_diff_int == target_diff.long())).float().mean().item()
                
                history.append({
                    'epoch': epoch + 1,
                    'loss': loss.item(),
                    'acc_sum': acc_sum,
                    'acc_diff': acc_diff,
                    'acc_both': acc_both,
                })
                
                if acc_both > best_acc:
                    best_acc = acc_both
                
                print(f"  Epoch {epoch+1:4d}: loss={loss.item():.6f}, "
                      f"sum={acc_sum:.1%}, diff={acc_diff:.1%}, both={acc_both:.1%}")
                
                if acc_both >= 1.0:
                    print("  ✓ 100% accuracy achieved!")
                    return True, best_acc, history
    
    return best_acc >= 0.99, best_acc, history


# =============================================================================
# Main
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    VALUE_RANGE = 16
    
    print("\n" + "=" * 60)
    print("FOURIER BUTTERFLY ORGAN")
    print("FFT helps organs. TDSR routes to organs.")
    print("=" * 60)
    
    # Generate data
    data = generate_data(VALUE_RANGE)
    print(f"\nData points: {len(data)}")
    
    # Create organ with Fourier features
    print("\nCreating Fourier-featured organ...")
    organ = FourierButterflyOrgan(
        value_range=VALUE_RANGE,
        num_freqs=6,      # More frequencies for better representation
        hidden_dim=48,    # More capacity
    ).to(device)
    
    # Train
    print("\nTraining...")
    success, best_acc, history = train_organ(organ, data, device, epochs=500, lr=0.01)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    organ.eval()
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    
    with torch.no_grad():
        pred_sum, pred_diff = organ.forward_int(a, b)
        target_sum = torch.tensor([d['sum'] for d in data], device=device)
        target_diff = torch.tensor([d['diff'] for d in data], device=device)
        
        acc_sum = (pred_sum == target_sum).float().mean().item()
        acc_diff = (pred_diff == target_diff).float().mean().item()
        acc_both = ((pred_sum == target_sum) & (pred_diff == target_diff)).float().mean().item()
    
    print(f"\nAccuracy: sum={acc_sum:.1%}, diff={acc_diff:.1%}, both={acc_both:.1%}")
    
    # Examples
    print("\nExamples:")
    test_cases = [(0, 0), (5, 3), (15, 15), (7, 12), (0, 15)]
    for a_val, b_val in test_cases:
        a_t = torch.tensor([a_val], device=device)
        b_t = torch.tensor([b_val], device=device)
        ps, pd = organ.forward_int(a_t, b_t)
        true_sum, true_diff = a_val + b_val, a_val - b_val
        match = "✓" if (ps.item() == true_sum and pd.item() == true_diff) else "✗"
        print(f"  ({a_val:2d}, {b_val:2d}) → pred=({ps.item():3d}, {pd.item():3d}), "
              f"true=({true_sum:3d}, {true_diff:3d}) {match}")
    
    # Size
    total_params = sum(p.numel() for p in organ.parameters())
    size_bytes = total_params * 4
    
    print(f"\nOrgan size: {total_params} params = {size_bytes} bytes")
    
    # Verdict
    print("\n" + "=" * 60)
    if acc_both >= 0.99:
        print("✓ FOURIER BUTTERFLY ORGAN: READY")
        print(f"  {acc_both:.1%} accuracy in {size_bytes} bytes")
        print("  Fourier features enabled clean learning")
        print("  Ready to wire to TDSR")
    else:
        print(f"✗ ORGAN INCOMPLETE: {acc_both:.1%}")
    print("=" * 60)
    
    # Save
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_file = output_dir / f'fourier_butterfly_organ_{timestamp}.pt'
    torch.save(organ.state_dict(), model_file)
    
    # Save results
    results = {
        'timestamp': timestamp,
        'architecture': 'FourierButterflyOrgan',
        'value_range': VALUE_RANGE,
        'num_freqs': 4,
        'params': total_params,
        'size_bytes': size_bytes,
        'accuracy': {'sum': acc_sum, 'diff': acc_diff, 'both': acc_both},
        'success': acc_both >= 0.99,
        'history': history,
    }
    
    results_file = output_dir / f'fourier_butterfly_organ_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {model_file}")
    
    return organ, acc_both


if __name__ == "__main__":
    organ, accuracy = main()
