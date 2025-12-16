#!/usr/bin/env python3
"""
Pure TriX FFT: Discrete Operation Selection
===========================================

Key insight: arithmetic errors compound in FFT.
Solution: Tiles hold EXACT operations, routing SELECTS.

Operations are not learned - they're provided.
Routing learns WHEN to use each operation.

This is still pure TriX:
- Tiles are the operations (fixed microcode)
- Routing is the control (learned selection)

Like a CPU: opcodes are fixed, control flow is learned.

CODENAME: ANN WILSON - ALONE
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
    'num_freqs': 8,
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.01,
    'seed': 1122911624,
}


class DiscreteOpButterfly(nn.Module):
    """
    Butterfly with discrete operation selection.
    
    Fixed operations (tiles/microcode):
      Op 0: (a, b) -> a + b (SUM)
      Op 1: (a, b) -> a - b (DIFF)
    
    Learned routing selects which operation.
    Arithmetic is EXACT because ops are fixed.
    """
    
    def __init__(self, d_model=64, num_freqs=8):
        super().__init__()
        
        self.d_model = d_model
        self.num_freqs = num_freqs
        
        # Operations are FIXED - not learned
        # This is the "microcode" - exact operations
        # Op 0: sum (coeff_a=1, coeff_b=1)
        # Op 1: diff (coeff_a=1, coeff_b=-1)
        self.register_buffer('op_coeffs', torch.tensor([
            [1.0, 1.0],   # SUM
            [1.0, -1.0],  # DIFF
        ]))
        
        # Context encoder (for routing decision)
        fourier_dim = 2 * num_freqs
        self.encoder = nn.Sequential(
            nn.Linear(fourier_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        
        # Router for SUM path
        self.router_sum = nn.Linear(d_model, 2)
        
        # Router for DIFF path  
        self.router_diff = nn.Linear(d_model, 2)
        
        self.temperature = 0.1  # Low temp = hard selection
    
    def _fourier_features(self, x, scale=32.0):
        """Scale-invariant Fourier encoding."""
        x_norm = x.unsqueeze(-1) / scale
        freqs = torch.arange(1, self.num_freqs + 1, device=x.device, dtype=torch.float)
        angles = x_norm * freqs * np.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, a, b, hard=True):
        """
        Args:
            a, b: (batch,) input values
            hard: if True, use argmax selection (exact arithmetic)
        
        Returns:
            sum_pred, diff_pred: (batch,) exact outputs
        """
        # Context
        a_feat = self._fourier_features(a)
        b_feat = self._fourier_features(b)
        ctx = torch.cat([a_feat, b_feat], dim=-1)
        ctx = self.encoder(ctx)  # (batch, d_model)
        
        # Route for SUM output
        sum_logits = self.router_sum(ctx) / self.temperature
        if hard:
            sum_op = sum_logits.argmax(dim=-1)  # (batch,)
            sum_coeffs = self.op_coeffs[sum_op]  # (batch, 2)
        else:
            sum_weights = F.softmax(sum_logits, dim=-1)
            sum_coeffs = sum_weights @ self.op_coeffs  # (batch, 2)
        
        # Route for DIFF output
        diff_logits = self.router_diff(ctx) / self.temperature
        if hard:
            diff_op = diff_logits.argmax(dim=-1)
            diff_coeffs = self.op_coeffs[diff_op]
        else:
            diff_weights = F.softmax(diff_logits, dim=-1)
            diff_coeffs = diff_weights @ self.op_coeffs
        
        # Apply coefficients - EXACT arithmetic
        sum_pred = sum_coeffs[:, 0] * a + sum_coeffs[:, 1] * b
        diff_pred = diff_coeffs[:, 0] * a + diff_coeffs[:, 1] * b
        
        return sum_pred, diff_pred
    
    def get_selections(self, a, b):
        """Return which ops were selected."""
        a_feat = self._fourier_features(a)
        b_feat = self._fourier_features(b)
        ctx = torch.cat([a_feat, b_feat], dim=-1)
        ctx = self.encoder(ctx)
        
        sum_op = self.router_sum(ctx).argmax(dim=-1)
        diff_op = self.router_diff(ctx).argmax(dim=-1)
        
        return sum_op, diff_op


def train_and_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("\n" + "=" * 70)
    print("PURE TRIX FFT - DISCRETE OPERATION SELECTION")
    print("Fixed ops (tiles), learned routing (control)")
    print("=" * 70)
    
    model = DiscreteOpButterfly(
        d_model=CONFIG['d_model'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")
    print(f"Fixed ops: SUM=(1,1), DIFF=(1,-1)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    # Training data
    train_data = []
    vr = CONFIG['value_range']
    for a in range(vr):
        for b in range(vr):
            train_data.append({'a': a, 'b': b, 'sum': a + b, 'diff': a - b})
    
    print(f"Training on {len(train_data)} pairs")
    
    batch_size = CONFIG['batch_size']
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        np.random.shuffle(train_data)
        total_loss = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            a = torch.tensor([d['a'] for d in batch], device=device, dtype=torch.float)
            b = torch.tensor([d['b'] for d in batch], device=device, dtype=torch.float)
            target_sum = torch.tensor([d['sum'] for d in batch], device=device, dtype=torch.float)
            target_diff = torch.tensor([d['diff'] for d in batch], device=device, dtype=torch.float)
            
            # Use soft routing for training (gradients flow)
            sum_pred, diff_pred = model(a, b, hard=False)
            
            loss = F.mse_loss(sum_pred, target_sum) + F.mse_loss(diff_pred, target_diff)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for d in train_data:
                    a = torch.tensor([d['a']], device=device, dtype=torch.float)
                    b = torch.tensor([d['b']], device=device, dtype=torch.float)
                    ps, pd = model(a, b, hard=True)  # Hard selection for eval
                    if int(ps.item()) == d['sum'] and int(pd.item()) == d['diff']:
                        correct += 1
            
            acc = correct / len(train_data)
            print(f"  Epoch {epoch+1:3d}: loss={total_loss:.4f}, acc={acc:.1%}")
            
            if acc >= 0.99:
                print("  ✓ Trained!")
                break
    
    # Check op selections
    print("\n[OPERATION SELECTION ANALYSIS]")
    model.eval()
    sum_selects = [0, 0]
    diff_selects = [0, 0]
    
    with torch.no_grad():
        for d in train_data:
            a = torch.tensor([d['a']], device=device, dtype=torch.float)
            b = torch.tensor([d['b']], device=device, dtype=torch.float)
            s_op, d_op = model.get_selections(a, b)
            sum_selects[s_op.item()] += 1
            diff_selects[d_op.item()] += 1
    
    print(f"  SUM path: Op0(SUM)={sum_selects[0]}, Op1(DIFF)={sum_selects[1]}")
    print(f"  DIFF path: Op0(SUM)={diff_selects[0]}, Op1(DIFF)={diff_selects[1]}")
    
    # Generalization test
    print("\n[GENERALIZATION TEST]")
    
    test_ranges = [
        ("Training range", 0, vr),
        ("2x range", 0, vr * 2),
        ("4x range", 0, vr * 4),
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
                
                ps, pd = model(a, b, hard=True)
                
                # Exact comparison (ops are exact)
                if int(ps.item()) == av + bv and int(pd.item()) == av - bv:
                    correct += 1
        
        print(f"  {name}: {correct}/{total} = {correct/total:.1%}")
    
    # Full FFT test
    print("\n[FULL FFT TEST]")
    
    N = CONFIG['N']
    
    def run_fft(x):
        model.eval()
        values = torch.tensor(x, device=device, dtype=torch.float)
        
        with torch.no_grad():
            for stage in range(int(np.log2(N))):
                stride = 2 ** stage
                new_values = values.clone()
                
                for i in range(N):
                    partner = i ^ stride
                    if i < partner:
                        a = values[i:i+1]
                        b = values[partner:partner+1]
                        ps, pd = model(a, b, hard=True)
                        new_values[i] = ps
                        new_values[partner] = pd
                
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
    
    # Examples
    print("\nExamples:")
    for _ in range(5):
        x = [np.random.randint(0, vr) for _ in range(N)]
        expected = reference_fft(x)
        predicted = run_fft(x)
        match = "✓" if expected == predicted else "✗"
        print(f"  {x}")
        print(f"    → {predicted} {match}")
    
    # Verdict
    print("\n" + "=" * 70)
    if fft_acc >= 0.95:
        print(f"✓ PURE TRIX FFT N={N}: SUCCESS!")
        print(f"  Discrete ops = exact arithmetic")
        print(f"  Routing learns control, tiles ARE the operations")
    elif fft_acc >= 0.50:
        print(f"◐ PARTIAL: {fft_acc:.1%}")
    else:
        print(f"✗ FFT: {fft_acc:.1%}")
    print("=" * 70)
    
    return fft_acc


if __name__ == "__main__":
    train_and_test()
