#!/usr/bin/env python3
"""
Pure TriX FFT: Twiddle Factors (Complex Rotation)
=================================================

The real boss: complex multiplication by roots of unity.

Architecture (same pattern that won):
- Fixed microcode: exact twiddle factors W_k = e^{-2πik/N}
- Learned routing: which twiddle based on (stage, position)

No coefficient learning. Just more opcodes.

CODENAME: ANN WILSON - MAGIC MAN
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
    'value_range': 8,  # Smaller for complex (real + imag)
    'd_model': 96,
    'num_freqs': 10,
    'epochs': 200,
    'batch_size': 64,
    'lr': 0.005,
    'seed': 1122911624,
}


def compute_twiddle_factors(N):
    """
    Compute all twiddle factors W_k = e^{-2πik/N} for k = 0..N-1
    
    Returns: (N, 2) tensor of (real, imag) pairs
    """
    k = torch.arange(N, dtype=torch.float)
    angles = -2 * np.pi * k / N
    real = torch.cos(angles)
    imag = torch.sin(angles)
    return torch.stack([real, imag], dim=1)


def complex_multiply(a_real, a_imag, b_real, b_imag):
    """(a_real + i*a_imag) * (b_real + i*b_imag)"""
    out_real = a_real * b_real - a_imag * b_imag
    out_imag = a_real * b_imag + a_imag * b_real
    return out_real, out_imag


class TwiddleFFT(nn.Module):
    """
    Complex FFT with twiddle factors.
    
    Fixed microcode:
      - Twiddle factors W_0, W_1, ..., W_{N-1} (exact)
      - Butterfly ops: ADD, SUB (exact)
    
    Learned routing:
      - Which twiddle to apply based on (stage, position)
    """
    
    def __init__(self, N=8, d_model=64, num_freqs=8):
        super().__init__()
        
        self.N = N
        self.num_stages = int(np.log2(N))
        self.d_model = d_model
        self.num_freqs = num_freqs
        
        # FIXED MICROCODE: Exact twiddle factors
        twiddles = compute_twiddle_factors(N)
        self.register_buffer('twiddle_factors', twiddles)  # (N, 2)
        
        # Context encoding
        fourier_dim = 2 * num_freqs
        
        # Encode: stage, position, values (a_real, a_imag, b_real, b_imag)
        self.stage_embed = nn.Embedding(self.num_stages, d_model // 4)
        self.pos_embed = nn.Embedding(N, d_model // 4)
        
        # Value encoder (4 values: a_real, a_imag, b_real, b_imag)
        self.value_encoder = nn.Sequential(
            nn.Linear(fourier_dim * 4, d_model // 2),
            nn.GELU(),
        )
        
        # Combined projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Twiddle router: selects which W_k to use
        self.twiddle_router = nn.Linear(d_model, N)
        
        self.temperature = 0.1
    
    def _fourier_features(self, x, scale=32.0):
        """Fourier encode scalar values."""
        x_norm = x.unsqueeze(-1) / scale
        freqs = torch.arange(1, self.num_freqs + 1, device=x.device, dtype=torch.float)
        angles = x_norm * freqs * np.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, stage, pos, a_real, a_imag, b_real, b_imag, hard=True):
        """
        Complex butterfly with twiddle.
        
        Computes:
          out1 = a + W_k * b
          out2 = a - W_k * b
        
        Where W_k is selected by routing based on (stage, pos).
        
        Args:
            stage: (batch,) stage index
            pos: (batch,) position index
            a_real, a_imag: (batch,) complex number a
            b_real, b_imag: (batch,) complex number b
            hard: use argmax selection (exact) vs soft (training)
        
        Returns:
            out1_real, out1_imag, out2_real, out2_imag
        """
        batch_size = stage.shape[0]
        
        # Encode context
        stage_emb = self.stage_embed(stage)  # (batch, d_model//4)
        pos_emb = self.pos_embed(pos)  # (batch, d_model//4)
        
        # Encode values
        ar_feat = self._fourier_features(a_real)
        ai_feat = self._fourier_features(a_imag)
        br_feat = self._fourier_features(b_real)
        bi_feat = self._fourier_features(b_imag)
        val_feat = torch.cat([ar_feat, ai_feat, br_feat, bi_feat], dim=-1)
        val_emb = self.value_encoder(val_feat)  # (batch, d_model//2)
        
        # Combine
        ctx = torch.cat([stage_emb, pos_emb, val_emb], dim=-1)
        ctx = self.proj(ctx)  # (batch, d_model)
        
        # Route to twiddle factor
        twiddle_logits = self.twiddle_router(ctx) / self.temperature
        
        if hard:
            twiddle_idx = twiddle_logits.argmax(dim=-1)  # (batch,)
            W = self.twiddle_factors[twiddle_idx]  # (batch, 2)
        else:
            twiddle_weights = F.softmax(twiddle_logits, dim=-1)  # (batch, N)
            W = twiddle_weights @ self.twiddle_factors  # (batch, 2)
        
        W_real = W[:, 0]
        W_imag = W[:, 1]
        
        # Complex multiply: W * b
        Wb_real, Wb_imag = complex_multiply(W_real, W_imag, b_real, b_imag)
        
        # Butterfly: out1 = a + W*b, out2 = a - W*b
        out1_real = a_real + Wb_real
        out1_imag = a_imag + Wb_imag
        out2_real = a_real - Wb_real
        out2_imag = a_imag - Wb_imag
        
        return out1_real, out1_imag, out2_real, out2_imag
    
    def get_twiddle_selection(self, stage, pos, a_real, a_imag, b_real, b_imag):
        """Return which twiddle was selected."""
        stage_emb = self.stage_embed(stage)
        pos_emb = self.pos_embed(pos)
        
        ar_feat = self._fourier_features(a_real)
        ai_feat = self._fourier_features(a_imag)
        br_feat = self._fourier_features(b_real)
        bi_feat = self._fourier_features(b_imag)
        val_feat = torch.cat([ar_feat, ai_feat, br_feat, bi_feat], dim=-1)
        val_emb = self.value_encoder(val_feat)
        
        ctx = torch.cat([stage_emb, pos_emb, val_emb], dim=-1)
        ctx = self.proj(ctx)
        
        twiddle_idx = self.twiddle_router(ctx).argmax(dim=-1)
        return twiddle_idx


def get_correct_twiddle_index(stage, pos, N):
    """
    Compute the correct twiddle index for FFT.
    
    For radix-2 DIT FFT:
      W_k where k = (pos % (2^(stage+1))) * (N / 2^(stage+1))
    
    Simplified for N=8:
      Stage 0: all use W_0 (no rotation)
      Stage 1: pos 0,1 use W_0; pos 2,3 use W_2; pos 4,5 use W_0; pos 6,7 use W_2
      Stage 2: pos 0 W_0, pos 1 W_1, pos 2 W_2, pos 3 W_3, pos 4 W_0, ...
    """
    # For DIT (decimation in time) FFT
    # The twiddle factor index depends on position within the butterfly group
    group_size = 2 ** (stage + 1)
    pos_in_group = pos % group_size
    
    # Only the second half of each group gets non-trivial twiddle
    if pos_in_group < group_size // 2:
        # First element of pair - no twiddle needed (W_0 = 1)
        # But we still compute butterfly
        return 0
    else:
        # Second element - apply twiddle
        k = (pos_in_group - group_size // 2) * (N // group_size)
        return k % N


def generate_training_data(N=8, value_range=8, num_samples=5000):
    """
    Generate training data for twiddle FFT.
    
    For each (stage, pos), compute correct butterfly with twiddle.
    """
    num_stages = int(np.log2(N))
    twiddles = compute_twiddle_factors(N).numpy()
    
    data = []
    
    for _ in range(num_samples):
        stage = np.random.randint(0, num_stages)
        pos = np.random.randint(0, N)
        
        # Random complex inputs
        a_real = np.random.uniform(-value_range, value_range)
        a_imag = np.random.uniform(-value_range, value_range)
        b_real = np.random.uniform(-value_range, value_range)
        b_imag = np.random.uniform(-value_range, value_range)
        
        # Get correct twiddle
        k = get_correct_twiddle_index(stage, pos, N)
        W_real, W_imag = twiddles[k]
        
        # Complex multiply: W * b
        Wb_real = W_real * b_real - W_imag * b_imag
        Wb_imag = W_real * b_imag + W_imag * b_real
        
        # Butterfly
        out1_real = a_real + Wb_real
        out1_imag = a_imag + Wb_imag
        out2_real = a_real - Wb_real
        out2_imag = a_imag - Wb_imag
        
        data.append({
            'stage': stage,
            'pos': pos,
            'a_real': a_real,
            'a_imag': a_imag,
            'b_real': b_real,
            'b_imag': b_imag,
            'twiddle_k': k,
            'out1_real': out1_real,
            'out1_imag': out1_imag,
            'out2_real': out2_real,
            'out2_imag': out2_imag,
        })
    
    return data


def train_and_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    N = CONFIG['N']
    
    print("\n" + "=" * 70)
    print(f"PURE TRIX FFT - TWIDDLE FACTORS (N={N})")
    print("Fixed microcode: W_k = e^{-2πik/N}")
    print("Learned routing: which twiddle for (stage, pos)")
    print("=" * 70)
    
    # Show twiddle factors
    twiddles = compute_twiddle_factors(N)
    print("\nTwiddle factors (microcode):")
    for k in range(N):
        W = twiddles[k]
        print(f"  W_{k} = {W[0]:.4f} + {W[1]:.4f}i")
    
    # Create model
    model = TwiddleFFT(
        N=N,
        d_model=CONFIG['d_model'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} params")
    
    # Generate data (more samples for complete coverage)
    train_data = generate_training_data(N, CONFIG['value_range'], num_samples=8000)
    test_data = generate_training_data(N, CONFIG['value_range'], num_samples=1000)
    
    print(f"Data: {len(train_data)} train, {len(test_data)} test")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    batch_size = CONFIG['batch_size']
    
    print("\n[TRAINING]")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        np.random.shuffle(train_data)
        total_loss = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            stage = torch.tensor([d['stage'] for d in batch], device=device)
            pos = torch.tensor([d['pos'] for d in batch], device=device)
            a_real = torch.tensor([d['a_real'] for d in batch], device=device, dtype=torch.float)
            a_imag = torch.tensor([d['a_imag'] for d in batch], device=device, dtype=torch.float)
            b_real = torch.tensor([d['b_real'] for d in batch], device=device, dtype=torch.float)
            b_imag = torch.tensor([d['b_imag'] for d in batch], device=device, dtype=torch.float)
            
            target_out1_real = torch.tensor([d['out1_real'] for d in batch], device=device, dtype=torch.float)
            target_out1_imag = torch.tensor([d['out1_imag'] for d in batch], device=device, dtype=torch.float)
            target_out2_real = torch.tensor([d['out2_real'] for d in batch], device=device, dtype=torch.float)
            target_out2_imag = torch.tensor([d['out2_imag'] for d in batch], device=device, dtype=torch.float)
            
            o1r, o1i, o2r, o2i = model(stage, pos, a_real, a_imag, b_real, b_imag, hard=False)
            
            loss = (F.mse_loss(o1r, target_out1_real) + 
                    F.mse_loss(o1i, target_out1_imag) +
                    F.mse_loss(o2r, target_out2_real) +
                    F.mse_loss(o2i, target_out2_imag))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 15 == 0:
            # Evaluate
            model.eval()
            correct_twiddle = 0
            correct_output = 0
            
            with torch.no_grad():
                for d in test_data:
                    stage = torch.tensor([d['stage']], device=device)
                    pos = torch.tensor([d['pos']], device=device)
                    a_real = torch.tensor([d['a_real']], device=device, dtype=torch.float)
                    a_imag = torch.tensor([d['a_imag']], device=device, dtype=torch.float)
                    b_real = torch.tensor([d['b_real']], device=device, dtype=torch.float)
                    b_imag = torch.tensor([d['b_imag']], device=device, dtype=torch.float)
                    
                    # Check twiddle selection
                    selected_k = model.get_twiddle_selection(stage, pos, a_real, a_imag, b_real, b_imag)
                    if selected_k.item() == d['twiddle_k']:
                        correct_twiddle += 1
                    
                    # Check output (with hard selection)
                    o1r, o1i, o2r, o2i = model(stage, pos, a_real, a_imag, b_real, b_imag, hard=True)
                    
                    tol = 1e-4
                    if (abs(o1r.item() - d['out1_real']) < tol and
                        abs(o1i.item() - d['out1_imag']) < tol and
                        abs(o2r.item() - d['out2_real']) < tol and
                        abs(o2i.item() - d['out2_imag']) < tol):
                        correct_output += 1
            
            twiddle_acc = correct_twiddle / len(test_data)
            output_acc = correct_output / len(test_data)
            
            print(f"  Epoch {epoch+1:3d}: loss={total_loss:.4f}, "
                  f"twiddle={twiddle_acc:.1%}, output={output_acc:.1%}")
            
            if output_acc >= 0.99:
                print("  ✓ Trained!")
                break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    model.eval()
    
    # Twiddle selection analysis by (stage, pos)
    print("\n[TWIDDLE SELECTION BY POSITION]")
    
    twiddle_counts = {}  # (stage, pos) -> {k: count}
    
    with torch.no_grad():
        for d in test_data:
            stage = torch.tensor([d['stage']], device=device)
            pos = torch.tensor([d['pos']], device=device)
            a_real = torch.tensor([d['a_real']], device=device, dtype=torch.float)
            a_imag = torch.tensor([d['a_imag']], device=device, dtype=torch.float)
            b_real = torch.tensor([d['b_real']], device=device, dtype=torch.float)
            b_imag = torch.tensor([d['b_imag']], device=device, dtype=torch.float)
            
            selected_k = model.get_twiddle_selection(stage, pos, a_real, a_imag, b_real, b_imag).item()
            
            key = (d['stage'], d['pos'])
            if key not in twiddle_counts:
                twiddle_counts[key] = {}
            if selected_k not in twiddle_counts[key]:
                twiddle_counts[key][selected_k] = 0
            twiddle_counts[key][selected_k] += 1
    
    # Show selection patterns
    num_stages = int(np.log2(N))
    for stage in range(num_stages):
        print(f"\n  Stage {stage}:")
        for pos in range(N):
            key = (stage, pos)
            if key in twiddle_counts:
                counts = twiddle_counts[key]
                dominant_k = max(counts, key=counts.get)
                correct_k = get_correct_twiddle_index(stage, pos, N)
                match = "✓" if dominant_k == correct_k else "✗"
                purity = counts[dominant_k] / sum(counts.values())
                print(f"    pos {pos}: W_{dominant_k} ({purity:.0%}) [correct: W_{correct_k}] {match}")
    
    # Output accuracy
    print("\n[OUTPUT ACCURACY]")
    
    correct = 0
    with torch.no_grad():
        for d in test_data:
            stage = torch.tensor([d['stage']], device=device)
            pos = torch.tensor([d['pos']], device=device)
            a_real = torch.tensor([d['a_real']], device=device, dtype=torch.float)
            a_imag = torch.tensor([d['a_imag']], device=device, dtype=torch.float)
            b_real = torch.tensor([d['b_real']], device=device, dtype=torch.float)
            b_imag = torch.tensor([d['b_imag']], device=device, dtype=torch.float)
            
            o1r, o1i, o2r, o2i = model(stage, pos, a_real, a_imag, b_real, b_imag, hard=True)
            
            tol = 1e-4
            if (abs(o1r.item() - d['out1_real']) < tol and
                abs(o1i.item() - d['out1_imag']) < tol and
                abs(o2r.item() - d['out2_real']) < tol and
                abs(o2i.item() - d['out2_imag']) < tol):
                correct += 1
    
    output_acc = correct / len(test_data)
    print(f"  Output accuracy: {correct}/{len(test_data)} = {output_acc:.1%}")
    
    # Verdict
    print("\n" + "=" * 70)
    if output_acc >= 0.95:
        print(f"✓ TWIDDLE FFT: SUCCESS!")
        print(f"  Complex rotation with exact microcode")
        print(f"  Routing learned twiddle selection")
    elif output_acc >= 0.50:
        print(f"◐ PARTIAL: {output_acc:.1%}")
    else:
        print(f"✗ NEEDS WORK: {output_acc:.1%}")
    print("=" * 70)
    
    return output_acc


if __name__ == "__main__":
    train_and_test()
