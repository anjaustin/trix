#!/usr/bin/env python3
"""
Pure TriX FFT: N-Scaling Test
=============================

The question: Does the learned structure generalize?

Train on N=8, test on N=16, 32, 64.
If routing truly learned FFT structure, it should scale.

Hypothesis:
- Twiddle patterns repeat fractally
- Same stage logic, more stages
- No retraining needed

CODENAME: ANN WILSON - CRAZY ON YOU
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
    'train_N': 8,
    'test_Ns': [8, 16, 32],
    'value_range': 4,
    'd_model': 64,
    'max_stages': 6,  # Support up to N=64
    'max_N': 64,
    'epochs': 100,
    'lr': 0.01,
    'seed': 1122911624,
}


def compute_twiddle_factors(N):
    """W_k = e^{-2πik/N}"""
    k = torch.arange(N, dtype=torch.float)
    angles = -2 * np.pi * k / N
    real = torch.cos(angles)
    imag = torch.sin(angles)
    return torch.stack([real, imag], dim=1)


def get_correct_twiddle_index(stage, pos, N):
    """Correct twiddle index for DIT FFT."""
    stride = 2 ** stage
    group_size = 2 * stride
    pos_in_group = pos % group_size
    
    if pos_in_group < stride:
        return 0
    else:
        k = (pos_in_group - stride) * (N // group_size)
        return k % N


class ScalableTwiddleRouter(nn.Module):
    """
    Router that can generalize across different N.
    
    Key: encode stage and position RELATIVELY, not absolutely.
    - Stage as fraction of total stages
    - Position as fraction of N
    - Pattern should transfer
    """
    
    def __init__(self, d_model=64, max_stages=6, max_N=64):
        super().__init__()
        
        self.d_model = d_model
        self.max_stages = max_stages
        self.max_N = max_N
        
        # Relative position encoding (continuous, not discrete)
        self.stage_encoder = nn.Sequential(
            nn.Linear(2, d_model // 2),  # (stage_idx, num_stages)
            nn.GELU(),
        )
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, d_model // 2),  # (pos_idx, N)
            nn.GELU(),
        )
        
        # Combined MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        
        # Output: predict twiddle index as fraction of N
        # This makes it scale-invariant
        self.head = nn.Linear(d_model, 1)  # Output: k/N ratio
    
    def forward(self, stage, pos, N, num_stages):
        """
        Args:
            stage: (batch,) stage indices
            pos: (batch,) position indices
            N: scalar, current FFT size
            num_stages: scalar, number of stages
        
        Returns:
            twiddle_ratio: (batch,) predicted k/N (0 to 1)
        """
        batch_size = stage.shape[0]
        device = stage.device
        
        # Normalize inputs
        stage_norm = stage.float() / num_stages
        pos_norm = pos.float() / N
        
        # Encode
        stage_feat = torch.stack([stage_norm, 
                                   torch.full((batch_size,), num_stages / self.max_stages, device=device)], dim=-1)
        pos_feat = torch.stack([pos_norm,
                                 torch.full((batch_size,), N / self.max_N, device=device)], dim=-1)
        
        s_emb = self.stage_encoder(stage_feat)
        p_emb = self.pos_encoder(pos_feat)
        
        x = torch.cat([s_emb, p_emb], dim=-1)
        x = self.mlp(x)
        
        # Output ratio
        ratio = torch.sigmoid(self.head(x).squeeze(-1))  # 0 to 1
        
        return ratio


class ScalableComplexButterfly(nn.Module):
    """Complex butterfly that works for any N."""
    
    def __init__(self, max_N=64):
        super().__init__()
        self.max_N = max_N
    
    def forward(self, a_real, a_imag, b_real, b_imag, twiddle_k, N):
        """
        Args:
            twiddle_k: (batch,) twiddle indices
            N: FFT size (for computing twiddle factors)
        """
        # Compute twiddle factors on the fly
        angles = -2 * np.pi * twiddle_k.float() / N
        W_real = torch.cos(angles)
        W_imag = torch.sin(angles)
        
        # Complex multiply: W * b
        Wb_real = W_real * b_real - W_imag * b_imag
        Wb_imag = W_real * b_imag + W_imag * b_real
        
        # Butterfly
        out1_real = a_real + Wb_real
        out1_imag = a_imag + Wb_imag
        out2_real = a_real - Wb_real
        out2_imag = a_imag - Wb_imag
        
        return out1_real, out1_imag, out2_real, out2_imag


def train_scalable_router():
    """Train router on N=8, test generalization."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    train_N = CONFIG['train_N']
    train_stages = int(np.log2(train_N))
    
    print("\n" + "=" * 70)
    print(f"PURE TRIX FFT - N-SCALING")
    print(f"Train on N={train_N}, test generalization to N=16, 32")
    print("=" * 70)
    
    # Create model
    router = ScalableTwiddleRouter(
        d_model=CONFIG['d_model'],
        max_stages=CONFIG['max_stages'],
        max_N=CONFIG['max_N'],
    ).to(device)
    
    butterfly = ScalableComplexButterfly(max_N=CONFIG['max_N']).to(device)
    
    print(f"Router params: {sum(p.numel() for p in router.parameters())}")
    
    # Generate training data for N=8
    train_data = []
    for stage in range(train_stages):
        for pos in range(train_N):
            k = get_correct_twiddle_index(stage, pos, train_N)
            ratio = k / train_N
            train_data.append({
                'stage': stage,
                'pos': pos,
                'N': train_N,
                'num_stages': train_stages,
                'twiddle_k': k,
                'ratio': ratio,
            })
    
    print(f"Training on {len(train_data)} (stage, pos) pairs for N={train_N}")
    
    optimizer = torch.optim.AdamW(router.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    print("\n[TRAINING]")
    
    for epoch in range(CONFIG['epochs']):
        router.train()
        np.random.shuffle(train_data)
        
        stages = torch.tensor([d['stage'] for d in train_data], device=device)
        positions = torch.tensor([d['pos'] for d in train_data], device=device)
        target_ratios = torch.tensor([d['ratio'] for d in train_data], device=device, dtype=torch.float)
        
        pred_ratios = router(stages, positions, train_N, train_stages)
        
        loss = F.mse_loss(pred_ratios, target_ratios)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            router.eval()
            with torch.no_grad():
                pred = router(stages, positions, train_N, train_stages)
                # Convert ratio to twiddle index
                pred_k = (pred * train_N).round().long() % train_N
                target_k = torch.tensor([d['twiddle_k'] for d in train_data], device=device)
                correct = (pred_k == target_k).sum().item()
                acc = correct / len(train_data)
            
            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.6f}, acc={acc:.1%}")
            
            if acc >= 0.99:
                print("  ✓ Router trained!")
                break
    
    # Test on different N
    print("\n" + "=" * 70)
    print("N-SCALING RESULTS")
    print("=" * 70)
    
    router.eval()
    
    for test_N in CONFIG['test_Ns']:
        test_stages = int(np.log2(test_N))
        
        print(f"\n[N={test_N}] ({test_stages} stages)")
        
        # Test twiddle selection
        correct = 0
        total = 0
        
        with torch.no_grad():
            for stage in range(test_stages):
                stage_correct = 0
                for pos in range(test_N):
                    s = torch.tensor([stage], device=device)
                    p = torch.tensor([pos], device=device)
                    
                    pred_ratio = router(s, p, test_N, test_stages)
                    pred_k = int((pred_ratio.item() * test_N).round()) % test_N
                    
                    correct_k = get_correct_twiddle_index(stage, pos, test_N)
                    
                    if pred_k == correct_k:
                        correct += 1
                        stage_correct += 1
                    total += 1
                
                print(f"    Stage {stage}: {stage_correct}/{test_N} = {stage_correct/test_N:.1%}")
        
        print(f"  Total: {correct}/{total} = {correct/total:.1%}")
        
        # Test full FFT
        print(f"\n  [Full FFT Test for N={test_N}]")
        
        vr = CONFIG['value_range']
        fft_correct = 0
        fft_total = 50
        
        for _ in range(fft_total):
            # Random complex input
            x_real = [np.random.uniform(-vr, vr) for _ in range(test_N)]
            x_imag = [np.random.uniform(-vr, vr) for _ in range(test_N)]
            
            # Run with learned router
            vals_real = torch.tensor(x_real, device=device, dtype=torch.float)
            vals_imag = torch.tensor(x_imag, device=device, dtype=torch.float)
            
            with torch.no_grad():
                for stage in range(test_stages):
                    stride = 2 ** stage
                    new_real = vals_real.clone()
                    new_imag = vals_imag.clone()
                    
                    for i in range(test_N):
                        partner = i ^ stride
                        
                        if i < partner:
                            s = torch.tensor([stage], device=device)
                            p = torch.tensor([i], device=device)
                            
                            pred_ratio = router(s, p, test_N, test_stages)
                            pred_k = int((pred_ratio.item() * test_N).round()) % test_N
                            twiddle_k = torch.tensor([pred_k], device=device)
                            
                            a_r = vals_real[i:i+1]
                            a_i = vals_imag[i:i+1]
                            b_r = vals_real[partner:partner+1]
                            b_i = vals_imag[partner:partner+1]
                            
                            o1r, o1i, o2r, o2i = butterfly(a_r, a_i, b_r, b_i, twiddle_k, test_N)
                            
                            new_real[i] = o1r
                            new_imag[i] = o1i
                            new_real[partner] = o2r
                            new_imag[partner] = o2i
                    
                    vals_real = new_real
                    vals_imag = new_imag
            
            pred_real = vals_real.tolist()
            pred_imag = vals_imag.tolist()
            
            # Reference FFT
            ref_real = list(x_real)
            ref_imag = list(x_imag)
            twiddles = compute_twiddle_factors(test_N).numpy()
            
            for stage in range(test_stages):
                stride = 2 ** stage
                new_r = ref_real.copy()
                new_i = ref_imag.copy()
                
                for i in range(test_N):
                    partner = i ^ stride
                    if i < partner:
                        k = get_correct_twiddle_index(stage, i, test_N)
                        W_r, W_i = twiddles[k]
                        
                        a_r, a_i = ref_real[i], ref_imag[i]
                        b_r, b_i = ref_real[partner], ref_imag[partner]
                        
                        Wb_r = W_r * b_r - W_i * b_i
                        Wb_i = W_r * b_i + W_i * b_r
                        
                        new_r[i] = a_r + Wb_r
                        new_i[i] = a_i + Wb_i
                        new_r[partner] = a_r - Wb_r
                        new_i[partner] = a_i - Wb_i
                
                ref_real = new_r
                ref_imag = new_i
            
            # Compare
            match = all(abs(p - r) < 1e-4 for p, r in zip(pred_real, ref_real))
            match = match and all(abs(p - r) < 1e-4 for p, r in zip(pred_imag, ref_imag))
            
            if match:
                fft_correct += 1
        
        print(f"  FFT accuracy: {fft_correct}/{fft_total} = {fft_correct/fft_total:.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("N-SCALING SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    train_scalable_router()
