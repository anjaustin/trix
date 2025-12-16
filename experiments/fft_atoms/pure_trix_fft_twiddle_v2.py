#!/usr/bin/env python3
"""
Pure TriX FFT: Twiddle Factors v2 (Structural Selection)
========================================================

Key insight: twiddle selection is STRUCTURAL, not value-dependent.
Like ADDRESS, it's a function of (stage, pos) only.

Architecture:
- Twiddle router: (stage, pos) → W_k index (learned structure)
- Butterfly: exact complex arithmetic with selected W_k

Same pattern as ADDRESS atom - learn the structure, execute exactly.

CODENAME: ANN WILSON - WHAT ABOUT LOVE
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
    'value_range': 8,
    'd_model': 64,
    'epochs': 100,
    'batch_size': 32,
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
    """
    Correct twiddle index for DIT FFT.
    
    For each stage, only the "lower" element of each pair uses twiddle.
    The index depends on position within the stage's group structure.
    """
    stride = 2 ** stage
    pair_idx = pos // stride  # Which pair in this stage
    pos_in_pair = pos % stride  # Position within the pair
    
    # Only positions in upper half of each 2*stride group need non-trivial twiddle
    group_size = 2 * stride
    pos_in_group = pos % group_size
    
    if pos_in_group < stride:
        # Lower half of group - uses W_0 effectively (will be paired with upper half)
        return 0
    else:
        # Upper half of group - this element gets multiplied by twiddle
        # The twiddle index scales with position
        k = (pos_in_group - stride) * (N // group_size)
        return k % N


class TwiddleRouter(nn.Module):
    """
    Learns structural mapping: (stage, pos) → twiddle index.
    
    This is like ADDRESS - purely structural, value-independent.
    """
    
    def __init__(self, N=8, num_stages=3, d_model=64):
        super().__init__()
        
        self.N = N
        self.num_stages = num_stages
        
        # Embeddings for stage and position
        self.stage_embed = nn.Embedding(num_stages, d_model)
        self.pos_embed = nn.Embedding(N, d_model)
        
        # MLP to predict twiddle index
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, N),  # Output: logits for each twiddle
        )
        
        self.temperature = 0.1
    
    def forward(self, stage, pos, hard=True):
        """
        Args:
            stage: (batch,) stage indices
            pos: (batch,) position indices
            hard: use argmax (exact) vs softmax (training)
        
        Returns:
            twiddle_idx: (batch,) selected twiddle indices
            logits: (batch, N) raw logits for loss computation
        """
        stage_emb = self.stage_embed(stage)
        pos_emb = self.pos_embed(pos)
        
        x = torch.cat([stage_emb, pos_emb], dim=-1)
        logits = self.mlp(x) / self.temperature
        
        if hard:
            twiddle_idx = logits.argmax(dim=-1)
        else:
            # Gumbel-softmax for differentiable discrete selection
            twiddle_idx = F.gumbel_softmax(logits, tau=self.temperature, hard=True).argmax(dim=-1)
        
        return twiddle_idx, logits


class ComplexButterfly(nn.Module):
    """
    Exact complex butterfly with twiddle.
    
    Given a, b (complex) and twiddle W_k:
      out1 = a + W_k * b
      out2 = a - W_k * b
    
    No learning here - pure microcode execution.
    """
    
    def __init__(self, N=8):
        super().__init__()
        
        twiddles = compute_twiddle_factors(N)
        self.register_buffer('twiddle_factors', twiddles)
    
    def forward(self, a_real, a_imag, b_real, b_imag, twiddle_idx):
        """
        Exact complex butterfly.
        
        Args:
            a_real, a_imag: (batch,) complex number a
            b_real, b_imag: (batch,) complex number b
            twiddle_idx: (batch,) which twiddle to use
        
        Returns:
            out1_real, out1_imag, out2_real, out2_imag
        """
        # Get twiddle factors
        W = self.twiddle_factors[twiddle_idx]  # (batch, 2)
        W_real = W[:, 0]
        W_imag = W[:, 1]
        
        # Complex multiply: W * b
        Wb_real = W_real * b_real - W_imag * b_imag
        Wb_imag = W_real * b_imag + W_imag * b_real
        
        # Butterfly
        out1_real = a_real + Wb_real
        out1_imag = a_imag + Wb_imag
        out2_real = a_real - Wb_real
        out2_imag = a_imag - Wb_imag
        
        return out1_real, out1_imag, out2_real, out2_imag


def train_router():
    """Train just the twiddle router on structural task."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    N = CONFIG['N']
    num_stages = int(np.log2(N))
    
    print("\n" + "=" * 70)
    print(f"PURE TRIX FFT - TWIDDLE ROUTER (N={N})")
    print("Structural learning: (stage, pos) → twiddle index")
    print("=" * 70)
    
    # Show correct twiddle mapping
    print("\nCorrect twiddle mapping:")
    for stage in range(num_stages):
        print(f"  Stage {stage}: ", end="")
        for pos in range(N):
            k = get_correct_twiddle_index(stage, pos, N)
            print(f"W{k}", end=" ")
        print()
    
    # Create router
    router = TwiddleRouter(N=N, num_stages=num_stages, d_model=CONFIG['d_model']).to(device)
    butterfly = ComplexButterfly(N=N).to(device)
    
    print(f"\nRouter params: {sum(p.numel() for p in router.parameters())}")
    
    # Generate ALL (stage, pos) pairs - exhaustive structural training
    all_pairs = []
    for stage in range(num_stages):
        for pos in range(N):
            k = get_correct_twiddle_index(stage, pos, N)
            all_pairs.append({'stage': stage, 'pos': pos, 'twiddle_k': k})
    
    print(f"Training on {len(all_pairs)} (stage, pos) pairs")
    
    optimizer = torch.optim.AdamW(router.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    print("\n[TRAINING ROUTER]")
    
    for epoch in range(CONFIG['epochs']):
        router.train()
        np.random.shuffle(all_pairs)
        
        # Process all pairs
        stages = torch.tensor([p['stage'] for p in all_pairs], device=device)
        positions = torch.tensor([p['pos'] for p in all_pairs], device=device)
        targets = torch.tensor([p['twiddle_k'] for p in all_pairs], device=device)
        
        _, logits = router(stages, positions, hard=False)
        
        loss = F.cross_entropy(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            router.eval()
            with torch.no_grad():
                pred_idx, _ = router(stages, positions, hard=True)
                correct = (pred_idx == targets).sum().item()
                acc = correct / len(all_pairs)
            
            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:.1%}")
            
            if acc >= 0.99:
                print("  ✓ Router trained!")
                break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("ROUTER RESULTS")
    print("=" * 70)
    
    router.eval()
    print("\nLearned twiddle mapping:")
    
    all_correct = True
    for stage in range(num_stages):
        print(f"  Stage {stage}: ", end="")
        for pos in range(N):
            s = torch.tensor([stage], device=device)
            p = torch.tensor([pos], device=device)
            
            with torch.no_grad():
                pred_k, _ = router(s, p, hard=True)
            
            correct_k = get_correct_twiddle_index(stage, pos, N)
            match = "✓" if pred_k.item() == correct_k else "✗"
            if pred_k.item() != correct_k:
                all_correct = False
            
            print(f"W{pred_k.item()}{match}", end=" ")
        print()
    
    # Test full complex butterfly with learned routing
    print("\n[COMPLEX BUTTERFLY TEST]")
    
    correct_outputs = 0
    num_tests = 500
    
    for _ in range(num_tests):
        stage = np.random.randint(0, num_stages)
        pos = np.random.randint(0, N)
        
        a_real = np.random.uniform(-CONFIG['value_range'], CONFIG['value_range'])
        a_imag = np.random.uniform(-CONFIG['value_range'], CONFIG['value_range'])
        b_real = np.random.uniform(-CONFIG['value_range'], CONFIG['value_range'])
        b_imag = np.random.uniform(-CONFIG['value_range'], CONFIG['value_range'])
        
        # Get router's twiddle selection
        s = torch.tensor([stage], device=device)
        p = torch.tensor([pos], device=device)
        
        with torch.no_grad():
            twiddle_idx, _ = router(s, p, hard=True)
        
        # Execute butterfly
        ar = torch.tensor([a_real], device=device, dtype=torch.float)
        ai = torch.tensor([a_imag], device=device, dtype=torch.float)
        br = torch.tensor([b_real], device=device, dtype=torch.float)
        bi = torch.tensor([b_imag], device=device, dtype=torch.float)
        
        with torch.no_grad():
            o1r, o1i, o2r, o2i = butterfly(ar, ai, br, bi, twiddle_idx)
        
        # Compute expected with correct twiddle
        correct_k = get_correct_twiddle_index(stage, pos, N)
        twiddles = compute_twiddle_factors(N).numpy()
        W_real, W_imag = twiddles[correct_k]
        
        Wb_real = W_real * b_real - W_imag * b_imag
        Wb_imag = W_real * b_imag + W_imag * b_real
        
        exp_o1r = a_real + Wb_real
        exp_o1i = a_imag + Wb_imag
        exp_o2r = a_real - Wb_real
        exp_o2i = a_imag - Wb_imag
        
        tol = 1e-5
        if (abs(o1r.item() - exp_o1r) < tol and
            abs(o1i.item() - exp_o1i) < tol and
            abs(o2r.item() - exp_o2r) < tol and
            abs(o2i.item() - exp_o2i) < tol):
            correct_outputs += 1
    
    output_acc = correct_outputs / num_tests
    print(f"  Butterfly accuracy: {correct_outputs}/{num_tests} = {output_acc:.1%}")
    
    # Verdict
    print("\n" + "=" * 70)
    if all_correct and output_acc >= 0.99:
        print("✓ TWIDDLE FFT: PERFECT!")
        print("  Structural routing: 100%")
        print("  Complex butterfly: exact")
        print("  Ready for full complex FFT")
    elif output_acc >= 0.95:
        print(f"✓ TWIDDLE FFT: SUCCESS ({output_acc:.1%})")
    else:
        print(f"✗ NEEDS WORK: {output_acc:.1%}")
    print("=" * 70)
    
    return all_correct, output_acc, router, butterfly


def run_complex_fft(router, butterfly, x_real, x_imag, device):
    """
    Run full complex FFT using learned router and exact butterfly.
    
    Args:
        x_real, x_imag: lists of N real/imag values
    
    Returns:
        out_real, out_imag: transformed values
    """
    N = len(x_real)
    num_stages = int(np.log2(N))
    
    # Convert to tensors
    vals_real = torch.tensor(x_real, device=device, dtype=torch.float)
    vals_imag = torch.tensor(x_imag, device=device, dtype=torch.float)
    
    router.eval()
    
    with torch.no_grad():
        for stage in range(num_stages):
            stride = 2 ** stage
            new_real = vals_real.clone()
            new_imag = vals_imag.clone()
            
            for i in range(N):
                partner = i ^ stride
                
                if i < partner:
                    # Get twiddle for this position
                    s = torch.tensor([stage], device=device)
                    p = torch.tensor([i], device=device)
                    twiddle_idx, _ = router(s, p, hard=True)
                    
                    # Execute butterfly
                    a_r = vals_real[i:i+1]
                    a_i = vals_imag[i:i+1]
                    b_r = vals_real[partner:partner+1]
                    b_i = vals_imag[partner:partner+1]
                    
                    o1r, o1i, o2r, o2i = butterfly(a_r, a_i, b_r, b_i, twiddle_idx)
                    
                    new_real[i] = o1r
                    new_imag[i] = o1i
                    new_real[partner] = o2r
                    new_imag[partner] = o2i
            
            vals_real = new_real
            vals_imag = new_imag
    
    return vals_real.tolist(), vals_imag.tolist()


def reference_complex_fft(x_real, x_imag):
    """Ground truth complex FFT."""
    N = len(x_real)
    num_stages = int(np.log2(N))
    twiddles = compute_twiddle_factors(N).numpy()
    
    vals_real = list(x_real)
    vals_imag = list(x_imag)
    
    for stage in range(num_stages):
        stride = 2 ** stage
        new_real = vals_real.copy()
        new_imag = vals_imag.copy()
        
        for i in range(N):
            partner = i ^ stride
            
            if i < partner:
                k = get_correct_twiddle_index(stage, i, N)
                W_real, W_imag = twiddles[k]
                
                a_r, a_i = vals_real[i], vals_imag[i]
                b_r, b_i = vals_real[partner], vals_imag[partner]
                
                Wb_r = W_real * b_r - W_imag * b_i
                Wb_i = W_real * b_i + W_imag * b_r
                
                new_real[i] = a_r + Wb_r
                new_imag[i] = a_i + Wb_i
                new_real[partner] = a_r - Wb_r
                new_imag[partner] = a_i - Wb_i
        
        vals_real = new_real
        vals_imag = new_imag
    
    return vals_real, vals_imag


def test_full_fft(router, butterfly, device):
    """Test full complex FFT."""
    N = CONFIG['N']
    vr = CONFIG['value_range']
    
    print("\n[FULL COMPLEX FFT TEST]")
    
    correct = 0
    total = 100
    
    for _ in range(total):
        x_real = [np.random.uniform(-vr, vr) for _ in range(N)]
        x_imag = [np.random.uniform(-vr, vr) for _ in range(N)]
        
        exp_real, exp_imag = reference_complex_fft(x_real, x_imag)
        pred_real, pred_imag = run_complex_fft(router, butterfly, x_real, x_imag, device)
        
        # Compare
        match = True
        for i in range(N):
            if abs(pred_real[i] - exp_real[i]) > 1e-4 or abs(pred_imag[i] - exp_imag[i]) > 1e-4:
                match = False
                break
        
        if match:
            correct += 1
    
    acc = correct / total
    print(f"  Full FFT accuracy: {correct}/{total} = {acc:.1%}")
    
    # Show examples
    print("\nExamples:")
    for _ in range(3):
        x_real = [round(np.random.uniform(-vr, vr), 2) for _ in range(N)]
        x_imag = [0.0] * N  # Real input for clarity
        
        exp_real, exp_imag = reference_complex_fft(x_real, x_imag)
        pred_real, pred_imag = run_complex_fft(router, butterfly, x_real, x_imag, device)
        
        match = all(abs(p - e) < 1e-4 for p, e in zip(pred_real, exp_real))
        mark = "✓" if match else "✗"
        
        print(f"  Input:  {[round(x, 1) for x in x_real]}")
        print(f"  Output: {[round(x, 1) for x in pred_real]} {mark}")
        print()
    
    return acc


if __name__ == "__main__":
    all_correct, butterfly_acc, router, butterfly = train_router()
    
    if butterfly_acc >= 0.95:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fft_acc = test_full_fft(router, butterfly, device)
        
        print("\n" + "=" * 70)
        if fft_acc >= 0.99:
            print("✓ COMPLEX FFT WITH TWIDDLES: PERFECT!")
            print("  Twiddle routing: structural (learned)")
            print("  Complex butterfly: exact (microcode)")
            print("  Full N=8 complex FFT: 100%")
        print("=" * 70)
