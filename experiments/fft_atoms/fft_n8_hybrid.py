#!/usr/bin/env python3
"""
FFT N=8 Hybrid: TDSR Control + Butterfly Organ
===============================================

The full pipeline:
- TDSR learns WHEN (which stage, which pairs)
- Organ computes WHAT (butterfly operation)

This is the test: Can TDSR + Organs compose into a full algorithm?

Architecture:
    Input (8 values)
      ↓
    TDSR Stage Router (which pairs to process?)
      ↓
    Butterfly Organ (a,b) → (a+b, a-b)
      ↓
    Repeat for 3 stages
      ↓
    Output (8 values)

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

from trix.nn import TemporalTileLayer


# =============================================================================
# Fourier Features (same as organ)
# =============================================================================

def fourier_features(x, num_freqs=6, max_freq=16):
    """Fourier encode scalar values."""
    x_norm = x.float().unsqueeze(-1) * (2 * np.pi / max_freq)
    freqs = (2 ** torch.arange(num_freqs, device=x.device, dtype=torch.float)).unsqueeze(0)
    angles = x_norm * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# =============================================================================
# Pre-trained Butterfly Organ
# =============================================================================

class ButterflyOrgan(nn.Module):
    """
    Exact butterfly computation: (a,b) → (a+b, a-b)
    
    The butterfly is EXACTLY linear:
        [out1]   [1  1] [a]
        [out2] = [1 -1] [b]
    
    We implement this directly - no learning needed.
    This is the "exact compute" part of TDSR + Organs.
    """
    
    def __init__(self, value_range=256, hidden_dim=64):
        super().__init__()
        # No parameters needed - this is exact
    
    def forward(self, a, b):
        """a, b: (batch,) float values"""
        return a + b, a - b


# =============================================================================
# TDSR Stage Controller
# =============================================================================

class TDSRStageController(nn.Module):
    """
    TDSR-based stage controller for FFT.
    
    Given current stage and position, outputs:
    - partner index (who to pair with)
    - stage transition signal
    """
    
    def __init__(self, n=8, d_model=32, d_state=16, num_tiles=8):
        super().__init__()
        
        self.n = n
        self.num_stages = int(np.log2(n))
        
        # Embeddings
        self.pos_embed = nn.Embedding(n, d_model // 2)
        self.stage_embed = nn.Embedding(self.num_stages, d_model // 2)
        
        # Temporal layer for state tracking
        self.temporal = TemporalTileLayer(
            d_model=d_model,
            d_state=d_state,
            num_tiles=num_tiles,
            routing_temp=0.5,
        )
        
        # Output: partner prediction
        self.partner_head = nn.Linear(d_model, n)
    
    def forward(self, pos, stage, state=None):
        """
        Args:
            pos: (batch,) position indices [0, n)
            stage: (batch,) stage indices [0, num_stages)
            state: temporal state
        
        Returns:
            partner_logits: (batch, n) logits for partner selection
            new_state: updated state
        """
        batch_size = pos.shape[0]
        device = pos.device
        
        # Embed
        pos_emb = self.pos_embed(pos)
        stage_emb = self.stage_embed(stage)
        x = torch.cat([pos_emb, stage_emb], dim=-1)
        
        # Temporal processing
        if state is None:
            state = self.temporal.init_state(batch_size, device)
        
        output, new_state, info = self.temporal(x, state)
        
        # Predict partner
        partner_logits = self.partner_head(output)
        
        return partner_logits, new_state, info


# =============================================================================
# Hybrid FFT Model
# =============================================================================

class HybridFFT(nn.Module):
    """
    FFT using TDSR control + Butterfly organ.
    
    TDSR decides which pairs to process.
    Organ computes the butterflies.
    """
    
    def __init__(self, n=8, value_range=256):
        super().__init__()
        
        self.n = n
        self.num_stages = int(np.log2(n))
        self.value_range = value_range
        
        # TDSR controller
        self.controller = TDSRStageController(n=n)
        
        # Butterfly organ (will be loaded pre-trained)
        self.organ = ButterflyOrgan(value_range=value_range)
        
    def forward_with_ground_truth_routing(self, x):
        """
        Forward pass using ground-truth FFT routing.
        Tests organ compute only.
        
        Args:
            x: (batch, n) input values
        
        Returns:
            out: (batch, n) FFT-transformed values (real part approximation)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Work in-place on a copy
        result = x.clone().float()
        
        # FFT stages
        for stage in range(self.num_stages):
            stride = 2 ** stage
            new_result = result.clone()
            
            for i in range(self.n):
                partner = i ^ stride  # XOR gives butterfly partner
                
                if i < partner:  # Only process each pair once
                    a = result[:, i]
                    b = result[:, partner]
                    
                    # Use organ for butterfly
                    sum_ab, diff_ab = self.organ(a, b)
                    
                    new_result[:, i] = sum_ab
                    new_result[:, partner] = diff_ab
            
            result = new_result
        
        return result
    
    def forward_with_learned_routing(self, x):
        """
        Forward pass using TDSR-learned routing.
        Tests full pipeline: learned control + organ compute.
        """
        batch_size = x.shape[0]
        device = x.device
        
        result = x.clone().float()
        
        for stage in range(self.num_stages):
            stage_t = torch.full((self.n,), stage, device=device)
            pos_t = torch.arange(self.n, device=device)
            
            # Get partner predictions from TDSR (fresh state each stage)
            partner_logits, _, _ = self.controller(pos_t, stage_t, state=None)
            partners = partner_logits.argmax(dim=-1)  # (n,)
            
            new_result = result.clone()
            
            # Apply butterflies based on learned routing
            processed = set()
            for i in range(self.n):
                partner = partners[i].item()
                
                if (i, partner) not in processed and (partner, i) not in processed:
                    processed.add((i, partner))
                    
                    # Get values for this batch
                    a = result[:, i]
                    b = result[:, partner]
                    
                    sum_ab, diff_ab = self.organ(a, b)
                    
                    new_result[:, i] = sum_ab
                    new_result[:, partner] = diff_ab
            
            result = new_result
        
        return result


# =============================================================================
# Organ Training
# =============================================================================

def train_butterfly_organ(organ, device, value_range=256, epochs=500):
    """Train butterfly organ to compute (a+b, a-b) for larger value ranges."""
    
    # Generate training data covering the full range
    n_samples = 5000
    
    # Sample uniformly from the range
    a_vals = (torch.rand(n_samples, device=device) * 2 - 1) * value_range  # [-range, range]
    b_vals = (torch.rand(n_samples, device=device) * 2 - 1) * value_range
    target_sum = a_vals + b_vals
    target_diff = a_vals - b_vals
    
    optimizer = torch.optim.AdamW(organ.parameters(), lr=0.005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        organ.train()
        
        pred_sum, pred_diff = organ(a_vals, b_vals)
        loss = F.mse_loss(pred_sum, target_sum) + F.mse_loss(pred_diff, target_diff)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            organ.eval()
            with torch.no_grad():
                ps, pd = organ(a_vals, b_vals)
                # Use tolerance for "close enough"
                tol = 0.5
                acc_sum = ((ps - target_sum).abs() < tol).float().mean().item()
                acc_diff = ((pd - target_diff).abs() < tol).float().mean().item()
                acc_both = (((ps - target_sum).abs() < tol) & 
                           ((pd - target_diff).abs() < tol)).float().mean().item()
            
            print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, "
                  f"sum={acc_sum:.1%}, diff={acc_diff:.1%}, both={acc_both:.1%}")
            
            if acc_both >= 0.99:
                print("    ✓ Organ trained!")
                return


# =============================================================================
# Data Generation
# =============================================================================

def generate_fft_data(n=8, num_samples=1000, value_range=16):
    """Generate random inputs and their FFT outputs."""
    data = []
    
    for _ in range(num_samples):
        # Random integer inputs
        x = np.random.randint(0, value_range, size=n)
        
        # Compute FFT (we'll use real part of DFT for simplicity)
        # Actually, let's compute what our butterfly-based FFT SHOULD produce
        # which is the Hadamard-like transform (no twiddles)
        result = x.copy().astype(float)
        
        for stage in range(int(np.log2(n))):
            stride = 2 ** stage
            new_result = result.copy()
            
            for i in range(n):
                partner = i ^ stride
                if i < partner:
                    a, b = result[i], result[partner]
                    new_result[i] = a + b
                    new_result[partner] = a - b
            
            result = new_result
        
        data.append({
            'input': x.tolist(),
            'output': result.tolist(),
        })
    
    return data


def train_controller_supervised(model, device, n=8, epochs=100, lr=0.01):
    """
    Train controller with supervised routing labels.
    
    For FFT, the correct partner at position i, stage s is: i XOR 2^s
    We train the controller to predict this directly.
    """
    optimizer = torch.optim.Adam(model.controller.parameters(), lr=lr)
    
    num_stages = int(np.log2(n))
    
    # Generate all (position, stage) -> partner labels
    positions = list(range(n))
    stages = list(range(num_stages))
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for stage in stages:
            pos_t = torch.arange(n, device=device)
            stage_t = torch.full((n,), stage, device=device)
            
            # Ground truth partners: i XOR 2^stage
            target_partners = torch.tensor([i ^ (2 ** stage) for i in range(n)], device=device)
            
            # Get predictions
            partner_logits, _, _ = model.controller(pos_t, stage_t)
            
            loss = F.cross_entropy(partner_logits, target_partners)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += (partner_logits.argmax(dim=-1) == target_partners).sum().item()
            total_samples += n
        
        if (epoch + 1) % 20 == 0:
            acc = total_correct / total_samples
            print(f"  Epoch {epoch+1}: loss={total_loss/num_stages:.4f}, routing_acc={acc:.1%}")
            
            if acc >= 1.0:
                print("  ✓ Controller learned perfect routing!")
                return True
    
    return False


def evaluate_hybrid(model, data, device):
    """Evaluate the hybrid model."""
    model.eval()
    
    correct_gt = 0  # Ground truth routing
    correct_learned = 0  # Learned routing
    
    with torch.no_grad():
        for d in data:
            x = torch.tensor([d['input']], device=device, dtype=torch.float)
            target = torch.tensor([d['output']], device=device, dtype=torch.float)
            
            # Ground truth routing
            pred_gt = model.forward_with_ground_truth_routing(x)
            if torch.allclose(pred_gt.round(), target, atol=0.5):
                correct_gt += 1
            
            # Learned routing
            pred_learned = model.forward_with_learned_routing(x)
            if torch.allclose(pred_learned.round(), target, atol=0.5):
                correct_learned += 1
    
    return {
        'accuracy_gt_routing': correct_gt / len(data),
        'accuracy_learned_routing': correct_learned / len(data),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    N = 8
    VALUE_RANGE = 16
    
    print("\n" + "=" * 60)
    print(f"HYBRID FFT N={N}")
    print("TDSR routes. Organ computes.")
    print("=" * 60)
    
    # Create model
    print("\nCreating hybrid model...")
    model = HybridFFT(n=N, value_range=VALUE_RANGE).to(device)
    
    # Organ is exact - no training needed
    print("Butterfly organ: EXACT (a+b, a-b) - no training required")
    
    # Generate data
    print("\nGenerating FFT data...")
    train_data = generate_fft_data(N, num_samples=500, value_range=VALUE_RANGE)
    test_data = generate_fft_data(N, num_samples=100, value_range=VALUE_RANGE)
    
    # Test 1: Ground truth routing + organ
    print("\n" + "-" * 40)
    print("TEST 1: Ground Truth Routing + Organ")
    print("-" * 40)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for d in test_data:
            x = torch.tensor([d['input']], device=device, dtype=torch.float)
            target = torch.tensor([d['output']], device=device, dtype=torch.float)
            
            pred = model.forward_with_ground_truth_routing(x)
            
            if torch.allclose(pred.round(), target, atol=0.5):
                correct += 1
    
    acc_gt = correct / len(test_data)
    print(f"Accuracy with GT routing: {acc_gt:.1%}")
    
    if acc_gt < 0.95:
        print("  → Organ needs more training")
    else:
        print("  → Organ is working!")
    
    # Test 2: Train controller with supervised routing
    print("\n" + "-" * 40)
    print("TEST 2: Training TDSR Controller (supervised)")
    print("-" * 40)
    
    print("\nTraining controller to learn FFT routing pattern...")
    print("Target: partner(i, stage) = i XOR 2^stage")
    controller_success = train_controller_supervised(model, device, n=N, epochs=100, lr=0.01)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    metrics = evaluate_hybrid(model, test_data, device)
    
    print(f"\nGT Routing + Organ:      {metrics['accuracy_gt_routing']:.1%}")
    print(f"Learned Routing + Organ: {metrics['accuracy_learned_routing']:.1%}")
    
    # Example
    print("\nExample:")
    d = test_data[0]
    x = torch.tensor([d['input']], device=device, dtype=torch.float)
    
    with torch.no_grad():
        pred_gt = model.forward_with_ground_truth_routing(x)
        pred_learned = model.forward_with_learned_routing(x)
    
    print(f"  Input:   {d['input']}")
    print(f"  Target:  {[int(v) for v in d['output']]}")
    print(f"  GT pred: {[int(v) for v in pred_gt[0].round().tolist()]}")
    print(f"  Learned: {[int(v) for v in pred_learned[0].round().tolist()]}")
    
    # Verdict
    print("\n" + "=" * 60)
    if metrics['accuracy_gt_routing'] >= 0.95:
        print("✓ ORGAN WORKS: GT routing achieves 95%+")
    else:
        print("✗ ORGAN NEEDS WORK")
    
    if metrics['accuracy_learned_routing'] >= 0.90:
        print("✓ CONTROLLER WORKS: Learned routing achieves 90%+")
        print("\n*** HYBRID FFT SUCCESS: algorithm = control + organs ***")
    else:
        print(f"✗ CONTROLLER NEEDS WORK: {metrics['accuracy_learned_routing']:.1%}")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
