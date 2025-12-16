#!/usr/bin/env python3
"""
Pure TriX FFT N=8: Complete Algorithm
=====================================

Full FFT using pure TriX:
- Tiles compute ADD and SUB
- Routing selects operations
- Staged composition produces FFT

No external organs. No hybrid compute.
Everything is TriX.

CODENAME: ANN WILSON - HEART COMPLETE
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
# Configuration
# =============================================================================

CONFIG = {
    'N': 8,                 # FFT size
    'value_range': 16,      # Input range 0-15
    'd_model': 64,
    'num_tiles': 4,
    'num_freqs': 6,
    'epochs': 300,
    'batch_size': 32,
    'lr': 0.003,
    'seed': 1122911624,
}


# =============================================================================
# Pure TriX FFT Layer
# =============================================================================

class PureTriXFFTLayer(nn.Module):
    """
    Single FFT stage using pure TriX.
    
    For each position, routes to ADD or SUB tile
    based on learned control.
    """
    
    def __init__(self, d_model=64, num_tiles=4):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        
        # Router: decides ADD or SUB for each position
        self.router = nn.Sequential(
            nn.Linear(d_model * 2 + 8, d_model),  # pair input + position encoding
            nn.GELU(),
            nn.Linear(d_model, num_tiles),
        )
        
        # Tile networks (2 will specialize to ADD, others to SUB)
        self.tile_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_tiles)
        ])
        
        self.temperature = 0.5
    
    def forward(self, a_vec, b_vec, pos_encoding):
        """
        Butterfly operation on vector pair.
        
        Args:
            a_vec: (batch, d_model) first element
            b_vec: (batch, d_model) second element
            pos_encoding: (batch, 8) position/stage info
        
        Returns:
            out1: (batch, d_model) - should be a+b
            out2: (batch, d_model) - should be a-b
        """
        batch_size = a_vec.shape[0]
        
        # Concatenate pair
        pair = torch.cat([a_vec, b_vec], dim=-1)  # (batch, d_model*2)
        
        # Route for first output (ADD)
        route_input1 = torch.cat([pair, pos_encoding], dim=-1)
        route_logits1 = self.router(route_input1) / self.temperature
        tile_idx1 = route_logits1.argmax(dim=-1)
        
        # Compute first output
        out1_list = []
        for i in range(batch_size):
            t = tile_idx1[i].item()
            out = self.tile_nets[t](pair[i:i+1])
            out1_list.append(out)
        out1 = torch.cat(out1_list, dim=0)
        
        # Route for second output (SUB) - flip the pair order as hint
        pair_flip = torch.cat([b_vec, a_vec], dim=-1)
        route_input2 = torch.cat([pair_flip, pos_encoding], dim=-1)
        route_logits2 = self.router(route_input2) / self.temperature
        tile_idx2 = route_logits2.argmax(dim=-1)
        
        # Compute second output
        out2_list = []
        for i in range(batch_size):
            t = tile_idx2[i].item()
            out = self.tile_nets[t](pair[i:i+1])
            out2_list.append(out)
        out2 = torch.cat(out2_list, dim=0)
        
        return out1, out2


class PureTriXFFTN8(nn.Module):
    """
    Complete N=8 FFT using pure TriX.
    
    Architecture:
    1. Embed 8 input values
    2. 3 stages of butterfly operations
    3. Decode 8 output values
    """
    
    def __init__(self, N=8, value_range=16, d_model=64, num_tiles=4, num_freqs=6):
        super().__init__()
        
        self.N = N
        self.num_stages = int(np.log2(N))
        self.value_range = value_range
        self.d_model = d_model
        self.num_freqs = num_freqs
        
        # Input embedding: value -> d_model
        fourier_dim = 2 * num_freqs
        self.value_embed = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Position/stage encoding
        self.pos_embed = nn.Embedding(N, 4)
        self.stage_embed = nn.Embedding(self.num_stages, 4)
        
        # One FFT layer per stage
        self.stages = nn.ModuleList([
            PureTriXFFTLayer(d_model=d_model, num_tiles=num_tiles)
            for _ in range(self.num_stages)
        ])
        
        # Output decoder: d_model -> value
        # Output range for N=8, inputs 0-15: roughly -120 to 120
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
    
    def _fourier_features(self, x):
        """Fourier encode values."""
        x_norm = x.float().unsqueeze(-1) * (2 * np.pi / self.value_range)
        freqs = (2 ** torch.arange(self.num_freqs, device=x.device, dtype=torch.float)).unsqueeze(0)
        angles = x_norm * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, N) input values
        
        Returns:
            output: (batch, N) FFT output values
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Embed each value
        # x: (batch, N) -> embeddings: (batch, N, d_model)
        x_flat = x.view(-1)  # (batch * N)
        x_feat = self._fourier_features(x_flat)  # (batch * N, fourier_dim)
        embeddings = self.value_embed(x_feat)  # (batch * N, d_model)
        embeddings = embeddings.view(batch_size, self.N, self.d_model)
        
        # Process through stages
        values = embeddings  # (batch, N, d_model)
        
        for stage_idx, stage_layer in enumerate(self.stages):
            stride = 2 ** stage_idx
            new_values = values.clone()
            
            # Stage encoding
            stage_enc = self.stage_embed(torch.tensor([stage_idx], device=device))
            stage_enc = stage_enc.expand(batch_size, -1)  # (batch, 4)
            
            # Process each butterfly pair
            for i in range(self.N):
                partner = i ^ stride
                
                if i < partner:
                    # Position encoding
                    pos_enc_i = self.pos_embed(torch.tensor([i], device=device)).expand(batch_size, -1)
                    pos_enc = torch.cat([pos_enc_i, stage_enc], dim=-1)  # (batch, 8)
                    
                    # Get values for this pair
                    a_vec = values[:, i, :]   # (batch, d_model)
                    b_vec = values[:, partner, :]
                    
                    # Butterfly via TriX routing
                    sum_vec, diff_vec = stage_layer(a_vec, b_vec, pos_enc)
                    
                    new_values[:, i, :] = sum_vec
                    new_values[:, partner, :] = diff_vec
            
            values = new_values
        
        # Decode to output values
        output = self.output_head(values).squeeze(-1)  # (batch, N)
        
        return output


# =============================================================================
# Data Generation
# =============================================================================

def compute_fft_reference(x):
    """
    Compute reference FFT output (Hadamard-like, no twiddles).
    
    This is the target our pure TriX should learn.
    """
    N = len(x)
    result = x.copy().astype(float)
    
    num_stages = int(np.log2(N))
    
    for stage in range(num_stages):
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


def generate_fft_data(N=8, value_range=16, num_samples=1000):
    """Generate FFT training data."""
    data = []
    
    for _ in range(num_samples):
        x = np.random.randint(0, value_range, size=N)
        y = compute_fft_reference(x)
        
        data.append({
            'input': x.tolist(),
            'output': y.tolist(),
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
        
        x = torch.tensor([d['input'] for d in batch], device=device, dtype=torch.float)
        target = torch.tensor([d['output'] for d in batch], device=device, dtype=torch.float)
        
        output = model(x)
        
        loss = F.mse_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (len(data) // batch_size + 1)


def evaluate(model, data, device):
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for d in data:
            x = torch.tensor([d['input']], device=device, dtype=torch.float)
            target = torch.tensor([d['output']], device=device, dtype=torch.float)
            
            output = model(x)
            
            # Check if rounded output matches target
            pred = output.round()
            if torch.allclose(pred, target, atol=0.5):
                correct += 1
            total += 1
    
    return correct / total


# =============================================================================
# Main
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    N = CONFIG['N']
    
    print("\n" + "=" * 70)
    print(f"PURE TRIX FFT N={N}")
    print("Complete algorithm: tiles compute, routing controls")
    print("=" * 70)
    
    # Generate data
    train_data = generate_fft_data(N, CONFIG['value_range'], num_samples=2000)
    test_data = generate_fft_data(N, CONFIG['value_range'], num_samples=500)
    
    print(f"\nData: {len(train_data)} train, {len(test_data)} test")
    print(f"Stages: {int(np.log2(N))}")
    
    # Show example
    print(f"\nExample:")
    print(f"  Input:  {train_data[0]['input']}")
    print(f"  Output: {train_data[0]['output']}")
    
    # Create model
    model = PureTriXFFTN8(
        N=N,
        value_range=CONFIG['value_range'],
        d_model=CONFIG['d_model'],
        num_tiles=CONFIG['num_tiles'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    # Training
    print("\n[TRAINING]")
    
    best_acc = 0
    for epoch in range(CONFIG['epochs']):
        loss = train_epoch(model, train_data, optimizer, device)
        scheduler.step()
        
        if (epoch + 1) % 30 == 0:
            train_acc = evaluate(model, train_data[:200], device)
            test_acc = evaluate(model, test_data[:200], device)
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, train={train_acc:.1%}, test={test_acc:.1%}")
            
            if test_acc >= 0.95:
                print(f"  ✓ 95%+ test accuracy achieved!")
                break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    final_train_acc = evaluate(model, train_data, device)
    final_test_acc = evaluate(model, test_data, device)
    
    print(f"\nTrain accuracy: {final_train_acc:.1%}")
    print(f"Test accuracy:  {final_test_acc:.1%}")
    
    # Show predictions
    print("\n[EXAMPLES]")
    model.eval()
    with torch.no_grad():
        for i in range(3):
            d = test_data[i]
            x = torch.tensor([d['input']], device=device, dtype=torch.float)
            output = model(x).round()[0].tolist()
            
            match = "✓" if output == d['output'] else "✗"
            print(f"  Input:  {d['input']}")
            print(f"  Target: {[int(v) for v in d['output']]}")
            print(f"  Pred:   {[int(v) for v in output]} {match}")
            print()
    
    # Verdict
    print("=" * 70)
    if final_test_acc >= 0.90:
        print(f"✓ PURE TRIX FFT N={N}: SUCCESS")
        print(f"  Test accuracy: {final_test_acc:.1%}")
        print("  Full FFT learned with pure TriX")
        print("  No organs. No hybrid. Pure routing + tiles.")
    elif final_test_acc >= 0.70:
        print(f"◐ PARTIAL SUCCESS: {final_test_acc:.1%}")
    else:
        print(f"✗ NEEDS MORE WORK: {final_test_acc:.1%}")
    print("=" * 70)
    
    # Save
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'config': CONFIG,
        'train_accuracy': final_train_acc,
        'test_accuracy': final_test_acc,
        'best_test_accuracy': best_acc,
    }
    
    with open(output_dir / f'pure_trix_fft_n8_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_dir / f'pure_trix_fft_n8_{timestamp}.json'}")
    
    return final_test_acc


if __name__ == "__main__":
    main()
