#!/usr/bin/env python3
"""
Pure TriX FFT: Staged Composition
=================================

The modular approach:
- Stage as input (gives routing meaning across FFT)
- Position as input (which pair)
- Butterfly tiles (ADD/SUB specialists)

Same pattern that made micro-ops work:
  op type → routing meaning → tile specialization

Now:
  stage + position → routing meaning → correct butterfly

CODENAME: ANN WILSON - BARRACUDA
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
    'N': 8,
    'value_range': 16,
    'd_model': 96,
    'num_tiles': 8,
    'num_freqs': 6,
    'epochs': 200,
    'batch_size': 64,
    'lr': 0.004,
    'seed': 1122911624,
}


class PureTriXStagedFFT(nn.Module):
    """
    Staged FFT using pure TriX.
    
    Key insight: (stage, position, a, b) → (a+b, a-b)
    Stage and position give routing meaning.
    Tiles specialize to different regimes.
    """
    
    def __init__(self, N=8, value_range=16, d_model=96, num_tiles=8, num_freqs=6):
        super().__init__()
        
        self.N = N
        self.num_stages = int(np.log2(N))
        self.value_range = value_range
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_freqs = num_freqs
        
        # Stage embedding
        self.stage_embed = nn.Embedding(self.num_stages, d_model // 4)
        
        # Position embedding
        self.pos_embed = nn.Embedding(N, d_model // 4)
        
        # Value encoding (Fourier features)
        fourier_dim = 2 * num_freqs
        
        # Input projection
        # stage_embed + pos_embed + a_fourier + b_fourier
        input_dim = d_model // 4 + d_model // 4 + 2 * fourier_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Routers for sum and diff paths
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
        
        # Tile networks
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
        
        # Tracking: tile usage by (stage, output_type)
        self.register_buffer('tile_stage_sum', torch.zeros(num_tiles, self.num_stages))
        self.register_buffer('tile_stage_diff', torch.zeros(num_tiles, self.num_stages))
    
    def _fourier_features(self, x):
        """Fourier encode values."""
        # Handle potentially large values from FFT intermediate results
        x_norm = x.float().unsqueeze(-1) * (2 * np.pi / 256)  # Larger range
        freqs = (2 ** torch.arange(self.num_freqs, device=x.device, dtype=torch.float)).unsqueeze(0)
        angles = x_norm * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, stage, pos, a, b, track=True):
        """
        Single butterfly operation with stage/position context.
        
        Args:
            stage: (batch,) stage index [0, num_stages)
            pos: (batch,) position index [0, N)
            a, b: (batch,) values to butterfly
        
        Returns:
            sum_pred: (batch,) predicted a+b
            diff_pred: (batch,) predicted a-b
        """
        batch_size = stage.shape[0]
        
        # Encode context
        stage_emb = self.stage_embed(stage)
        pos_emb = self.pos_embed(pos)
        
        # Encode values
        a_feat = self._fourier_features(a)
        b_feat = self._fourier_features(b)
        
        # Combine all inputs
        x = torch.cat([stage_emb, pos_emb, a_feat, b_feat], dim=-1)
        x = self.input_proj(x)
        
        # Route for sum
        sum_logits = self.router_sum(x) / self.temperature
        sum_tile = sum_logits.argmax(dim=-1)
        
        # Route for diff
        diff_logits = self.router_diff(x) / self.temperature
        diff_tile = diff_logits.argmax(dim=-1)
        
        # Compute through tiles
        sum_outputs = []
        diff_outputs = []
        
        for i in range(batch_size):
            st = sum_tile[i].item()
            dt = diff_tile[i].item()
            
            sum_out = self.tile_nets[st](x[i:i+1])
            diff_out = self.tile_nets[dt](x[i:i+1])
            
            sum_outputs.append(sum_out)
            diff_outputs.append(diff_out)
            
            # Track
            if track and self.training:
                s = stage[i].item()
                self.tile_stage_sum[st, s] += 1
                self.tile_stage_diff[dt, s] += 1
        
        sum_hidden = torch.cat(sum_outputs, dim=0)
        diff_hidden = torch.cat(diff_outputs, dim=0)
        
        sum_pred = self.sum_head(sum_hidden).squeeze(-1)
        diff_pred = self.diff_head(diff_hidden).squeeze(-1)
        
        return sum_pred, diff_pred
    
    def get_tile_analysis(self):
        """Analyze tile specialization by stage."""
        sum_counts = self.tile_stage_sum.cpu().numpy()
        diff_counts = self.tile_stage_diff.cpu().numpy()
        
        analysis = {'sum_tiles': {}, 'diff_tiles': {}}
        
        for t in range(self.num_tiles):
            sum_total = sum_counts[t].sum()
            diff_total = diff_counts[t].sum()
            
            if sum_total > 0:
                dominant_stage = sum_counts[t].argmax()
                purity = sum_counts[t, dominant_stage] / sum_total
                analysis['sum_tiles'][t] = {
                    'dominant_stage': int(dominant_stage),
                    'purity': float(purity),
                    'total': int(sum_total),
                }
            
            if diff_total > 0:
                dominant_stage = diff_counts[t].argmax()
                purity = diff_counts[t, dominant_stage] / diff_total
                analysis['diff_tiles'][t] = {
                    'dominant_stage': int(dominant_stage),
                    'purity': float(purity),
                    'total': int(diff_total),
                }
        
        return analysis
    
    def reset_tracking(self):
        self.tile_stage_sum.zero_()
        self.tile_stage_diff.zero_()


def generate_staged_data(N=8, value_range=16, num_samples=2000):
    """
    Generate training data: (stage, pos, a, b) → (a+b, a-b)
    
    For each sample, pick a random stage and position,
    then random values, compute the butterfly.
    """
    num_stages = int(np.log2(N))
    data = []
    
    for _ in range(num_samples):
        stage = np.random.randint(0, num_stages)
        pos = np.random.randint(0, N)
        
        # Values can be from original range or intermediate (larger)
        # For training, use a range that covers FFT intermediates
        max_val = value_range * (2 ** stage)  # Values grow with stage
        a = np.random.randint(-max_val, max_val + 1)
        b = np.random.randint(-max_val, max_val + 1)
        
        data.append({
            'stage': stage,
            'pos': pos,
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
        
        stage = torch.tensor([d['stage'] for d in batch], device=device)
        pos = torch.tensor([d['pos'] for d in batch], device=device)
        a = torch.tensor([d['a'] for d in batch], device=device, dtype=torch.float)
        b = torch.tensor([d['b'] for d in batch], device=device, dtype=torch.float)
        target_sum = torch.tensor([d['sum'] for d in batch], device=device, dtype=torch.float)
        target_diff = torch.tensor([d['diff'] for d in batch], device=device, dtype=torch.float)
        
        pred_sum, pred_diff = model(stage, pos, a, b)
        
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
    per_stage_correct = {s: {'sum': 0, 'diff': 0, 'both': 0, 'total': 0} 
                         for s in range(int(np.log2(CONFIG['N'])))}
    
    with torch.no_grad():
        for d in data:
            stage = torch.tensor([d['stage']], device=device)
            pos = torch.tensor([d['pos']], device=device)
            a = torch.tensor([d['a']], device=device, dtype=torch.float)
            b = torch.tensor([d['b']], device=device, dtype=torch.float)
            
            pred_sum, pred_diff = model(stage, pos, a, b, track=False)
            
            ps = pred_sum.round().item()
            pd = pred_diff.round().item()
            
            sum_ok = (ps == d['sum'])
            diff_ok = (pd == d['diff'])
            both_ok = sum_ok and diff_ok
            
            correct_sum += sum_ok
            correct_diff += diff_ok
            correct_both += both_ok
            
            s = d['stage']
            per_stage_correct[s]['sum'] += sum_ok
            per_stage_correct[s]['diff'] += diff_ok
            per_stage_correct[s]['both'] += both_ok
            per_stage_correct[s]['total'] += 1
    
    n = len(data)
    
    # Compute per-stage accuracy
    per_stage_acc = {}
    for s, counts in per_stage_correct.items():
        if counts['total'] > 0:
            per_stage_acc[s] = counts['both'] / counts['total']
    
    return {
        'sum': correct_sum / n,
        'diff': correct_diff / n,
        'both': correct_both / n,
        'per_stage': per_stage_acc,
    }


def run_full_fft(model, x, device):
    """
    Run complete FFT using the trained staged model.
    
    Args:
        x: list of N input values
    
    Returns:
        list of N output values
    """
    model.eval()
    N = len(x)
    num_stages = int(np.log2(N))
    
    values = [float(v) for v in x]
    
    with torch.no_grad():
        for stage in range(num_stages):
            stride = 2 ** stage
            new_values = values.copy()
            
            for i in range(N):
                partner = i ^ stride
                
                if i < partner:
                    stage_t = torch.tensor([stage], device=device)
                    pos_t = torch.tensor([i], device=device)
                    a_t = torch.tensor([values[i]], device=device, dtype=torch.float)
                    b_t = torch.tensor([values[partner]], device=device, dtype=torch.float)
                    
                    pred_sum, pred_diff = model(stage_t, pos_t, a_t, b_t, track=False)
                    
                    new_values[i] = pred_sum.round().item()
                    new_values[partner] = pred_diff.round().item()
            
            values = new_values
    
    return [int(v) for v in values]


def compute_reference_fft(x):
    """Ground truth FFT."""
    N = len(x)
    result = [float(v) for v in x]
    
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
    
    return [int(v) for v in result]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    N = CONFIG['N']
    
    print("\n" + "=" * 70)
    print(f"PURE TRIX FFT - STAGED COMPOSITION (N={N})")
    print("Stage + Position → Routing meaning → Correct butterfly")
    print("=" * 70)
    
    # Generate data
    train_data = generate_staged_data(N, CONFIG['value_range'], num_samples=5000)
    test_data = generate_staged_data(N, CONFIG['value_range'], num_samples=1000)
    
    print(f"\nData: {len(train_data)} train, {len(test_data)} test")
    print(f"Stages: {int(np.log2(N))}")
    
    # Create model
    model = PureTriXStagedFFT(
        N=N,
        value_range=CONFIG['value_range'],
        d_model=CONFIG['d_model'],
        num_tiles=CONFIG['num_tiles'],
        num_freqs=CONFIG['num_freqs'],
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    
    print("\n[TRAINING]")
    
    best_acc = 0
    for epoch in range(CONFIG['epochs']):
        model.reset_tracking()
        loss = train_epoch(model, train_data, optimizer, device)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            metrics = evaluate(model, test_data, device)
            
            if metrics['both'] > best_acc:
                best_acc = metrics['both']
            
            stage_str = ' '.join(f"S{s}:{a:.0%}" for s, a in metrics['per_stage'].items())
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, "
                  f"both={metrics['both']:.1%} [{stage_str}]")
            
            if metrics['both'] >= 0.99:
                print("  ✓ 99%+ achieved!")
                break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    final = evaluate(model, test_data, device)
    
    print(f"\nAccuracy:")
    print(f"  Sum:  {final['sum']:.1%}")
    print(f"  Diff: {final['diff']:.1%}")
    print(f"  Both: {final['both']:.1%}")
    print(f"\nPer-stage:")
    for s, acc in final['per_stage'].items():
        print(f"  Stage {s}: {acc:.1%}")
    
    # Test full FFT composition
    print("\n[FULL FFT TEST]")
    
    num_correct = 0
    num_tests = 50
    
    for _ in range(num_tests):
        x = [np.random.randint(0, CONFIG['value_range']) for _ in range(N)]
        
        expected = compute_reference_fft(x)
        predicted = run_full_fft(model, x, device)
        
        if expected == predicted:
            num_correct += 1
    
    fft_accuracy = num_correct / num_tests
    print(f"Full FFT accuracy: {num_correct}/{num_tests} = {fft_accuracy:.1%}")
    
    # Show examples
    print("\nExamples:")
    for i in range(3):
        x = [np.random.randint(0, CONFIG['value_range']) for _ in range(N)]
        expected = compute_reference_fft(x)
        predicted = run_full_fft(model, x, device)
        
        match = "✓" if expected == predicted else "✗"
        print(f"  In:  {x}")
        print(f"  Exp: {expected}")
        print(f"  Got: {predicted} {match}")
        print()
    
    # Tile analysis
    print("[TILE SPECIALIZATION]")
    model.reset_tracking()
    model.train()
    for d in train_data[:2000]:
        stage = torch.tensor([d['stage']], device=device)
        pos = torch.tensor([d['pos']], device=device)
        a = torch.tensor([d['a']], device=device, dtype=torch.float)
        b = torch.tensor([d['b']], device=device, dtype=torch.float)
        model(stage, pos, a, b)
    
    analysis = model.get_tile_analysis()
    
    print("\nSUM path tiles:")
    for t, info in sorted(analysis['sum_tiles'].items()):
        print(f"  Tile {t}: Stage {info['dominant_stage']}, purity={info['purity']:.0%}")
    
    print("\nDIFF path tiles:")
    for t, info in sorted(analysis['diff_tiles'].items()):
        print(f"  Tile {t}: Stage {info['dominant_stage']}, purity={info['purity']:.0%}")
    
    # Verdict
    print("\n" + "=" * 70)
    if fft_accuracy >= 0.90:
        print(f"✓ PURE TRIX FFT N={N}: SUCCESS")
        print(f"  Staged butterfly accuracy: {final['both']:.1%}")
        print(f"  Full FFT accuracy: {fft_accuracy:.1%}")
        print("  Pure TriX composes into complete FFT!")
    elif fft_accuracy >= 0.50:
        print(f"◐ PARTIAL: FFT={fft_accuracy:.1%}, Butterfly={final['both']:.1%}")
    else:
        print(f"✗ NEEDS WORK: FFT={fft_accuracy:.1%}")
    print("=" * 70)
    
    # Save
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'config': CONFIG,
        'butterfly_accuracy': final,
        'fft_accuracy': fft_accuracy,
        'tile_analysis': analysis,
    }
    
    with open(output_dir / f'pure_trix_fft_staged_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nSaved to: results/fft_atoms/pure_trix_fft_staged_{timestamp}.json")
    
    return fft_accuracy


if __name__ == "__main__":
    main()
