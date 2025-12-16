#!/usr/bin/env python3
"""
FFT Atom 1: ADDRESS
===================

The algorithmic heart of FFT: given (stage, index), output partner index.

partner(i, s) = i XOR 2^s

This tests whether TDSR can learn algorithmic structure - no arithmetic,
pure control flow and addressing logic.

CODENAME: ANN WILSON - HEART Atom Test
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
from collections import defaultdict

from trix.nn import TemporalTileLayer


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'N': 8,                    # FFT size
    'num_stages': 3,           # log2(8) = 3 stages
    'd_model': 32,
    'd_state': 16,
    'num_tiles': 8,            # One per index, or learns differently
    'epochs': 100,
    'batch_size': 64,
    'lr': 0.005,
    'seeds': [42, 123, 456],   # Multiple runs for stability
}


# =============================================================================
# Data Generation
# =============================================================================

def generate_address_data(N=8):
    """
    Generate all (stage, index) -> partner pairs for N-point FFT.
    
    partner(i, s) = i XOR 2^s
    
    Returns list of (stage, index, partner) tuples.
    """
    num_stages = int(np.log2(N))
    data = []
    
    for s in range(num_stages):
        stride = 2 ** s
        for i in range(N):
            partner = i ^ stride  # XOR with 2^s
            data.append({
                'stage': s,
                'index': i,
                'partner': partner,
                'stride': stride,
            })
    
    return data


def create_datasets(N=8, train_ratio=0.8):
    """Create train/test splits."""
    all_data = generate_address_data(N)
    
    # For this small dataset, we'll use all combinations for both
    # but shuffle differently for train/test
    np.random.shuffle(all_data)
    
    # Actually, for ADDRESS, the dataset is small (N * log2(N) = 24 for N=8)
    # So we'll train on all and test on all (it should memorize perfectly if it works)
    # The real test is whether the STRUCTURE emerges in the tiles
    
    return all_data, all_data


# =============================================================================
# Model
# =============================================================================

class AddressPredictor(nn.Module):
    """
    Predict FFT butterfly partner addresses using temporal tiles.
    
    Input: (stage, index) encoded
    Output: partner index (0 to N-1)
    
    The key question: Do tiles specialize by stage?
    """
    
    def __init__(self, N=8, d_model=32, d_state=16, num_tiles=8):
        super().__init__()
        
        self.N = N
        self.num_stages = int(np.log2(N))
        
        # Embeddings for stage and index
        self.stage_embed = nn.Embedding(self.num_stages, d_model // 2)
        self.index_embed = nn.Embedding(N, d_model // 2)
        
        # Temporal tile layer
        self.temporal = TemporalTileLayer(
            d_model=d_model,
            d_state=d_state,
            num_tiles=num_tiles,
            routing_temp=0.5,
        )
        
        # Output head: predict partner index
        self.head = nn.Linear(d_model, N)
        
        self.d_model = d_model
        self.d_state = d_state
        self.num_tiles = num_tiles
        
        # Tracking: which tiles handle which stages
        self.register_buffer('tile_stage_counts', torch.zeros(num_tiles, self.num_stages))
    
    def forward(self, stage, index, track=True):
        """
        Forward pass.
        
        Args:
            stage: (batch,) stage indices 0 to num_stages-1
            index: (batch,) input indices 0 to N-1
            track: whether to track tile-stage correspondence
        
        Returns:
            logits: (batch, N) predictions for partner index
            info: routing information
        """
        batch_size = stage.shape[0]
        device = stage.device
        
        # Embed inputs
        stage_emb = self.stage_embed(stage)
        index_emb = self.index_embed(index)
        x = torch.cat([stage_emb, index_emb], dim=-1)
        
        # Initialize state
        state = self.temporal.init_state(batch_size, device)
        
        # Single-step temporal processing
        # (Could do multi-step for sequence, but ADDRESS is single-query)
        output, new_state, info = self.temporal(x, state)
        
        # Predict partner
        logits = self.head(output)
        
        # Track tile-stage correspondence
        if track and self.training:
            with torch.no_grad():
                for b in range(batch_size):
                    tile = info['tile_idx'][b].item()
                    s = stage[b].item()
                    self.tile_stage_counts[tile, s] += 1
        
        return logits, info
    
    def get_tile_stage_purity(self):
        """Analyze tile specialization by stage."""
        counts = self.tile_stage_counts.cpu().numpy()
        analysis = {}
        
        for tile in range(self.num_tiles):
            total = counts[tile].sum()
            if total > 0:
                dominant_stage = counts[tile].argmax()
                purity = counts[tile, dominant_stage] / total
                analysis[tile] = {
                    'dominant_stage': int(dominant_stage),
                    'purity': float(purity),
                    'counts': counts[tile].tolist(),
                    'total': int(total),
                }
        
        return analysis
    
    def reset_tracking(self):
        self.tile_stage_counts.zero_()
        self.temporal.reset_tracking()


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, data, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    total_loss = 0
    correct = 0
    total = 0
    
    batch_size = CONFIG['batch_size']
    
    for i in range(0, len(data), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = [data[j] for j in batch_idx]
        
        stage = torch.tensor([d['stage'] for d in batch], device=device)
        index = torch.tensor([d['index'] for d in batch], device=device)
        partner = torch.tensor([d['partner'] for d in batch], device=device)
        
        logits, info = model(stage, index)
        loss = F.cross_entropy(logits, partner)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == partner).sum().item()
        total += len(batch)
    
    return total_loss / (len(data) // batch_size + 1), correct / total


def evaluate(model, data, device):
    """Evaluate model."""
    model.eval()
    
    correct = 0
    total = 0
    
    # Per-stage accuracy
    stage_correct = defaultdict(int)
    stage_total = defaultdict(int)
    
    # Predictions for analysis
    predictions = []
    
    with torch.no_grad():
        for d in data:
            stage = torch.tensor([d['stage']], device=device)
            index = torch.tensor([d['index']], device=device)
            partner = d['partner']
            
            logits, info = model(stage, index, track=False)
            pred = logits.argmax(dim=-1).item()
            tile = info['tile_idx'][0].item()
            
            is_correct = pred == partner
            correct += is_correct
            total += 1
            
            stage_correct[d['stage']] += is_correct
            stage_total[d['stage']] += 1
            
            predictions.append({
                'stage': d['stage'],
                'index': d['index'],
                'partner_true': partner,
                'partner_pred': pred,
                'correct': is_correct,
                'tile': tile,
            })
    
    per_stage = {s: stage_correct[s] / stage_total[s] for s in stage_total}
    
    return correct / total, per_stage, predictions


# =============================================================================
# Analysis
# =============================================================================

def analyze_results(model, predictions, device):
    """Deep analysis of what the model learned."""
    
    # Tile-stage correspondence
    tile_stage = model.get_tile_stage_purity()
    
    # Tile-index correspondence
    tile_index_counts = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        tile_index_counts[p['tile']][p['index']] += 1
    
    # Build transition info from temporal layer
    transition_matrix = model.temporal.get_transition_matrix().cpu().numpy()
    
    # Check if tiles are stage-aligned
    stage_aligned = True
    stage_to_tile = {}
    for tile, info in tile_stage.items():
        if info['purity'] < 0.8:
            stage_aligned = False
        else:
            stage_to_tile[info['dominant_stage']] = tile
    
    # Compile readiness: can we freeze this as a dispatch table?
    compile_ready = stage_aligned and len(stage_to_tile) >= model.num_stages
    
    analysis = {
        'tile_stage_purity': tile_stage,
        'stage_to_tile_mapping': stage_to_tile,
        'stage_aligned': stage_aligned,
        'compile_ready': compile_ready,
        'transition_matrix': transition_matrix.tolist(),
    }
    
    return analysis


def print_report(accuracy, per_stage, analysis, epoch=None):
    """Print human-readable report."""
    
    print("\n" + "=" * 70)
    if epoch is not None:
        print(f"EPOCH {epoch} REPORT")
    else:
        print("FINAL REPORT")
    print("=" * 70)
    
    print(f"\n[ACCURACY]")
    print(f"  Overall: {accuracy:.1%}")
    print(f"  Per-stage:")
    for s, acc in sorted(per_stage.items()):
        bar = '█' * int(acc * 20)
        print(f"    Stage {s}: {acc:.1%} {bar}")
    
    print(f"\n[TILE-STAGE SPECIALIZATION]")
    for tile, info in sorted(analysis['tile_stage_purity'].items()):
        print(f"  Tile {tile}: stage={info['dominant_stage']}, "
              f"purity={info['purity']:.0%}, n={info['total']}")
    
    print(f"\n[STRUCTURE]")
    print(f"  Stage-aligned: {'✓' if analysis['stage_aligned'] else '✗'}")
    print(f"  Compile-ready: {'✓' if analysis['compile_ready'] else '✗'}")
    
    if analysis['stage_to_tile_mapping']:
        print(f"  Stage→Tile mapping: {analysis['stage_to_tile_mapping']}")
    
    print("=" * 70)


# =============================================================================
# Main Runner
# =============================================================================

def run_single_seed(seed, device):
    """Run experiment with single seed."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data
    train_data, test_data = create_datasets(CONFIG['N'])
    
    # Create model
    model = AddressPredictor(
        N=CONFIG['N'],
        d_model=CONFIG['d_model'],
        d_state=CONFIG['d_state'],
        num_tiles=CONFIG['num_tiles'],
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Training loop
    history = []
    best_accuracy = 0
    
    for epoch in range(CONFIG['epochs']):
        model.reset_tracking()
        
        loss, train_acc = train_epoch(model, train_data, optimizer, device)
        test_acc, per_stage, predictions = evaluate(model, test_data, device)
        
        history.append({
            'epoch': epoch,
            'loss': loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'per_stage': per_stage,
        })
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        
        # Early stopping on perfect accuracy
        if test_acc >= 1.0:
            print(f"  [Seed {seed}] Perfect accuracy at epoch {epoch + 1}!")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"  [Seed {seed}] Epoch {epoch + 1}: loss={loss:.4f}, "
                  f"acc={test_acc:.1%}")
    
    # Final analysis
    model.reset_tracking()
    
    # Re-run to populate tracking
    model.train()
    for d in train_data:
        stage = torch.tensor([d['stage']], device=device)
        index = torch.tensor([d['index']], device=device)
        model(stage, index, track=True)
    
    final_acc, per_stage, predictions = evaluate(model, test_data, device)
    analysis = analyze_results(model, predictions, device)
    
    return {
        'seed': seed,
        'final_accuracy': final_acc,
        'per_stage_accuracy': per_stage,
        'analysis': analysis,
        'history': history,
        'predictions': predictions,
    }


def main():
    """Main experiment runner."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"N={CONFIG['N']}, stages={CONFIG['num_stages']}")
    
    # Show the task
    print("\n[TASK: ADDRESS]")
    print("Learn partner(i, s) = i XOR 2^s")
    print("\nExamples:")
    for s in range(CONFIG['num_stages']):
        stride = 2 ** s
        examples = [(i, i ^ stride) for i in range(4)]
        print(f"  Stage {s} (stride={stride}): {examples[:4]}...")
    
    # Run multiple seeds
    print(f"\n[TRAINING]")
    all_results = []
    
    for seed in CONFIG['seeds']:
        print(f"\nSeed {seed}:")
        result = run_single_seed(seed, device)
        all_results.append(result)
    
    # Aggregate results
    accuracies = [r['final_accuracy'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # Best run for detailed analysis
    best_run = max(all_results, key=lambda r: r['final_accuracy'])
    
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"\nAccuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"Seeds: {accuracies}")
    
    print_report(
        best_run['final_accuracy'],
        best_run['per_stage_accuracy'],
        best_run['analysis'],
    )
    
    # Pass/Fail determination
    print("\n[VERDICT]")
    passed = mean_acc >= 0.95 and best_run['analysis']['stage_aligned']
    
    if mean_acc >= 0.95:
        print("✓ ACCURACY: ≥95% achieved")
    else:
        print(f"✗ ACCURACY: {mean_acc:.1%} < 95%")
    
    if best_run['analysis']['stage_aligned']:
        print("✓ STRUCTURE: Tiles are stage-aligned")
    else:
        print("✗ STRUCTURE: Tiles not stage-aligned")
    
    if best_run['analysis']['compile_ready']:
        print("✓ COMPILE: Ready to freeze as dispatch table")
    else:
        print("✗ COMPILE: Not ready for compilation")
    
    print(f"\n{'=' * 70}")
    if passed:
        print("ATOM ADDRESS: ✓ PASS")
        print("TDSR learned FFT butterfly addressing structure!")
    else:
        print("ATOM ADDRESS: ✗ FAIL")
        print("TDSR did not learn the addressing structure cleanly.")
    print("=" * 70)
    
    # Save results
    output_dir = Path('/workspace/trix_latest/results/fft_atoms')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'atom_address_{timestamp}.json'
    
    results_json = {
        'config': CONFIG,
        'timestamp': timestamp,
        'device': device,
        'aggregate': {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'all_accuracies': accuracies,
            'passed': passed,
        },
        'best_run': {
            'seed': best_run['seed'],
            'accuracy': best_run['final_accuracy'],
            'per_stage': best_run['per_stage_accuracy'],
            'stage_aligned': best_run['analysis']['stage_aligned'],
            'compile_ready': best_run['analysis']['compile_ready'],
            'stage_to_tile': best_run['analysis']['stage_to_tile_mapping'],
        },
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results_json


if __name__ == "__main__":
    results = main()
