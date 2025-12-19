#!/usr/bin/env python3
"""
Bracket Matching Experiment - The Atom of Temporal Tiles

Can 4-8 temporal tiles learn to count parentheses?

This is the minimal test of the temporal tile thesis:
- Requires counting (can't be done statelessly)
- Has clear ground truth (valid/invalid)
- Should produce interpretable tile structure

Expected outcome:
- Tiles correspond to depth levels (tile 0 = depth 0, etc.)
- Transition matrix shows counting logic
- Model achieves >95% accuracy

Usage:
    python experiments/bracket_matching.py
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import json
from collections import defaultdict

from trix.nn.temporal_tiles import TemporalTileLayer


# =============================================================================
# Data Generation
# =============================================================================

def generate_bracket_sequence(max_depth: int = 4, max_len: int = 20) -> Tuple[str, bool, List[int]]:
    """
    Generate a random bracket sequence with depth tracking.
    
    Returns:
        sequence: String of '(' and ')'
        valid: Whether sequence is valid
        depths: List of depth at each position (after the bracket)
    """
    sequence = []
    depths = []
    depth = 0
    
    length = np.random.randint(2, max_len + 1)
    
    for _ in range(length):
        # Decide what to add
        can_open = depth < max_depth
        can_close = depth > 0
        
        if can_open and can_close:
            # Random choice, slightly biased toward balance
            if np.random.random() < 0.5:
                choice = '('
            else:
                choice = ')'
        elif can_open:
            choice = '('
        elif can_close:
            choice = ')'
        else:
            break
        
        sequence.append(choice)
        if choice == '(':
            depth += 1
        else:
            depth -= 1
        depths.append(depth)
    
    # Valid if we end at depth 0 and never went negative
    valid = depth == 0 and all(d >= 0 for d in depths)
    
    return ''.join(sequence), valid, depths


def generate_invalid_sequence(max_len: int = 20) -> Tuple[str, bool, List[int]]:
    """Generate an intentionally invalid sequence."""
    method = np.random.choice(['unmatched_close', 'unmatched_open', 'negative'])
    
    if method == 'unmatched_close':
        # Start with close paren
        prefix_len = np.random.randint(0, max_len // 2)
        prefix = '()' * (prefix_len // 2)
        sequence = prefix + ')' + '(' * np.random.randint(0, 3)
    
    elif method == 'unmatched_open':
        # End with unclosed opens
        sequence = '(' * np.random.randint(1, 4) + '()' * np.random.randint(1, max_len // 4)
    
    else:  # negative
        # Go negative in the middle
        base = '()' * np.random.randint(1, max_len // 4)
        insert_pos = np.random.randint(0, len(base) + 1)
        sequence = base[:insert_pos] + ')(' + base[insert_pos:]
    
    # Compute depths
    depths = []
    depth = 0
    for c in sequence:
        if c == '(':
            depth += 1
        else:
            depth -= 1
        depths.append(depth)
    
    valid = depth == 0 and all(d >= 0 for d in depths)
    
    return sequence, valid, depths


def generate_dataset(n_samples: int, max_depth: int = 4, max_len: int = 20) -> List[Dict]:
    """Generate balanced dataset of valid/invalid sequences."""
    data = []
    
    n_valid = 0
    n_invalid = 0
    target_each = n_samples // 2
    
    while n_valid < target_each or n_invalid < target_each:
        if n_valid < target_each:
            seq, valid, depths = generate_bracket_sequence(max_depth, max_len)
            if valid:
                data.append({'sequence': seq, 'valid': True, 'depths': depths})
                n_valid += 1
        
        if n_invalid < target_each:
            seq, valid, depths = generate_invalid_sequence(max_len)
            if not valid:
                data.append({'sequence': seq, 'valid': False, 'depths': depths})
                n_invalid += 1
    
    np.random.shuffle(data)
    return data


# =============================================================================
# Model
# =============================================================================

class BracketMatcher(nn.Module):
    """
    Model that uses temporal tiles to determine bracket validity.
    
    Architecture:
        embed('(' or ')') → temporal tile layer → classifier
    
    The temporal tile layer should learn depth-tracking behavior.
    """
    
    def __init__(self, d_model: int = 32, d_state: int = 16, num_tiles: int = 8):
        super().__init__()
        
        # Embedding for ( and )
        self.embed = nn.Embedding(2, d_model)  # 0 = (, 1 = )
        
        # Temporal tile layer (the thing we're testing)
        self.temporal = TemporalTileLayer(
            d_model=d_model,
            d_state=d_state,
            num_tiles=num_tiles,
            state_init='zero',
        )
        
        # Classifier: takes final state and outputs valid/invalid
        self.classifier = nn.Sequential(
            nn.Linear(d_state + d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        
        self.d_model = d_model
        self.d_state = d_state
        self.num_tiles = num_tiles
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass.
        
        Args:
            x: Token indices (batch, seq_len), 0 = '(', 1 = ')'
        
        Returns:
            logits: Classification logits (batch, 2)
            infos: Routing info per timestep
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embed
        embedded = self.embed(x)  # (batch, seq_len, d_model)
        
        # Process through temporal tiles
        output, final_state, infos = self.temporal.forward_sequence(embedded)
        
        # Use final output and state for classification
        final_output = output[:, -1]  # (batch, d_model)
        classifier_input = torch.cat([final_output, final_state], dim=-1)
        
        logits = self.classifier(classifier_input)
        
        return logits, infos
    
    def get_tile_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """Get the sequence of tile selections for analysis."""
        _, infos = self.forward(x)
        tiles = torch.stack([info['tile_idx'] for info in infos], dim=1)
        return tiles


# =============================================================================
# Training
# =============================================================================

def prepare_batch(samples: List[Dict], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Prepare a batch with padding."""
    max_len = max(len(s['sequence']) for s in samples)
    
    batch_x = torch.zeros(len(samples), max_len, dtype=torch.long, device=device)
    batch_y = torch.zeros(len(samples), dtype=torch.long, device=device)
    
    for i, sample in enumerate(samples):
        seq = sample['sequence']
        for j, c in enumerate(seq):
            batch_x[i, j] = 0 if c == '(' else 1
        batch_y[i] = 1 if sample['valid'] else 0
    
    return batch_x, batch_y, max_len


def train_epoch(model: BracketMatcher, data: List[Dict], optimizer, device: torch.device, batch_size: int = 32) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    # Shuffle data
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    for i in range(0, len(data), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_samples = [data[j] for j in batch_indices]
        
        x, y, _ = prepare_batch(batch_samples, device)
        
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model: BracketMatcher, data: List[Dict], device: torch.device, batch_size: int = 32) -> Dict:
    """Evaluate model."""
    model.eval()
    
    correct = 0
    total = 0
    correct_valid = 0
    total_valid = 0
    correct_invalid = 0
    total_invalid = 0
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_samples = data[i:i + batch_size]
            x, y, _ = prepare_batch(batch_samples, device)
            
            logits, _ = model(x)
            preds = logits.argmax(dim=-1)
            
            correct += (preds == y).sum().item()
            total += len(y)
            
            # Per-class accuracy
            valid_mask = y == 1
            invalid_mask = y == 0
            
            correct_valid += ((preds == y) & valid_mask).sum().item()
            total_valid += valid_mask.sum().item()
            
            correct_invalid += ((preds == y) & invalid_mask).sum().item()
            total_invalid += invalid_mask.sum().item()
    
    return {
        'accuracy': correct / total,
        'valid_accuracy': correct_valid / total_valid if total_valid > 0 else 0,
        'invalid_accuracy': correct_invalid / total_invalid if total_invalid > 0 else 0,
    }


# =============================================================================
# Analysis
# =============================================================================

def analyze_tile_behavior(model: BracketMatcher, data: List[Dict], device: torch.device) -> Dict:
    """Analyze what tiles learned about depth."""
    model.eval()
    
    # Track: for each (tile, depth), count occurrences
    tile_depth_counts = defaultdict(lambda: defaultdict(int))
    tile_token_counts = defaultdict(lambda: defaultdict(int))  # tile -> token -> count
    
    with torch.no_grad():
        for sample in data[:500]:  # Analyze subset
            seq = sample['sequence']
            depths = [0] + sample['depths'][:-1]  # Depth BEFORE each token
            
            # Prepare single sample
            x = torch.zeros(1, len(seq), dtype=torch.long, device=device)
            for j, c in enumerate(seq):
                x[0, j] = 0 if c == '(' else 1
            
            # Get tile trajectory
            tiles = model.get_tile_trajectory(x)[0]  # (seq_len,)
            
            for t, (tile, depth, token) in enumerate(zip(tiles.tolist(), depths, seq)):
                tile_depth_counts[tile][depth] += 1
                tile_token_counts[tile][token] += 1
    
    # Compute purity: for each tile, what's the dominant depth?
    tile_analysis = {}
    for tile in range(model.num_tiles):
        depth_counts = tile_depth_counts[tile]
        if depth_counts:
            total = sum(depth_counts.values())
            dominant_depth = max(depth_counts, key=depth_counts.get)
            purity = depth_counts[dominant_depth] / total
            
            token_counts = tile_token_counts[tile]
            token_dist = {k: v / sum(token_counts.values()) for k, v in token_counts.items()}
            
            tile_analysis[tile] = {
                'dominant_depth': dominant_depth,
                'purity': purity,
                'depth_distribution': dict(depth_counts),
                'token_distribution': token_dist,
                'total_activations': total,
            }
    
    return tile_analysis


def print_analysis(analysis: Dict, transition_matrix: torch.Tensor):
    """Print readable analysis."""
    print("\n" + "=" * 70)
    print("TILE BEHAVIOR ANALYSIS")
    print("=" * 70)
    
    # Sort by dominant depth for readability
    sorted_tiles = sorted(
        analysis.items(),
        key=lambda x: (x[1]['dominant_depth'], -x[1]['total_activations'])
    )
    
    print("\nTile Specialization:")
    print("-" * 50)
    for tile_id, info in sorted_tiles:
        if info['total_activations'] > 10:
            token_bias = info['token_distribution'].get('(', 0) - info['token_distribution'].get(')', 0)
            token_str = "open-biased" if token_bias > 0.2 else "close-biased" if token_bias < -0.2 else "balanced"
            print(f"  Tile {tile_id}: depth={info['dominant_depth']}, "
                  f"purity={info['purity']:.0%}, "
                  f"n={info['total_activations']}, "
                  f"{token_str}")
    
    print("\nTransition Matrix (rows=from, cols=to):")
    print("-" * 50)
    trans = transition_matrix.cpu().numpy()
    
    # Print header
    print("     ", end="")
    for j in range(trans.shape[1]):
        print(f"  T{j:d}  ", end="")
    print()
    
    for i in range(trans.shape[0]):
        print(f"T{i}: ", end="")
        for j in range(trans.shape[1]):
            val = trans[i, j]
            if val > 0.1:
                print(f" {val:.2f} ", end="")
            else:
                print("  .   ", end="")
        print()
    
    print("\nInterpretation:")
    print("-" * 50)
    
    # Try to identify the counting structure
    depth_to_tile = {}
    for tile_id, info in analysis.items():
        if info['purity'] > 0.6 and info['total_activations'] > 20:
            depth = info['dominant_depth']
            if depth not in depth_to_tile or info['purity'] > analysis[depth_to_tile[depth]]['purity']:
                depth_to_tile[depth] = tile_id
    
    if depth_to_tile:
        print("  Learned depth mapping:")
        for depth in sorted(depth_to_tile.keys()):
            tile = depth_to_tile[depth]
            purity = analysis[tile]['purity']
            print(f"    Depth {depth} → Tile {tile} (purity={purity:.0%})")
    else:
        print("  No clear depth mapping found (tiles not yet specialized)")


# =============================================================================
# Main
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Hyperparameters
    D_MODEL = 32
    D_STATE = 16
    NUM_TILES = 8  # Enough for depths 0-7
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 0.001
    
    # Generate data
    print("\n[1] Generating data...")
    train_data = generate_dataset(2000, max_depth=4, max_len=16)
    test_data = generate_dataset(500, max_depth=4, max_len=16)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Sample
    print("\n  Examples:")
    for sample in train_data[:5]:
        status = "valid" if sample['valid'] else "INVALID"
        print(f"    {sample['sequence']:20s} → {status}")
    
    # Create model
    print(f"\n[2] Creating model...")
    model = BracketMatcher(
        d_model=D_MODEL,
        d_state=D_STATE,
        num_tiles=NUM_TILES,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Tiles: {NUM_TILES}")
    print(f"  State dim: {D_STATE}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Train
    print(f"\n[3] Training for {EPOCHS} epochs...")
    model.temporal.reset_tracking()
    
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_data, optimizer, device, BATCH_SIZE)
        
        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, test_data, device, BATCH_SIZE)
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, "
                  f"acc={metrics['accuracy']:.1%}, "
                  f"valid={metrics['valid_accuracy']:.1%}, "
                  f"invalid={metrics['invalid_accuracy']:.1%}")
    
    # Final evaluation
    print("\n[4] Final evaluation...")
    final_metrics = evaluate(model, test_data, device, BATCH_SIZE)
    print(f"  Accuracy: {final_metrics['accuracy']:.1%}")
    print(f"  Valid sequences: {final_metrics['valid_accuracy']:.1%}")
    print(f"  Invalid sequences: {final_metrics['invalid_accuracy']:.1%}")
    
    # Analyze tile behavior
    print("\n[5] Analyzing tile behavior...")
    analysis = analyze_tile_behavior(model, test_data, device)
    transition_matrix = model.temporal.get_transition_matrix()
    
    print_analysis(analysis, transition_matrix)
    
    # Regime analysis
    print("\n[6] Regime analysis...")
    regime_info = model.temporal.get_regime_analysis()
    print(f"  Stable tiles (>50% self-transition): {regime_info['stable_tiles']}")
    print(f"  Hub tiles (high transition entropy): {regime_info['hub_tiles']}")
    
    # Usage distribution
    usage = regime_info['usage'].cpu().numpy()
    print(f"\n  Tile usage distribution:")
    for i, u in enumerate(usage):
        if u > 0.01:
            bar = '█' * int(u * 50)
            print(f"    Tile {i}: {u:.1%} {bar}")
    
    # Test specific sequences
    print("\n[7] Testing specific sequences...")
    test_cases = [
        "()",
        "(())",
        "((()))",
        "()()",
        "(()())",
        "(",
        ")",
        "())",
        "(()",
        ")(",
    ]
    
    model.eval()
    with torch.no_grad():
        for seq in test_cases:
            x = torch.zeros(1, len(seq), dtype=torch.long, device=device)
            for j, c in enumerate(seq):
                x[0, j] = 0 if c == '(' else 1
            
            logits, infos = model(x)
            pred = logits.argmax(dim=-1).item()
            pred_str = "valid" if pred == 1 else "INVALID"
            
            # Ground truth
            depth = 0
            valid = True
            for c in seq:
                depth += 1 if c == '(' else -1
                if depth < 0:
                    valid = False
                    break
            if depth != 0:
                valid = False
            true_str = "valid" if valid else "INVALID"
            
            match = "✓" if (pred == 1) == valid else "✗"
            
            # Tile trajectory
            tiles = [info['tile_idx'][0].item() for info in infos]
            tile_str = "→".join(str(t) for t in tiles)
            
            print(f"  {seq:12s} pred={pred_str:8s} true={true_str:8s} {match}  tiles: {tile_str}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if final_metrics['accuracy'] > 0.95:
        print("✓ SUCCESS: Temporal tiles learned to count brackets!")
        print(f"  Accuracy: {final_metrics['accuracy']:.1%}")
        
        # Check if tiles correspond to depths
        depth_mapped = sum(1 for info in analysis.values() 
                          if info['purity'] > 0.6 and info['total_activations'] > 20)
        print(f"  Depth-specialized tiles: {depth_mapped}/{NUM_TILES}")
        
    elif final_metrics['accuracy'] > 0.80:
        print("~ PARTIAL: Learning occurring but not complete")
        print(f"  Accuracy: {final_metrics['accuracy']:.1%}")
        print("  Consider: more epochs, larger state, or architecture changes")
        
    else:
        print("✗ FAILURE: Temporal tiles did not learn counting")
        print(f"  Accuracy: {final_metrics['accuracy']:.1%}")
        print("  Investigate: state contribution, tile diversity, gradient flow")
    
    print("=" * 70)
    
    # Return results for further analysis
    return {
        'accuracy': final_metrics['accuracy'],
        'tile_analysis': analysis,
        'transition_matrix': transition_matrix.cpu().numpy().tolist(),
        'model': model,
    }


if __name__ == "__main__":
    results = main()
