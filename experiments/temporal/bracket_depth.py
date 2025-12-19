#!/usr/bin/env python3
"""
Bracket Depth Prediction - Forces Actual Counting

Instead of predicting valid/invalid, predict the depth after each bracket.
This forces the temporal tiles to actually learn counting.

If tile_t predicts depth_t correctly, it MUST be tracking state.

Usage:
    python experiments/bracket_depth.py
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from trix.nn.temporal_tiles import TemporalTileLayer


# =============================================================================
# Data Generation
# =============================================================================

def generate_sequence(max_depth: int = 4, length: int = 10) -> Tuple[List[int], List[int]]:
    """
    Generate a bracket sequence with depth labels.
    
    Returns:
        tokens: List of 0 (open) or 1 (close)
        depths: List of depth after each token
    """
    tokens = []
    depths = []
    depth = 0
    
    for _ in range(length):
        can_open = depth < max_depth
        can_close = depth > 0
        
        if can_open and can_close:
            token = np.random.randint(0, 2)
        elif can_open:
            token = 0  # Must open
        else:
            token = 1  # Must close
        
        tokens.append(token)
        depth = depth + 1 if token == 0 else depth - 1
        depths.append(depth)
    
    return tokens, depths


def generate_dataset(n_samples: int, max_depth: int = 4, min_len: int = 4, max_len: int = 12) -> List[Dict]:
    """Generate dataset with variable length sequences."""
    data = []
    for _ in range(n_samples):
        length = np.random.randint(min_len, max_len + 1)
        tokens, depths = generate_sequence(max_depth, length)
        data.append({
            'tokens': tokens,
            'depths': depths,
            'length': length,
        })
    return data


# =============================================================================
# Model
# =============================================================================

class DepthPredictor(nn.Module):
    """
    Predict depth at each position using temporal tiles.
    
    The model must track depth in its state to predict correctly.
    """
    
    def __init__(self, d_model: int = 32, d_state: int = 8, num_tiles: int = 6, max_depth: int = 5):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.num_tiles = num_tiles
        self.max_depth = max_depth
        
        # Simple embedding
        self.embed = nn.Embedding(2, d_model)
        
        # Temporal tile layer
        self.temporal = TemporalTileLayer(
            d_model=d_model,
            d_state=d_state,
            num_tiles=num_tiles,
            state_init='zero',
            routing_temp=0.5,  # Sharper routing
        )
        
        # Predict depth (0 to max_depth)
        self.depth_head = nn.Linear(d_model, max_depth + 1)
    
    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Args:
            tokens: (batch, seq_len) of 0s and 1s
        
        Returns:
            depth_logits: (batch, seq_len, max_depth+1)
            infos: Routing info per timestep
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Embed
        x = self.embed(tokens)
        
        # Process through temporal tiles
        output, final_state, infos = self.temporal.forward_sequence(x)
        
        # Predict depth at each position
        depth_logits = self.depth_head(output)
        
        return depth_logits, infos


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, data, optimizer, device, batch_size=32):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    
    for i in range(0, len(data), batch_size):
        batch_data = [data[indices[j]] for j in range(i, min(i + batch_size, len(data)))]
        
        # Pad to max length in batch
        max_len = max(d['length'] for d in batch_data)
        
        tokens = torch.zeros(len(batch_data), max_len, dtype=torch.long, device=device)
        depths = torch.zeros(len(batch_data), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(batch_data), max_len, dtype=torch.bool, device=device)
        
        for j, d in enumerate(batch_data):
            tokens[j, :d['length']] = torch.tensor(d['tokens'], device=device)
            depths[j, :d['length']] = torch.tensor(d['depths'], device=device)
            mask[j, :d['length']] = True
        
        # Forward
        logits, _ = model(tokens)
        
        # Loss only on valid positions
        loss = F.cross_entropy(
            logits[mask].view(-1, model.max_depth + 1),
            depths[mask].view(-1)
        )
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        correct = ((preds == depths) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (len(data) // batch_size), total_correct / total_tokens


def evaluate(model, data, device, batch_size=32):
    model.eval()
    total_correct = 0
    total_tokens = 0
    
    # Per-depth accuracy
    depth_correct = defaultdict(int)
    depth_total = defaultdict(int)
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            max_len = max(d['length'] for d in batch_data)
            
            tokens = torch.zeros(len(batch_data), max_len, dtype=torch.long, device=device)
            depths = torch.zeros(len(batch_data), max_len, dtype=torch.long, device=device)
            mask = torch.zeros(len(batch_data), max_len, dtype=torch.bool, device=device)
            
            for j, d in enumerate(batch_data):
                tokens[j, :d['length']] = torch.tensor(d['tokens'], device=device)
                depths[j, :d['length']] = torch.tensor(d['depths'], device=device)
                mask[j, :d['length']] = True
            
            logits, _ = model(tokens)
            preds = logits.argmax(dim=-1)
            
            correct = ((preds == depths) & mask)
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Per-depth tracking
            for j, d in enumerate(batch_data):
                for t, true_depth in enumerate(d['depths']):
                    pred_depth = preds[j, t].item()
                    depth_total[true_depth] += 1
                    if pred_depth == true_depth:
                        depth_correct[true_depth] += 1
    
    per_depth_acc = {d: depth_correct[d] / depth_total[d] 
                     for d in sorted(depth_total.keys()) if depth_total[d] > 0}
    
    return total_correct / total_tokens, per_depth_acc


# =============================================================================
# Analysis
# =============================================================================

def analyze_tiles(model, data, device):
    """Analyze tile-depth correspondence."""
    model.eval()
    
    tile_depth_counts = defaultdict(lambda: defaultdict(int))
    tile_token_counts = defaultdict(lambda: defaultdict(int))
    
    with torch.no_grad():
        for sample in data[:200]:
            tokens = torch.tensor([sample['tokens']], device=device)
            _, infos = model(tokens)
            
            prev_depth = 0
            for t, (token, true_depth) in enumerate(zip(sample['tokens'], sample['depths'])):
                tile = infos[t]['tile_idx'][0].item()
                # The tile sees input at depth prev_depth and should transition to true_depth
                tile_depth_counts[tile][prev_depth] += 1
                tile_token_counts[tile][token] += 1
                prev_depth = true_depth
    
    print("\n" + "=" * 60)
    print("TILE-DEPTH ANALYSIS")
    print("=" * 60)
    
    for tile in range(model.num_tiles):
        dc = tile_depth_counts[tile]
        tc = tile_token_counts[tile]
        if dc:
            total = sum(dc.values())
            dom_depth = max(dc, key=dc.get)
            purity = dc[dom_depth] / total
            
            open_pct = tc.get(0, 0) / total * 100
            close_pct = tc.get(1, 0) / total * 100
            
            print(f"Tile {tile}: handles depth {dom_depth} (purity={purity:.0%}, n={total})")
            print(f"         open={open_pct:.0f}%, close={close_pct:.0f}%")
            print(f"         depth dist: {dict(dc)}")


# =============================================================================
# Main
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Smaller, focused config
    MAX_DEPTH = 4
    NUM_TILES = 6  # Slightly more than max_depth
    D_MODEL = 24
    D_STATE = 8
    EPOCHS = 100
    
    print(f"\nConfig: max_depth={MAX_DEPTH}, tiles={NUM_TILES}, d_state={D_STATE}")
    
    # Generate data
    print("\n[1] Generating data...")
    train_data = generate_dataset(3000, max_depth=MAX_DEPTH, min_len=4, max_len=12)
    test_data = generate_dataset(500, max_depth=MAX_DEPTH, min_len=4, max_len=12)
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Show examples
    print("\n  Examples:")
    for sample in train_data[:3]:
        toks = ''.join('(' if t == 0 else ')' for t in sample['tokens'])
        deps = ''.join(str(d) for d in sample['depths'])
        print(f"    {toks}")
        print(f"    {deps}")
        print()
    
    # Create model
    model = DepthPredictor(
        d_model=D_MODEL,
        d_state=D_STATE,
        num_tiles=NUM_TILES,
        max_depth=MAX_DEPTH,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[2] Model: {n_params:,} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    # Train
    print(f"\n[3] Training...")
    model.temporal.reset_tracking()
    
    best_acc = 0
    for epoch in range(EPOCHS):
        loss, train_acc = train_epoch(model, train_data, optimizer, device)
        
        if (epoch + 1) % 20 == 0:
            test_acc, per_depth = evaluate(model, test_data, device)
            depth_str = ' '.join(f"d{d}:{a:.0%}" for d, a in sorted(per_depth.items()))
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f} train={train_acc:.1%} test={test_acc:.1%}")
            print(f"            {depth_str}")
            
            if test_acc > best_acc:
                best_acc = test_acc
    
    # Final eval
    print(f"\n[4] Final evaluation...")
    test_acc, per_depth = evaluate(model, test_data, device)
    print(f"  Overall accuracy: {test_acc:.1%}")
    print(f"  Per-depth accuracy:")
    for d, acc in sorted(per_depth.items()):
        bar = '█' * int(acc * 20)
        print(f"    Depth {d}: {acc:5.1%} {bar}")
    
    # Analyze tiles
    analyze_tiles(model, test_data, device)
    
    # Show transition matrix
    print("\n[5] Transition matrix:")
    trans = model.temporal.get_transition_matrix().cpu().numpy()
    print("    ", end="")
    for j in range(NUM_TILES):
        print(f" T{j}  ", end="")
    print()
    for i in range(NUM_TILES):
        print(f"T{i}: ", end="")
        for j in range(NUM_TILES):
            v = trans[i, j]
            if v > 0.1:
                print(f"{v:.2f} ", end="")
            else:
                print(" .   ", end="")
        print()
    
    # Test specific sequences
    print("\n[6] Test sequences:")
    test_seqs = [
        [0, 1],           # ()       -> depths: 1, 0
        [0, 0, 1, 1],     # (())     -> depths: 1, 2, 1, 0
        [0, 1, 0, 1],     # ()()     -> depths: 1, 0, 1, 0
        [0, 0, 0, 1, 1, 1], # ((()))  -> depths: 1, 2, 3, 2, 1, 0
    ]
    
    model.eval()
    with torch.no_grad():
        for seq in test_seqs:
            tokens = torch.tensor([seq], device=device)
            logits, infos = model(tokens)
            preds = logits.argmax(dim=-1)[0].tolist()
            tiles = [info['tile_idx'][0].item() for info in infos]
            
            true_depths = []
            d = 0
            for t in seq:
                d = d + 1 if t == 0 else d - 1
                true_depths.append(d)
            
            seq_str = ''.join('(' if t == 0 else ')' for t in seq)
            match = '✓' if preds == true_depths else '✗'
            
            print(f"  {seq_str:12s} true={true_depths} pred={preds} {match}")
            print(f"             tiles={tiles}")
    
    # Summary
    print("\n" + "=" * 60)
    if test_acc > 0.90:
        print(f"✓ SUCCESS: {test_acc:.1%} accuracy - temporal tiles learned counting!")
    elif test_acc > 0.70:
        print(f"~ PARTIAL: {test_acc:.1%} accuracy - learning but not complete")
    else:
        print(f"✗ NEEDS WORK: {test_acc:.1%} accuracy")
    print("=" * 60)
    
    return test_acc, model


if __name__ == "__main__":
    main()
