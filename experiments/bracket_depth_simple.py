#!/usr/bin/env python3
"""
Bracket Depth - Simplified Temporal State Model

Even simpler: just use a GRU/RNN-like state update, but with discrete
tile selection for interpretability. The key insight is whether
discrete routing can track depth.

Usage:
    python experiments/bracket_depth_simple.py
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


# =============================================================================
# Simple Temporal Model
# =============================================================================

class SimpleTemporalTiles(nn.Module):
    """
    Simplified temporal tiles - shared compute with tile-specific gating.
    
    Each tile has a signature but they share the state update network.
    This makes training faster while preserving interpretable routing.
    """
    
    def __init__(self, d_input, d_state, num_tiles):
        super().__init__()
        
        self.d_input = d_input
        self.d_state = d_state
        self.num_tiles = num_tiles
        
        # Signatures for routing (match on input + state)
        self.signatures = nn.Parameter(torch.randn(num_tiles, d_input + d_state) * 0.1)
        
        # Shared state update (GRU-like)
        self.state_update = nn.GRUCell(d_input, d_state)
        
        # Tile-specific output modulation
        self.tile_gates = nn.Parameter(torch.ones(num_tiles, d_state))
        self.tile_biases = nn.Parameter(torch.zeros(num_tiles, d_state))
        
        # Tracking
        self.register_buffer('tile_counts', torch.zeros(num_tiles))
        self.register_buffer('tile_depth_sum', torch.zeros(num_tiles))  # For analysis
    
    def forward(self, x, state, true_depth=None):
        """
        Args:
            x: (batch, d_input)
            state: (batch, d_state)
            true_depth: Optional depth for tracking
        
        Returns:
            new_state: (batch, d_state)
            tile_idx: (batch,)
        """
        batch_size = x.shape[0]
        
        # Route based on (input, state)
        combined = torch.cat([x, state], dim=-1)
        scores = combined @ self.signatures.T
        tile_idx = scores.argmax(dim=-1)
        
        # Shared state update
        base_state = self.state_update(x, state)
        
        # Apply tile-specific modulation (differentiable via soft weights during training)
        if self.training:
            weights = F.softmax(scores * 2, dim=-1)  # Sharper
            gate = (weights.unsqueeze(-1) * self.tile_gates.unsqueeze(0)).sum(dim=1)
            bias = (weights.unsqueeze(-1) * self.tile_biases.unsqueeze(0)).sum(dim=1)
        else:
            gate = self.tile_gates[tile_idx]
            bias = self.tile_biases[tile_idx]
        
        new_state = base_state * torch.sigmoid(gate) + bias
        
        # Track
        if true_depth is not None:
            with torch.no_grad():
                for b in range(batch_size):
                    t = tile_idx[b].item()
                    self.tile_counts[t] += 1
                    self.tile_depth_sum[t] += true_depth[b].item()
        
        return new_state, tile_idx
    
    def get_tile_avg_depth(self):
        """Get average depth per tile."""
        counts = self.tile_counts.clamp(min=1)
        return (self.tile_depth_sum / counts).cpu().numpy()
    
    def reset_tracking(self):
        self.tile_counts.zero_()
        self.tile_depth_sum.zero_()


class DepthCounter(nn.Module):
    """Count bracket depth using temporal tiles."""
    
    def __init__(self, d_model=16, d_state=8, num_tiles=6, max_depth=4):
        super().__init__()
        
        self.embed = nn.Embedding(2, d_model)
        self.temporal = SimpleTemporalTiles(d_model, d_state, num_tiles)
        self.depth_head = nn.Linear(d_state, max_depth + 1)
        
        self.d_state = d_state
        self.num_tiles = num_tiles
        self.max_depth = max_depth
    
    def forward(self, tokens, depths=None):
        """
        Args:
            tokens: (batch, seq_len)
            depths: Optional (batch, seq_len) for tracking
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        state = torch.zeros(batch_size, self.d_state, device=device)
        
        all_logits = []
        all_tiles = []
        
        for t in range(seq_len):
            x = self.embed(tokens[:, t])
            depth_t = depths[:, t] if depths is not None else None
            state, tile_idx = self.temporal(x, state, depth_t)
            logits = self.depth_head(state)
            
            all_logits.append(logits)
            all_tiles.append(tile_idx)
        
        return torch.stack(all_logits, dim=1), all_tiles


# =============================================================================
# Data & Training (inline for speed)
# =============================================================================

def generate_data(n, max_depth=4, max_len=10):
    data = []
    for _ in range(n):
        length = np.random.randint(4, max_len + 1)
        tokens, depths = [], []
        d = 0
        for _ in range(length):
            if d == 0:
                t = 0
            elif d == max_depth:
                t = 1
            else:
                t = np.random.randint(0, 2)
            tokens.append(t)
            d = d + 1 if t == 0 else d - 1
            depths.append(d)
        data.append((tokens, depths))
    return data


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    MAX_DEPTH = 4
    NUM_TILES = 6
    
    # Generate data
    print("\n[1] Generating data...")
    train_data = generate_data(2000, MAX_DEPTH, 10)
    test_data = generate_data(500, MAX_DEPTH, 10)
    
    # Model
    model = DepthCounter(d_model=16, d_state=8, num_tiles=NUM_TILES, max_depth=MAX_DEPTH).to(device)
    print(f"[2] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    print("\n[3] Training...")
    
    for epoch in range(100):
        model.train()
        model.temporal.reset_tracking()
        
        # Shuffle and batch
        np.random.shuffle(train_data)
        total_loss = 0
        
        for i in range(0, len(train_data), 64):
            batch = train_data[i:i+64]
            max_len = max(len(t) for t, _ in batch)
            
            tokens = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            depths = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            mask = torch.zeros(len(batch), max_len, dtype=torch.bool, device=device)
            
            for j, (t, d) in enumerate(batch):
                tokens[j, :len(t)] = torch.tensor(t)
                depths[j, :len(d)] = torch.tensor(d)
                mask[j, :len(t)] = True
            
            logits, _ = model(tokens, depths)
            loss = F.cross_entropy(logits[mask], depths[mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Eval every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for i in range(0, len(test_data), 64):
                    batch = test_data[i:i+64]
                    max_len = max(len(t) for t, _ in batch)
                    
                    tokens = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
                    depths = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
                    mask = torch.zeros(len(batch), max_len, dtype=torch.bool, device=device)
                    
                    for j, (t, d) in enumerate(batch):
                        tokens[j, :len(t)] = torch.tensor(t)
                        depths[j, :len(d)] = torch.tensor(d)
                        mask[j, :len(t)] = True
                    
                    logits, _ = model(tokens)
                    preds = logits.argmax(dim=-1)
                    correct += ((preds == depths) & mask).sum().item()
                    total += mask.sum().item()
            
            acc = correct / total
            avg_depths = model.temporal.get_tile_avg_depth()
            depth_str = ' '.join(f'T{i}:{d:.1f}' for i, d in enumerate(avg_depths) if model.temporal.tile_counts[i] > 10)
            print(f"  Epoch {epoch+1}: acc={acc:.1%}  {depth_str}")
    
    # Final analysis
    print("\n[4] Final test...")
    model.eval()
    
    # Test specific sequences
    test_seqs = [
        ([0, 1], [1, 0]),                    # ()
        ([0, 0, 1, 1], [1, 2, 1, 0]),        # (())
        ([0, 1, 0, 1], [1, 0, 1, 0]),        # ()()
        ([0, 0, 0, 1, 1, 1], [1, 2, 3, 2, 1, 0]),  # ((()))
    ]
    
    print("\nTest sequences:")
    with torch.no_grad():
        for tokens, true_depths in test_seqs:
            t = torch.tensor([tokens], device=device)
            logits, tiles = model(t)
            preds = logits.argmax(dim=-1)[0].tolist()
            tile_seq = [tile[0].item() for tile in tiles]
            
            seq_str = ''.join('(' if x == 0 else ')' for x in tokens)
            match = '✓' if preds == true_depths else '✗'
            
            print(f"  {seq_str:10s} true={true_depths} pred={preds} {match}")
            print(f"           tiles={tile_seq}")
    
    # Tile summary
    print("\nTile-depth correspondence:")
    avg_depths = model.temporal.get_tile_avg_depth()
    counts = model.temporal.tile_counts.cpu().numpy()
    for i in range(NUM_TILES):
        if counts[i] > 10:
            print(f"  Tile {i}: avg_depth={avg_depths[i]:.2f}, count={int(counts[i])}")
    
    print("\n" + "=" * 50)
    correct, total = 0, 0
    with torch.no_grad():
        for tokens, depths in test_data:
            t = torch.tensor([tokens], device=device)
            d = torch.tensor([depths], device=device)
            logits, _ = model(t)
            preds = logits.argmax(dim=-1)
            correct += (preds == d).sum().item()
            total += len(tokens)
    
    final_acc = correct / total
    if final_acc > 0.95:
        print(f"✓ SUCCESS: {final_acc:.1%} - Temporal tiles learned to count!")
    elif final_acc > 0.80:
        print(f"~ PARTIAL: {final_acc:.1%} - Learning but incomplete")
    else:
        print(f"✗ NEEDS WORK: {final_acc:.1%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
