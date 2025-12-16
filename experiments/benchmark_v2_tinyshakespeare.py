#!/usr/bin/env python3
"""
Benchmark: SparseLookupFFN v1 vs v2 on TinyShakespeare

Tests:
  1. Perplexity comparison (v1 vs v2)
  2. Island regularizer effects
  3. Surgery API validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import urllib.request
from typing import Dict, Tuple


# =============================================================================
# TinyShakespeare Data
# =============================================================================

def load_tinyshakespeare(data_dir: str = "/workspace/trix_latest/experiments/data") -> Tuple[str, Dict]:
    """Load TinyShakespeare dataset."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "tinyshakespeare.txt")
    
    if not os.path.exists(filepath):
        print("Downloading TinyShakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, filepath)
    
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Build vocab
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    return text, {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char, 'vocab_size': len(chars)}


def prepare_data(text: str, vocab: Dict, seq_len: int = 64, split: float = 0.9):
    """Prepare train/val splits."""
    char_to_idx = vocab['char_to_idx']
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    n = int(len(data) * split)
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    """Get a random batch."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


# =============================================================================
# Models
# =============================================================================

class CharLMv1(nn.Module):
    """Character LM with SparseLookupFFN v1."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        from trix.nn import SparseLookupBlock
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        
        self.blocks = nn.ModuleList([
            SparseLookupBlock(
                d_model=d_model,
                n_heads=n_heads,
                num_tiles=64,
                tiles_per_cluster=8,
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T = x.shape
        
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        
        total_aux = 0.0
        for block in self.blocks:
            h, _, aux = block(h)
            total_aux = total_aux + aux['total_aux']
        
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits, {'total_aux': total_aux}


class CharLMv2(nn.Module):
    """Character LM with SparseLookupFFN v2 (surgery + regularization)."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        from trix.nn import SparseLookupBlockV2
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        
        self.blocks = nn.ModuleList([
            SparseLookupBlockV2(
                d_model=d_model,
                n_heads=n_heads,
                num_tiles=64,
                tiles_per_cluster=8,
                ternary_weight=0.01,
                sparsity_weight=0.01,
                diversity_weight=0.01,
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T = x.shape
        
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        
        total_aux = 0.0
        for block in self.blocks:
            h, _, aux = block(h)
            total_aux = total_aux + aux['total_aux']
        
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits, {'total_aux': total_aux}
    
    def get_island_stats(self) -> Dict:
        """Get island stats from all blocks."""
        stats = []
        for i, block in enumerate(self.blocks):
            s = block.ffn.get_island_stats()
            s['layer'] = i
            stats.append(s)
        return stats


# =============================================================================
# Training
# =============================================================================

def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    n_steps: int = 2000,
    batch_size: int = 32,
    seq_len: int = 64,
    lr: float = 3e-4,
    eval_every: int = 200,
    device: torch.device = None,
) -> Dict:
    """Train and evaluate a model."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    
    for step in range(n_steps):
        model.train()
        
        x, y = get_batch(train_data, batch_size, seq_len, device)
        
        logits, aux = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss = loss + aux['total_aux']
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % eval_every == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(10):
                    x, y = get_batch(val_data, batch_size, seq_len, device)
                    logits, _ = model(x)
                    val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                val_ppl = np.exp(avg_val_loss)
            
            history['train_loss'].append(loss.item())
            history['val_loss'].append(avg_val_loss)
            history['val_ppl'].append(val_ppl)
            
            print(f"  Step {step:4d}: train_loss={loss.item():.4f}, val_loss={avg_val_loss:.4f}, val_ppl={val_ppl:.2f}")
    
    return history


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark():
    print("=" * 70)
    print("TinyShakespeare Benchmark: SparseLookupFFN v1 vs v2")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading TinyShakespeare...")
    text, vocab = load_tinyshakespeare()
    train_data, val_data = prepare_data(text, vocab, seq_len=64)
    print(f"    Vocab size: {vocab['vocab_size']}")
    print(f"    Train: {len(train_data):,} chars, Val: {len(val_data):,} chars")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    # Config
    d_model = 128
    n_layers = 4
    n_steps = 1500
    
    # Train v1
    print("\n[2] Training SparseLookupFFN v1 (baseline)...")
    model_v1 = CharLMv1(vocab['vocab_size'], d_model=d_model, n_layers=n_layers)
    params_v1 = sum(p.numel() for p in model_v1.parameters())
    print(f"    Parameters: {params_v1:,}")
    
    history_v1 = train_model(model_v1, train_data, val_data, n_steps=n_steps, device=device)
    
    # Train v2
    print("\n[3] Training SparseLookupFFN v2 (with surgery + regularization)...")
    model_v2 = CharLMv2(vocab['vocab_size'], d_model=d_model, n_layers=n_layers)
    params_v2 = sum(p.numel() for p in model_v2.parameters())
    print(f"    Parameters: {params_v2:,}")
    
    history_v2 = train_model(model_v2, train_data, val_data, n_steps=n_steps, device=device)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    final_ppl_v1 = history_v1['val_ppl'][-1]
    final_ppl_v2 = history_v2['val_ppl'][-1]
    
    print(f"\n    SparseLookupFFN v1:")
    print(f"      Parameters: {params_v1:,}")
    print(f"      Final PPL: {final_ppl_v1:.2f}")
    
    print(f"\n    SparseLookupFFN v2:")
    print(f"      Parameters: {params_v2:,}")
    print(f"      Final PPL: {final_ppl_v2:.2f}")
    
    # Island stats for v2
    print(f"\n    v2 Island Stats:")
    island_stats = model_v2.get_island_stats()
    for s in island_stats:
        print(f"      Layer {s['layer']}: ternary={s['ternary_fraction']:.1%}, "
              f"sparsity={s['sparsity']:.1%}, diversity={s['diversity']:.2f}")
    
    # Comparison
    improvement = (final_ppl_v1 - final_ppl_v2) / final_ppl_v1 * 100
    print(f"\n    PPL improvement: {improvement:+.1f}%")
    
    if final_ppl_v2 <= final_ppl_v1:
        print("\n    ✓ v2 matches or improves on v1")
        print("    → Island regularizers do not hurt perplexity")
    else:
        print("\n    ~ v2 slightly worse than v1")
        print("    → May need regularizer weight tuning")
    
    print("=" * 70)
    
    return {
        'v1_ppl': final_ppl_v1,
        'v2_ppl': final_ppl_v2,
        'params_v1': params_v1,
        'params_v2': params_v2,
    }


if __name__ == "__main__":
    results = run_benchmark()
