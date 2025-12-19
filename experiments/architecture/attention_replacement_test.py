#!/usr/bin/env python3
"""
Attention Replacement Test: Can routed memory replace attention?

Compares:
  1. Standard Transformer (with attention)
  2. RoutedMemoryTransformer (attention replaced)

On TinyShakespeare character-level language modeling.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import urllib.request
import time
from typing import Dict

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

from trix.nn.routed_memory import RoutedMemoryTransformer, StandardTransformer


def load_data():
    """Load TinyShakespeare."""
    data_dir = '/workspace/trix_latest/experiments/data'
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'tinyshakespeare.txt')
    
    if not os.path.exists(filepath):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, filepath)
    
    with open(filepath, 'r') as f:
        text = f.read()
    
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    n = int(len(data) * 0.9)
    
    return data[:n], data[n:], vocab_size, char_to_idx


def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


def train_and_eval(model, train_data, val_data, vocab_size, device, n_steps=1000, name="Model"):
    """Train and evaluate a model."""
    print(f"\n  Training {name}...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model = model.to(device)
    
    batch_size = 32
    seq_len = 64
    
    start_time = time.time()
    
    for step in range(n_steps):
        model.train()
        x, y = get_batch(train_data, batch_size, seq_len, device)
        
        # Forward
        if hasattr(model, 'forward') and 'RoutedMemory' in model.__class__.__name__:
            logits, _ = model(x)
        else:
            logits = model(x)
        
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(5):
                    x, y = get_batch(val_data, batch_size, seq_len, device)
                    if hasattr(model, 'forward') and 'RoutedMemory' in model.__class__.__name__:
                        logits, _ = model(x)
                    else:
                        logits = model(x)
                    val_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                    val_losses.append(val_loss.item())
                
                avg_loss = np.mean(val_losses)
                ppl = np.exp(avg_loss)
            
            print(f"    Step {step:4d}: loss={loss.item():.3f}, val_loss={avg_loss:.3f}, ppl={ppl:.1f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in range(20):
            x, y = get_batch(val_data, batch_size, seq_len, device)
            if hasattr(model, 'forward') and 'RoutedMemory' in model.__class__.__name__:
                logits, _ = model(x)
            else:
                logits = model(x)
            val_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            val_losses.append(val_loss.item())
        
        final_loss = np.mean(val_losses)
        final_ppl = np.exp(final_loss)
    
    elapsed = time.time() - start_time
    
    return {
        'final_loss': final_loss,
        'final_ppl': final_ppl,
        'time': elapsed,
    }


def main():
    print("=" * 70)
    print("ATTENTION REPLACEMENT TEST")
    print("Can routed memory replace attention?")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading TinyShakespeare...")
    train_data, val_data, vocab_size, _ = load_data()
    print(f"    Vocab: {vocab_size}, Train: {len(train_data):,}, Val: {len(val_data):,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    # Config
    d_model = 128
    n_layers = 4
    n_heads = 4
    n_slots = 64  # Same as n_heads * 16
    n_steps = 1000
    
    # Standard transformer
    print("\n[2] Building models...")
    
    standard = StandardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    standard_params = sum(p.numel() for p in standard.parameters())
    print(f"    Standard Transformer: {standard_params:,} params")
    
    routed = RoutedMemoryTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_slots=n_slots,
        n_heads=n_heads,
        top_k=1,  # Hard routing
    )
    routed_params = sum(p.numel() for p in routed.parameters())
    print(f"    RoutedMemory Transformer: {routed_params:,} params")
    print(f"    Param ratio: {routed_params/standard_params:.2f}x")
    
    # Train both
    print("\n[3] Training...")
    
    results_standard = train_and_eval(
        standard, train_data, val_data, vocab_size, device, 
        n_steps=n_steps, name="Standard Transformer"
    )
    
    results_routed = train_and_eval(
        routed, train_data, val_data, vocab_size, device,
        n_steps=n_steps, name="RoutedMemory Transformer"
    )
    
    # Check routing stats
    print("\n[4] Routing Stats...")
    stats = routed.get_all_routing_stats()
    for block_name, block_stats in stats.items():
        active = block_stats.get('active_slots_per_head', [])
        print(f"    {block_name}: active slots per head = {active}")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n    Standard Transformer:")
    print(f"      Parameters: {standard_params:,}")
    print(f"      Final PPL: {results_standard['final_ppl']:.2f}")
    print(f"      Time: {results_standard['time']:.1f}s")
    
    print(f"\n    RoutedMemory Transformer:")
    print(f"      Parameters: {routed_params:,}")
    print(f"      Final PPL: {results_routed['final_ppl']:.2f}")
    print(f"      Time: {results_routed['time']:.1f}s")
    
    # Analysis
    ppl_diff = results_routed['final_ppl'] - results_standard['final_ppl']
    ppl_diff_pct = ppl_diff / results_standard['final_ppl'] * 100
    
    print(f"\n    PPL difference: {ppl_diff:+.2f} ({ppl_diff_pct:+.1f}%)")
    
    if abs(ppl_diff_pct) < 10:
        print("\n    ✓ ROUTED MEMORY VIABLE")
        print("    → Comparable perplexity to standard attention")
        print("    → O(n·M) complexity instead of O(n²)")
    elif ppl_diff_pct < 0:
        print("\n    ✓ ROUTED MEMORY BETTER")
        print("    → Lower perplexity than standard attention!")
    else:
        print("\n    ~ ROUTED MEMORY WORSE")
        print(f"    → {ppl_diff_pct:.1f}% higher perplexity")
        print("    → May need more slots or different routing")
    
    print("=" * 70)
    
    return {
        'standard': results_standard,
        'routed': results_routed,
    }


if __name__ == "__main__":
    results = main()
