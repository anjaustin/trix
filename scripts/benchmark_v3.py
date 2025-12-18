#!/usr/bin/env python3
"""
Benchmark: SparseLookupFFN v1 vs v2 vs v3 (Geometric TriX)

Test on TinyShakespeare character-level language modeling.
Measures:
- Validation perplexity
- Parameter count
- Training speed
- Routing patterns (diagonality for v3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import math
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trix.nn.sparse_lookup import SparseLookupFFN
from trix.nn.sparse_lookup_v3 import SparseLookupFFNv3


# === Data ===

def download_shakespeare():
    """Download TinyShakespeare if needed."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    path = os.path.join(data_dir, 'shakespeare.txt')
    if not os.path.exists(path):
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f'Downloading TinyShakespeare...')
        urllib.request.urlretrieve(url, path)
    
    with open(path, 'r') as f:
        text = f.read()
    return text


class CharDataset(Dataset):
    def __init__(self, text, seq_len=128):
        self.seq_len = seq_len
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
    
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# === Model ===

class SimpleTransformer(nn.Module):
    """Minimal transformer for benchmarking FFN variants."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ffn_module, max_seq_len=512):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True),
                'ffn': ffn_module(),
            }))
        
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, positions=None):
        B, T = x.shape
        device = x.device
        
        # Embeddings
        tok_emb = self.embedding(x)
        if positions is None:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        h = tok_emb + pos_emb
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        
        total_aux = 0.0
        routing_infos = []
        
        # Transformer layers
        for layer in self.layers:
            # Attention
            h_norm = layer['ln1'](h)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm, attn_mask=mask)
            h = h + attn_out
            
            # FFN (with routing info)
            ffn_out, routing_info, aux_losses = layer['ffn'](h, positions.float())
            h = ffn_out  # FFN already does residual
            total_aux = total_aux + aux_losses['total_aux']
            routing_infos.append(routing_info)
        
        # Output
        h = self.ln_final(h)
        logits = self.head(h)
        
        return logits, total_aux, routing_infos


# === Training ===

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, aux_loss, _ = model(x)
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss_batch = loss + aux_loss
        
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        
        if batch_idx >= 100:  # Limit batches per epoch for speed
            break
    
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        
        if batch_idx >= 50:
            break
    
    return math.exp(total_loss / total_tokens)  # Perplexity


def measure_diagonality(model, dataloader, device, num_batches=10):
    """Measure routing diagonality (wave pattern)."""
    model.eval()
    
    all_tiles = []
    all_positions = []
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            x = x.to(device)
            B, T = x.shape
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            
            _, _, routing_infos = model(x, positions)
            
            # Collect from first layer
            tile_idx = routing_infos[0]['tile_idx']  # [B, T]
            
            all_tiles.append(tile_idx.cpu())
            all_positions.append(positions.cpu())
    
    tiles = torch.cat(all_tiles, dim=0).float()  # [total_B, T]
    positions = torch.cat(all_positions, dim=0).float()
    
    # Compute correlation between position and tile index
    tiles_flat = tiles.flatten()
    positions_flat = positions.flatten()
    
    # Normalize
    tiles_norm = (tiles_flat - tiles_flat.mean()) / (tiles_flat.std() + 1e-8)
    positions_norm = (positions_flat - positions_flat.mean()) / (positions_flat.std() + 1e-8)
    
    correlation = (tiles_norm * positions_norm).mean().item()
    
    return max(0, correlation)


# === Main ===

def main():
    print("=" * 70)
    print("BENCHMARK: SparseLookupFFN v1 vs v3 (Geometric TriX)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Hyperparameters (reduced for speed)
    d_model = 64
    n_heads = 4
    n_layers = 2
    num_tiles = 16
    batch_size = 32
    seq_len = 64
    max_seq_len = 256
    n_epochs = 3
    lr = 3e-4
    
    # Data
    text = download_shakespeare()
    dataset = CharDataset(text, seq_len)
    
    n_train = int(0.9 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    vocab_size = dataset.vocab_size
    print(f"Vocab size: {vocab_size}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print()
    
    results = []
    
    # === V1: Original SparseLookupFFN ===
    print("-" * 70)
    print("Training: SparseLookupFFN v1 (Content-only routing)")
    print("-" * 70)
    
    def make_v1():
        return SparseLookupFFN(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
        )
    
    model_v1 = SimpleTransformer(vocab_size, d_model, n_heads, n_layers, make_v1, max_seq_len).to(device)
    
    # Wrap forward to match expected signature
    original_forward = model_v1.layers[0]['ffn'].forward
    def wrapped_forward(x, positions=None):
        return original_forward(x)
    for layer in model_v1.layers:
        layer['ffn'].forward = lambda x, pos=None, f=layer['ffn']: f.__class__.forward(f, x)
    
    param_count_v1 = sum(p.numel() for p in model_v1.parameters())
    print(f"Parameters: {param_count_v1:,}")
    
    optimizer_v1 = torch.optim.AdamW(model_v1.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss = train_epoch(model_v1, train_loader, optimizer_v1, device)
        val_ppl = evaluate(model_v1, val_loader, device)
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_ppl={val_ppl:.2f}")
    
    v1_time = time.time() - start_time
    v1_ppl = evaluate(model_v1, val_loader, device)
    print(f"Final PPL: {v1_ppl:.2f}, Time: {v1_time:.1f}s")
    
    results.append(('v1 (Content)', param_count_v1, v1_ppl, v1_time, 0.0))
    
    # === V3: Geometric TriX ===
    print()
    print("-" * 70)
    print("Training: SparseLookupFFNv3 (Geometric TriX)")
    print("-" * 70)
    
    def make_v3():
        return SparseLookupFFNv3(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=4,
            max_seq_len=max_seq_len,
            position_spread=2.0,
            use_gauge=True,
            use_vortex=True,
            track_topology=True,
        )
    
    model_v3 = SimpleTransformer(vocab_size, d_model, n_heads, n_layers, make_v3, max_seq_len).to(device)
    param_count_v3 = sum(p.numel() for p in model_v3.parameters())
    print(f"Parameters: {param_count_v3:,}")
    
    optimizer_v3 = torch.optim.AdamW(model_v3.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss = train_epoch(model_v3, train_loader, optimizer_v3, device)
        val_ppl = evaluate(model_v3, val_loader, device)
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_ppl={val_ppl:.2f}")
    
    v3_time = time.time() - start_time
    v3_ppl = evaluate(model_v3, val_loader, device)
    v3_diag = measure_diagonality(model_v3, val_loader, device)
    print(f"Final PPL: {v3_ppl:.2f}, Time: {v3_time:.1f}s, Diagonality: {v3_diag:.3f}")
    
    results.append(('v3 (Geometric)', param_count_v3, v3_ppl, v3_time, v3_diag))
    
    # === Summary ===
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>12} {'Val PPL':>10} {'Time':>10} {'Diag':>8}")
    print("-" * 70)
    
    for name, params, ppl, train_time, diag in results:
        print(f"{name:<20} {params:>12,} {ppl:>10.2f} {train_time:>9.1f}s {diag:>8.3f}")
    
    print("-" * 70)
    
    # Comparison
    v1_result = results[0]
    v3_result = results[1]
    
    ppl_improvement = (v1_result[2] - v3_result[2]) / v1_result[2] * 100
    param_overhead = (v3_result[1] - v1_result[1]) / v1_result[1] * 100
    
    print()
    print(f"PPL improvement (v3 vs v1): {ppl_improvement:+.1f}%")
    print(f"Parameter overhead: {param_overhead:+.1f}%")
    print(f"Routing diagonality: {v3_result[4]:.3f} (0=random, 1=perfect wave)")
    
    if v3_result[4] > 0.3:
        print()
        print("✓ WAVE PATTERN DETECTED - Geometry encoded in routing!")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
