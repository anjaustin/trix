#!/usr/bin/env python3
"""
Benchmark: HierarchicalTriXFFN vs HybridKANFFN

Compares the two FFN approaches on TinyShakespeare char-level LM.
Tracks: perplexity, routing entropy, tile utilization, training stability.

Usage:
    python scripts/benchmark_ffn.py
"""

import os
import sys
import json
import math
import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix.nn.hierarchical import HierarchicalTriXFFN
from trix.nn.hybrid_kan import HybridKANFFN
from trix.nn.sparse_lookup import SparseLookupFFN


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Data
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path: str = "data/tinyshakespeare.txt"
    train_split: float = 0.9
    context_length: int = 128
    max_train_samples: int = 50000  # Limit training samples for faster iteration
    
    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    num_tiles: int = 16
    tiles_per_cluster: int = 4
    dropout: float = 0.1
    
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 100
    grad_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 1
    eval_batches: int = 50
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# =============================================================================
# Data
# =============================================================================

def download_data(config: Config) -> str:
    """Download TinyShakespeare if not present."""
    path = Path(config.data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not path.exists():
        print(f"Downloading TinyShakespeare to {path}...")
        urllib.request.urlretrieve(config.data_url, path)
    
    with open(path, 'r') as f:
        text = f.read()
    
    print(f"Loaded {len(text):,} characters")
    return text


class CharDataset(Dataset):
    """Character-level dataset."""
    
    def __init__(self, text: str, context_length: int, char_to_idx: Dict[str, int]):
        self.data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
        self.context_length = context_length
    
    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, int, Dict]:
    """Create train and val dataloaders."""
    text = download_data(config)
    
    # Build vocabulary
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Split
    split_idx = int(len(text) * config.train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_dataset = CharDataset(train_text, config.context_length, char_to_idx)
    val_dataset = CharDataset(val_text, config.context_length, char_to_idx)
    
    # Limit training data for faster iteration
    if config.max_train_samples and len(train_dataset) > config.max_train_samples:
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            range(config.max_train_samples)
        )
        print(f"Limited training to {config.max_train_samples:,} samples")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, vocab_size, {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}


# =============================================================================
# Model
# =============================================================================

class TransformerBlock(nn.Module):
    """Transformer block with pluggable FFN."""
    
    def __init__(self, d_model: int, n_heads: int, ffn: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, is_causal: bool = True) -> Tuple[torch.Tensor, Dict]:
        # Attention
        B, T, D = x.shape
        x_norm = self.ln1(x)
        
        # Create causal mask
        if is_causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        x = x + self.dropout(attn_out)
        
        # FFN
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        
        # Handle different FFN return types
        aux_losses = {}
        if isinstance(ffn_out, tuple):
            if len(ffn_out) == 3:  # HierarchicalTriXFFN returns (out, routing_info, aux_losses)
                ffn_out, routing_info, aux_losses = ffn_out
            elif len(ffn_out) == 2:
                ffn_out, aux_losses = ffn_out
        
        x = x + self.dropout(ffn_out)
        
        return x, aux_losses


class CharTransformer(nn.Module):
    """Character-level transformer with pluggable FFN type."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_type: str,  # 'hierarchical' or 'hybrid_kan'
        num_tiles: int,
        tiles_per_cluster: int,
        dropout: float,
        context_length: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.ffn_type = ffn_type
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Create FFN layers based on type
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            if ffn_type == 'hierarchical':
                ffn = HierarchicalTriXFFN(
                    d_model=d_model,
                    num_tiles=num_tiles,
                    tiles_per_cluster=tiles_per_cluster,
                    dropout=dropout,
                )
            elif ffn_type == 'hybrid_kan':
                ffn = HybridKANFFN(
                    d_model=d_model,
                    num_tiles=num_tiles,
                    tiles_per_cluster=tiles_per_cluster,
                    dropout=dropout,
                )
            elif ffn_type == 'sparse_lookup':
                ffn = SparseLookupFFN(
                    d_model=d_model,
                    num_tiles=num_tiles,
                    tiles_per_cluster=tiles_per_cluster,
                    ternary_splines=True,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown ffn_type: {ffn_type}")
            
            block = TransformerBlock(d_model, n_heads, ffn, dropout)
            self.blocks.append(block)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T = x.shape
        device = x.device
        
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)
        
        total_aux = {}
        for block in self.blocks:
            x, aux = block(x, is_causal=True)
            for k, v in aux.items():
                if k not in total_aux:
                    total_aux[k] = 0
                total_aux[k] = total_aux[k] + v
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits, total_aux
    
    def get_routing_stats(self) -> Dict:
        """Collect routing statistics from all FFN layers."""
        stats = {'layers': []}
        
        for i, block in enumerate(self.blocks):
            ffn = block.ffn
            # Try both method names
            if hasattr(ffn, 'get_stats'):
                layer_stats = ffn.get_stats()
                layer_stats['layer'] = i
                stats['layers'].append(layer_stats)
            elif hasattr(ffn, 'get_routing_stats'):
                layer_stats = ffn.get_routing_stats()
                layer_stats['layer'] = i
                stats['layers'].append(layer_stats)
        
        # Aggregate
        if stats['layers']:
            all_usage = []
            for layer in stats['layers']:
                if 'usage_mean' in layer:
                    all_usage.append(layer['usage_mean'])
            
            stats['mean_usage'] = sum(all_usage) / len(all_usage) if all_usage else 0
            stats['active_tiles'] = sum(l.get('active_tiles', 0) for l in stats['layers'])
        
        return stats


# =============================================================================
# Training
# =============================================================================

@dataclass
class Metrics:
    """Training metrics for one epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_ppl: float
    aux_loss: float
    routing_entropy: float
    active_tiles: int
    time_seconds: float
    
    def to_dict(self):
        return asdict(self)


def compute_routing_entropy(model: CharTransformer) -> float:
    """Compute entropy of tile usage distribution."""
    stats = model.get_routing_stats()
    
    if not stats['layers']:
        return 0.0
    
    # Get usage rates from all layers
    all_usage = []
    for layer in stats['layers']:
        if hasattr(model.blocks[layer['layer']].ffn, 'tiles'):
            tiles = model.blocks[layer['layer']].ffn.tiles
            for tile in tiles:
                rate = tile.usage_rate
                if rate > 0:
                    all_usage.append(rate)
    
    if not all_usage:
        return 0.0
    
    # Normalize to distribution
    total = sum(all_usage)
    if total == 0:
        return 0.0
    
    probs = [u / total for u in all_usage]
    
    # Compute entropy
    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
    
    # Normalize by max entropy
    max_entropy = math.log(len(probs))
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def train_epoch(
    model: CharTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_aux = 0
    n_batches = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(config.device), y.to(config.device)
        
        optimizer.zero_grad()
        
        logits, aux_losses = model(x)
        
        # Main loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Aux loss
        aux = aux_losses.get('total_aux', 0)
        if isinstance(aux, torch.Tensor):
            aux = aux.mean()
        
        total = loss + aux
        
        total.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_aux += aux.item() if isinstance(aux, torch.Tensor) else aux
        n_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / n_batches, total_aux / n_batches


@torch.no_grad()
def evaluate(model: CharTransformer, loader: DataLoader, config: Config, max_batches: int = None) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        
        x, y = x.to(config.device), y.to(config.device)
        
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)
    
    return avg_loss, ppl


def train_model(
    model: CharTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    name: str,
) -> List[Metrics]:
    """Full training loop."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    model.to(config.device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    total_steps = len(train_loader) * config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    history = []
    
    for epoch in range(config.epochs):
        start_time = time.time()
        
        train_loss, aux_loss = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
        val_loss, val_ppl = evaluate(model, val_loader, config, config.eval_batches)
        
        # Routing metrics
        routing_entropy = compute_routing_entropy(model)
        stats = model.get_routing_stats()
        active_tiles = stats.get('active_tiles', 0)
        
        elapsed = time.time() - start_time
        
        metrics = Metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_ppl=val_ppl,
            aux_loss=aux_loss,
            routing_entropy=routing_entropy,
            active_tiles=active_tiles,
            time_seconds=elapsed,
        )
        history.append(metrics)
        
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val PPL:    {val_ppl:.2f}")
        print(f"  Aux Loss:   {aux_loss:.4f}")
        print(f"  Routing H:  {routing_entropy:.3f}")
        print(f"  Active:     {active_tiles} tiles")
        print(f"  Time:       {elapsed:.1f}s")
    
    return history


# =============================================================================
# Main
# =============================================================================

def run_benchmark(config: Config) -> Dict:
    """Run full benchmark comparison."""
    print("="*60)
    print("  TriX FFN Benchmark")
    print("  HierarchicalTriXFFN vs HybridKANFFN")
    print("="*60)
    
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load data
    train_loader, val_loader, vocab_size, vocab = create_dataloaders(config)
    
    results = {}
    
    # Train HierarchicalTriXFFN
    model_hier = CharTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_type='hierarchical',
        num_tiles=config.num_tiles,
        tiles_per_cluster=config.tiles_per_cluster,
        dropout=config.dropout,
        context_length=config.context_length,
    )
    
    history_hier = train_model(model_hier, train_loader, val_loader, config, "HierarchicalTriXFFN")
    results['hierarchical'] = {
        'history': [m.to_dict() for m in history_hier],
        'final_ppl': history_hier[-1].val_ppl,
        'final_loss': history_hier[-1].val_loss,
        'params': sum(p.numel() for p in model_hier.parameters()),
    }
    
    # Reset seed for fair comparison
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Train HybridKANFFN
    model_kan = CharTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_type='hybrid_kan',
        num_tiles=config.num_tiles,
        tiles_per_cluster=config.tiles_per_cluster,
        dropout=config.dropout,
        context_length=config.context_length,
    )
    
    history_kan = train_model(model_kan, train_loader, val_loader, config, "HybridKANFFN")
    results['hybrid_kan'] = {
        'history': [m.to_dict() for m in history_kan],
        'final_ppl': history_kan[-1].val_ppl,
        'final_loss': history_kan[-1].val_loss,
        'params': sum(p.numel() for p in model_kan.parameters()),
    }
    
    # Summary
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<25} {'Params':>12} {'Val Loss':>12} {'Val PPL':>12}")
    print("-"*60)
    
    for name, data in results.items():
        print(f"{name:<25} {data['params']:>12,} {data['final_loss']:>12.4f} {data['final_ppl']:>12.2f}")
    
    # Comparison
    hier_ppl = results['hierarchical']['final_ppl']
    kan_ppl = results['hybrid_kan']['final_ppl']
    
    diff = (kan_ppl - hier_ppl) / hier_ppl * 100
    
    print(f"\nHybridKAN vs Hierarchical: {diff:+.2f}% PPL")
    
    if kan_ppl < hier_ppl:
        print(">>> HybridKANFFN wins!")
    else:
        print(">>> HierarchicalTriXFFN wins!")
    
    # Save results
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)
    
    with open(results_path / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path / 'benchmark_results.json'}")
    
    return results


if __name__ == "__main__":
    config = Config()
    
    # Adjust for available resources
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, reducing batch size")
        config.batch_size = 32
        config.epochs = 5
    
    results = run_benchmark(config)
