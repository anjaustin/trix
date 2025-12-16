#!/usr/bin/env python3
"""
Thor Hardware Benchmark for TriX Training

Measures:
1. Memory usage for different model sizes
2. Training throughput (tokens/sec) for various batch sizes
3. TriX SparseLookupFFN vs standard FFN performance
4. Optimal configurations for 5M/50M/500M token training runs

Target: Find optimal batch_size, seq_len, and model config for Thor's 122GB memory.
"""

import sys
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix import SparseLookupFFN, SparseLookupBlock


# =============================================================================
# Model Definitions
# =============================================================================

class StandardFFN(nn.Module):
    """Standard transformer FFN for comparison."""
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x + residual


class StandardBlock(nn.Module):
    """Standard transformer block for comparison."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = StandardFFN(d_model)
    
    def forward(self, x, is_causal=True):
        # Attention
        h = self.ln1(x)
        mask = None
        if is_causal:
            seq_len = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, is_causal=is_causal)
        x = x + attn_out
        # FFN
        x = self.ffn(x)
        return x


class MiniTransformer(nn.Module):
    """Minimal transformer for benchmarking."""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, 
                 use_trix: bool = False, num_tiles: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)  # Max seq len
        
        if use_trix:
            self.blocks = nn.ModuleList([
                SparseLookupBlock(d_model, n_heads, num_tiles=num_tiles)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                StandardBlock(d_model, n_heads)
                for _ in range(n_layers)
            ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.use_trix = use_trix
    
    def forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        
        total_aux = 0.0
        for block in self.blocks:
            if self.use_trix:
                x, _, aux = block(x, is_causal=True)
                total_aux = total_aux + aux.get('total_aux', 0.0)
            else:
                x = block(x, is_causal=True)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, total_aux
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Benchmark Functions
# =============================================================================

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def benchmark_throughput(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_steps: int = 20,
    warmup_steps: int = 5,
    device: str = "cuda"
) -> dict:
    """
    Benchmark training throughput.
    
    Returns:
        dict with tokens_per_sec, memory_mb, step_time_ms
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(warmup_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits, aux = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), x.view(-1)) + aux * 0.01
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    memory_before = get_memory_mb()
    start = time.perf_counter()
    
    for _ in range(num_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits, aux = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), x.view(-1)) + aux * 0.01
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    memory_peak = get_memory_mb()
    
    total_tokens = batch_size * seq_len * num_steps
    tokens_per_sec = total_tokens / elapsed
    step_time_ms = (elapsed / num_steps) * 1000
    
    return {
        'tokens_per_sec': tokens_per_sec,
        'memory_mb': memory_peak,
        'step_time_ms': step_time_ms,
        'batch_size': batch_size,
        'seq_len': seq_len,
    }


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    vocab_size: int = 32000
    num_tiles: int = 16
    
    @property
    def approx_params(self) -> int:
        """Approximate parameter count."""
        # Rough estimate: embed + n_layers * (attn + ffn) + head
        embed = self.vocab_size * self.d_model
        attn_per_layer = 4 * self.d_model * self.d_model  # Q, K, V, O
        ffn_per_layer = 8 * self.d_model * self.d_model   # 4x expansion
        head = self.vocab_size * self.d_model
        return embed + self.n_layers * (attn_per_layer + ffn_per_layer) + head


# Model configs matched to dataset sizes (Chinchilla scaling)
CONFIGS = {
    '5M': ModelConfig('5M-tokens', d_model=128, n_heads=4, n_layers=4, num_tiles=8),      # ~1M params
    '50M': ModelConfig('50M-tokens', d_model=256, n_heads=8, n_layers=6, num_tiles=16),   # ~10M params
    '500M': ModelConfig('500M-tokens', d_model=512, n_heads=8, n_layers=12, num_tiles=32), # ~50M params
}


def run_benchmarks():
    """Run all benchmarks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}")
    print(f"Initial memory: {get_memory_mb():.1f} MB")
    print("=" * 80)
    
    results = {}
    
    for config_name, config in CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"CONFIG: {config_name}")
        print(f"  d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")
        print(f"  Approx params: {config.approx_params:,}")
        print("=" * 80)
        
        results[config_name] = {'standard': {}, 'trix': {}}
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32, 64, 128]
        seq_len = 256  # Fixed sequence length
        
        for use_trix in [False, True]:
            model_type = "trix" if use_trix else "standard"
            print(f"\n--- {model_type.upper()} FFN ---")
            
            for batch_size in batch_sizes:
                clear_memory()
                
                try:
                    model = MiniTransformer(
                        vocab_size=config.vocab_size,
                        d_model=config.d_model,
                        n_heads=config.n_heads,
                        n_layers=config.n_layers,
                        use_trix=use_trix,
                        num_tiles=config.num_tiles,
                    )
                    
                    actual_params = model.count_params()
                    
                    result = benchmark_throughput(
                        model, batch_size, seq_len, config.vocab_size,
                        num_steps=20, warmup_steps=5, device=device
                    )
                    result['params'] = actual_params
                    results[config_name][model_type][batch_size] = result
                    
                    print(f"  batch={batch_size:3d}: {result['tokens_per_sec']:,.0f} tok/s, "
                          f"{result['memory_mb']:,.0f} MB, {result['step_time_ms']:.1f} ms/step")
                    
                    del model
                    clear_memory()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  batch={batch_size:3d}: OOM")
                        clear_memory()
                        break
                    raise
    
    return results


def print_summary(results: dict):
    """Print summary with recommendations."""
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    for config_name in results:
        print(f"\n{config_name} Model:")
        
        std_results = results[config_name]['standard']
        trix_results = results[config_name]['trix']
        
        if not std_results or not trix_results:
            print("  Insufficient data")
            continue
        
        # Find best batch size for each
        best_std = max(std_results.items(), key=lambda x: x[1]['tokens_per_sec'])
        best_trix = max(trix_results.items(), key=lambda x: x[1]['tokens_per_sec'])
        
        print(f"  Standard FFN: batch={best_std[0]}, {best_std[1]['tokens_per_sec']:,.0f} tok/s")
        print(f"  TriX FFN:     batch={best_trix[0]}, {best_trix[1]['tokens_per_sec']:,.0f} tok/s")
        
        speedup = best_trix[1]['tokens_per_sec'] / best_std[1]['tokens_per_sec']
        print(f"  Speedup: {speedup:.2f}x")
        
        # Training time estimate
        tokens_map = {'5M': 5_000_000, '50M': 50_000_000, '500M': 500_000_000}
        tokens = tokens_map[config_name]
        
        std_time = tokens / best_std[1]['tokens_per_sec']
        trix_time = tokens / best_trix[1]['tokens_per_sec']
        
        print(f"  Training time (standard): {std_time/3600:.1f} hours")
        print(f"  Training time (TriX):     {trix_time/3600:.1f} hours")


if __name__ == "__main__":
    results = run_benchmarks()
    print_summary(results)
