#!/usr/bin/env python3
"""
TriX Training Script

Comprehensive training with detailed logging for scientific reproducibility.
Designed for parallel execution on NVIDIA Jetson AGX Thor.

Usage:
    python experiments/train.py --config experiments/configs/tiny.yaml
    python experiments/train.py --config experiments/configs/small.yaml --gpu 0
    
Parallel:
    python experiments/run_all.py  # Launches all 4 models in parallel

Features:
- Per-step metrics logging
- Routing statistics tracking
- Checkpoint saving with full state
- Automatic mixed precision (BF16)
- Gradient accumulation
- Learning rate scheduling with warmup
- Early stopping on NaN
"""

import os
import sys
import json
import time
import math
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix import SparseLookupFFN, SparseLookupBlock


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Experiment
    name: str = "trix-experiment"
    description: str = ""
    seed: int = 42
    
    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    vocab_size: int = 8192
    num_tiles: int = 16
    tiles_per_cluster: int = 4
    grid_size: int = 16
    dropout: float = 0.1
    
    # Data
    data_dir: str = ""
    total_tokens: int = 5_000_000
    seq_length: int = 256
    
    # Training
    batch_size: int = 256
    gradient_accumulation: int = 1
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_tokens: int = 500_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bf16"
    aux_balance_weight: float = 0.01
    aux_diversity_weight: float = 0.001
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    log_dir: str = ""
    checkpoint_dir: str = ""
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Flatten nested structure
        for section in ['experiment', 'model', 'data', 'training', 'logging']:
            if section in data:
                for key, value in data[section].items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        return config


# =============================================================================
# Dataset
# =============================================================================

class BinaryTokenDataset(Dataset):
    """
    Memory-mapped binary token dataset.
    
    Efficient loading of pre-tokenized data.
    """
    
    def __init__(self, path: Path, seq_length: int, vocab_size: int):
        self.path = Path(path)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Memory-map the file
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)
        self.num_sequences = self.num_tokens // seq_length
        
        logging.info(f"Loaded {self.num_tokens:,} tokens from {path}")
        logging.info(f"  Sequences: {self.num_sequences:,}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        tokens = self.data[start:end].astype(np.int64)
        
        # Clamp to vocab size
        tokens = np.clip(tokens, 0, self.vocab_size - 1)
        
        return torch.from_numpy(tokens)


# =============================================================================
# Model
# =============================================================================

class TriXLanguageModel(nn.Module):
    """
    TriX Language Model using SparseLookupFFN.
    
    Architecture:
        Embedding -> [SparseLookupBlock] x N -> LayerNorm -> LM Head
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        
        self.config = config
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks with SparseLookupFFN
        self.blocks = nn.ModuleList([
            SparseLookupBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                num_tiles=config.num_tiles,
                tiles_per_cluster=config.tiles_per_cluster,
                grid_size=config.grid_size,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Model parameters: {self.num_params:,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: Token indices [batch, seq_length]
            
        Returns:
            logits: Output logits [batch, seq_length, vocab_size]
            aux_losses: Dictionary of auxiliary losses
        """
        B, T = idx.shape
        device = idx.device
        
        # Embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok_emb + pos_emb)
        
        # Collect routing info and aux losses
        all_routing_info = []
        total_aux = torch.tensor(0.0, device=device)
        total_balance = torch.tensor(0.0, device=device)
        total_diversity = torch.tensor(0.0, device=device)
        
        # Transformer blocks
        for block in self.blocks:
            x, routing_info, aux_losses = block(x, is_causal=True)
            all_routing_info.append(routing_info)
            total_aux = total_aux + aux_losses.get('total_aux', 0.0)
            total_balance = total_balance + aux_losses.get('balance_loss', 0.0)
            total_diversity = total_diversity + aux_losses.get('diversity_loss', 0.0)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        aux = {
            'total_aux': total_aux,
            'balance_loss': total_balance,
            'diversity_loss': total_diversity,
            'routing_info': all_routing_info,
        }
        
        return logits, aux
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics from all blocks."""
        stats = {}
        for i, block in enumerate(self.blocks):
            block_stats = block.ffn.get_routing_stats()
            for key, value in block_stats.items():
                stats[f"layer_{i}/{key}"] = value
        return stats


# =============================================================================
# Training Utilities
# =============================================================================

class TrainingLogger:
    """
    Comprehensive training logger.
    
    Logs to both file and console with structured JSON metrics.
    """
    
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Setup file logging
        self.log_file = self.log_dir / "train.log"
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
        # Setup Python logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.step_metrics = []
        self.eval_metrics = []
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log per-step metrics."""
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.step_metrics.append(metrics)
        
        # Write to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Console output
        loss = metrics.get('loss', 0)
        lr = metrics.get('lr', 0)
        throughput = metrics.get('throughput', 0)
        logging.info(
            f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e} | "
            f"Throughput: {throughput:,.0f} tok/s"
        )
    
    def log_eval(self, step: int, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        metrics['step'] = step
        metrics['type'] = 'eval'
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.eval_metrics.append(metrics)
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        val_loss = metrics.get('val_loss', 0)
        val_ppl = metrics.get('val_ppl', 0)
        logging.info(f"Eval Step {step:6d} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
    
    def log_checkpoint(self, step: int, path: str):
        """Log checkpoint save."""
        logging.info(f"Checkpoint saved: {path}")
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps({
                'step': step,
                'type': 'checkpoint',
                'path': path,
                'timestamp': datetime.now().isoformat()
            }) + '\n')
    
    def finalize(self, final_metrics: Dict[str, Any]):
        """Save final summary."""
        summary = {
            'experiment': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'final_metrics': final_metrics,
            'total_steps': len(self.step_metrics),
            'total_evals': len(self.eval_metrics),
        }
        
        with open(self.log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Training complete. Duration: {summary['duration_seconds']/3600:.2f} hours")


def get_lr(step: int, warmup_steps: int, max_lr: float, min_lr: float, total_steps: int) -> float:
    """
    Learning rate schedule with warmup and cosine decay.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    decay_steps = total_steps - warmup_steps
    decay_progress = (step - warmup_steps) / decay_steps
    
    # Cosine decay
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_progress))


def compute_routing_entropy(routing_info: List[Dict]) -> float:
    """Compute average routing entropy across layers."""
    entropies = []
    
    for info in routing_info:
        if 'tile_idx' in info:
            tile_idx = info['tile_idx'].flatten()
            counts = torch.bincount(tile_idx, minlength=64).float()
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -(probs * probs.log()).sum().item()
            entropies.append(entropy)
    
    return np.mean(entropies) if entropies else 0.0


# =============================================================================
# Training Loop
# =============================================================================

def train(config: ExperimentConfig):
    """Main training function."""
    
    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logging
    logger = TrainingLogger(config.log_dir, config.name)
    logging.info(f"Starting experiment: {config.name}")
    logging.info(f"Device: {device}")
    logging.info(f"Config: {asdict(config)}")
    
    # Data
    train_dataset = BinaryTokenDataset(
        Path(config.data_dir) / "train.bin",
        config.seq_length,
        config.vocab_size
    )
    val_dataset = BinaryTokenDataset(
        Path(config.data_dir) / "val.bin",
        config.seq_length,
        config.vocab_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Model
    model = TriXLanguageModel(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Mixed precision
    use_amp = config.precision == "bf16" and device.type == 'cuda'
    dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # BF16 doesn't need scaling
    
    # Training state
    tokens_per_step = config.batch_size * config.seq_length * config.gradient_accumulation
    total_steps = config.total_tokens // tokens_per_step
    warmup_steps = config.warmup_tokens // tokens_per_step
    
    logging.info(f"Total steps: {total_steps:,}")
    logging.info(f"Warmup steps: {warmup_steps:,}")
    logging.info(f"Tokens per step: {tokens_per_step:,}")
    
    # Training loop
    model.train()
    step = 0
    tokens_processed = 0
    best_val_loss = float('inf')
    grad_norm = 0.0  # Initialize for logging before first gradient step
    
    train_iter = iter(train_loader)
    step_start_time = time.perf_counter()
    
    while step < total_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        batch = batch.to(device)
        
        # Learning rate schedule
        lr = get_lr(step, warmup_steps, config.learning_rate, config.min_lr, total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            logits, aux = model(batch)
            
            # Language modeling loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Auxiliary losses
            aux_loss = aux['total_aux'] * config.aux_balance_weight
            
            # Total loss
            loss = lm_loss + aux_loss
            loss = loss / config.gradient_accumulation
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config.gradient_accumulation == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            
            optimizer.step()
            optimizer.zero_grad()
        
        tokens_processed += config.batch_size * config.seq_length
        
        # Logging
        if step % config.log_interval == 0:
            step_time = time.perf_counter() - step_start_time
            throughput = config.log_interval * config.batch_size * config.seq_length / step_time
            
            memory_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0
            routing_entropy = compute_routing_entropy(aux.get('routing_info', []))
            
            metrics = {
                'loss': lm_loss.item(),
                'aux_loss': aux['total_aux'].item() if isinstance(aux['total_aux'], torch.Tensor) else aux['total_aux'],
                'lr': lr,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'throughput': throughput,
                'memory_gb': memory_gb,
                'tokens': tokens_processed,
                'routing_entropy': routing_entropy,
            }
            
            logger.log_step(step, metrics)
            step_start_time = time.perf_counter()
            
            # Check for NaN
            if math.isnan(lm_loss.item()):
                logging.error("NaN loss detected! Stopping training.")
                break
        
        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    
                    with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
                        logits, _ = model(val_batch)
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = val_batch[:, 1:].contiguous()
                        val_loss = F.cross_entropy(
                            shift_logits.view(-1, config.vocab_size),
                            shift_labels.view(-1)
                        )
                    
                    val_losses.append(val_loss.item())
                    
                    # Limit eval batches for speed
                    if len(val_losses) >= 50:
                        break
            
            avg_val_loss = np.mean(val_losses)
            val_ppl = math.exp(avg_val_loss)
            
            routing_stats = model.get_routing_stats()
            
            eval_metrics = {
                'val_loss': avg_val_loss,
                'val_ppl': val_ppl,
                **{k: v for k, v in routing_stats.items() if isinstance(v, (int, float))}
            }
            
            logger.log_eval(step, eval_metrics)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(
                    model, optimizer, step, config,
                    Path(config.checkpoint_dir) / "best.pt"
                )
            
            model.train()
        
        # Checkpoint
        if step > 0 and step % config.save_interval == 0:
            ckpt_path = Path(config.checkpoint_dir) / f"step_{step}.pt"
            save_checkpoint(model, optimizer, step, config, ckpt_path)
            logger.log_checkpoint(step, str(ckpt_path))
        
        step += 1
    
    # Final evaluation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.to(device)
            with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
                logits, _ = model(val_batch)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = val_batch[:, 1:].contiguous()
                val_loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1)
                )
            val_losses.append(val_loss.item())
    
    final_val_loss = np.mean(val_losses)
    final_val_ppl = math.exp(final_val_loss)
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, step, config,
        Path(config.checkpoint_dir) / "final.pt"
    )
    
    # Finalize logging
    final_metrics = {
        'final_val_loss': final_val_loss,
        'final_val_ppl': final_val_ppl,
        'best_val_loss': best_val_loss,
        'best_val_ppl': math.exp(best_val_loss),
        'total_tokens': tokens_processed,
        'total_steps': step,
        'model_params': model.num_params,
    }
    
    logger.finalize(final_metrics)
    
    return final_metrics


def save_checkpoint(model, optimizer, step, config, path):
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': asdict(config),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train TriX model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, help="Override data directory")
    parser.add_argument("--log-dir", type=str, help="Override log directory")
    parser.add_argument("--checkpoint-dir", type=str, help="Override checkpoint directory")
    
    args = parser.parse_args()
    
    # Load config
    config = ExperimentConfig.from_yaml(args.config)
    
    # Override paths if specified
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    # Set default paths based on experiment name
    if not config.data_dir:
        config.data_dir = f"experiments/data/fineweb_{config.total_tokens // 1_000_000}m"
    if not config.log_dir:
        config.log_dir = f"experiments/logs/{config.name}"
    if not config.checkpoint_dir:
        config.checkpoint_dir = f"experiments/checkpoints/{config.name}"
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
