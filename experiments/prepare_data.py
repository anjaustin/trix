#!/usr/bin/env python3
"""
Data Preparation for TriX Training Experiments

Downloads and tokenizes subsets of:
- FineWeb-Edu (educational web text)
- The Stack v2 (permissively licensed code)

Outputs binary token files for efficient training.

Usage:
    python experiments/prepare_data.py --dataset fineweb --tokens 5M --output experiments/data/fineweb_5m
    python experiments/prepare_data.py --dataset stack --tokens 250M --output experiments/data/stack_250m
    python experiments/prepare_data.py --all  # Prepare all datasets
"""

import os
import sys
import json
import struct
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Iterator, Optional
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DatasetMetadata:
    """Metadata for a prepared dataset."""
    name: str
    source: str
    total_tokens: int
    vocab_size: int
    seq_length: int
    num_sequences: int
    created_at: str
    sha256: str
    tokenizer: str
    languages: Optional[list] = None
    
    def save(self, path: Path):
        with open(path / "metadata.json", "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "DatasetMetadata":
        with open(path / "metadata.json") as f:
            return cls(**json.load(f))


def parse_tokens(s: str) -> int:
    """Parse token count string (e.g., '5M', '250M', '1B')."""
    s = s.upper().strip()
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def get_tokenizer(vocab_size: int):
    """Get or create a tokenizer with specified vocab size."""
    try:
        from transformers import AutoTokenizer
        
        # Use GPT-2 tokenizer as base, it has ~50k vocab
        if vocab_size >= 32000:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif vocab_size >= 16000:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            # For smaller vocabs, use character-level or small BPE
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        return tokenizer
    except ImportError:
        print("transformers not installed. Using simple character tokenizer.")
        return None


class SimpleTokenizer:
    """Fallback character-level tokenizer."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = min(vocab_size, 65536)
    
    def encode(self, text: str) -> list:
        # Simple UTF-8 byte encoding
        return list(text.encode('utf-8', errors='replace'))[:self.vocab_size]
    
    def decode(self, tokens: list) -> str:
        return bytes(tokens).decode('utf-8', errors='replace')


def stream_fineweb_edu(num_tokens: int, vocab_size: int) -> Iterator[np.ndarray]:
    """
    Stream tokenized sequences from FineWeb-Edu.
    
    Yields numpy arrays of token IDs.
    """
    try:
        from datasets import load_dataset
        
        print(f"Loading FineWeb-Edu from HuggingFace...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",  # Use 10B sample for faster download
            split="train",
            streaming=True
        )
        
        tokenizer = get_tokenizer(vocab_size)
        if tokenizer is None:
            tokenizer = SimpleTokenizer(vocab_size)
        
        tokens_collected = 0
        buffer = []
        seq_length = 512  # Will be adjusted by caller
        
        for example in dataset:
            text = example.get('text', '')
            if not text:
                continue
            
            if hasattr(tokenizer, 'encode'):
                if hasattr(tokenizer, 'add_special_tokens'):
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                else:
                    tokens = tokenizer.encode(text)
            else:
                tokens = list(text.encode('utf-8'))
            
            # Truncate to vocab size
            tokens = [t % vocab_size for t in tokens]
            buffer.extend(tokens)
            
            while len(buffer) >= seq_length:
                seq = np.array(buffer[:seq_length], dtype=np.uint16)
                buffer = buffer[seq_length:]
                tokens_collected += seq_length
                yield seq
                
                if tokens_collected >= num_tokens:
                    return
                
                if tokens_collected % 1_000_000 == 0:
                    print(f"  Collected {tokens_collected / 1_000_000:.1f}M tokens...")
        
    except ImportError:
        print("datasets library not installed. Using synthetic data.")
        yield from generate_synthetic_data(num_tokens, vocab_size)


def stream_stack_v2(num_tokens: int, vocab_size: int, languages: list = None) -> Iterator[np.ndarray]:
    """
    Stream tokenized sequences from CodeParrot-clean (real GitHub code).
    
    Yields numpy arrays of token IDs.
    """
    try:
        from datasets import load_dataset
        
        print(f"Loading codeparrot/codeparrot-clean from HuggingFace...")
        print(f"  This contains real, permissively-licensed code from GitHub")
        
        tokenizer = get_tokenizer(vocab_size)
        if tokenizer is None:
            tokenizer = SimpleTokenizer(vocab_size)
        
        tokens_collected = 0
        buffer = []
        seq_length = 512
        
        dataset = load_dataset(
            "codeparrot/codeparrot-clean",
            streaming=True,
            split="train",
        )
        
        print(f"  Streaming real code...")
        
        for example in dataset:
            if tokens_collected >= num_tokens:
                break
                
            text = example.get('content', '')
            if not text or len(text) < 100:  # Skip tiny files
                continue
            
            if hasattr(tokenizer, 'encode'):
                if hasattr(tokenizer, 'add_special_tokens'):
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                else:
                    tokens = tokenizer.encode(text)
            else:
                tokens = list(text.encode('utf-8'))
            
            tokens = [t % vocab_size for t in tokens]
            buffer.extend(tokens)
            
            while len(buffer) >= seq_length:
                seq = np.array(buffer[:seq_length], dtype=np.uint16)
                buffer = buffer[seq_length:]
                tokens_collected += seq_length
                yield seq
                
                if tokens_collected >= num_tokens:
                    return
                
                if tokens_collected % 1_000_000 == 0:
                    print(f"  Collected {tokens_collected / 1_000_000:.1f}M tokens...")
        
    except ImportError:
        print("datasets library not installed. Using synthetic data.")
        yield from generate_synthetic_data(num_tokens, vocab_size)
    except Exception as e:
        print(f"Error loading CodeParrot-clean: {e}")
        print("Falling back to synthetic data...")
        yield from generate_synthetic_data(num_tokens, vocab_size)


def generate_synthetic_data(num_tokens: int, vocab_size: int, seq_length: int = 512) -> Iterator[np.ndarray]:
    """
    Generate synthetic data for testing when real data unavailable.
    
    Creates structured patterns that are learnable but not trivial.
    """
    print("Generating synthetic training data...")
    
    np.random.seed(42)
    tokens_generated = 0
    
    while tokens_generated < num_tokens:
        # Mix of patterns:
        # 1. Random tokens (30%)
        # 2. Repeated patterns (30%)
        # 3. Sequential patterns (40%)
        
        pattern_type = np.random.choice([0, 1, 2], p=[0.3, 0.3, 0.4])
        
        if pattern_type == 0:
            # Random
            seq = np.random.randint(0, vocab_size, seq_length, dtype=np.uint16)
        elif pattern_type == 1:
            # Repeated pattern
            pattern_len = np.random.randint(4, 32)
            pattern = np.random.randint(0, vocab_size, pattern_len, dtype=np.uint16)
            seq = np.tile(pattern, seq_length // pattern_len + 1)[:seq_length]
        else:
            # Sequential with noise
            start = np.random.randint(0, vocab_size)
            seq = (np.arange(seq_length) + start) % vocab_size
            noise_mask = np.random.random(seq_length) < 0.1
            seq[noise_mask] = np.random.randint(0, vocab_size, noise_mask.sum())
            seq = seq.astype(np.uint16)
        
        tokens_generated += seq_length
        yield seq
        
        if tokens_generated % 1_000_000 == 0:
            print(f"  Generated {tokens_generated / 1_000_000:.1f}M tokens...")


def write_binary_dataset(
    sequences: Iterator[np.ndarray],
    output_dir: Path,
    name: str,
    source: str,
    total_tokens: int,
    vocab_size: int,
    seq_length: int,
    val_ratio: float = 0.05,
    languages: list = None
):
    """
    Write tokenized sequences to binary files.
    
    Format: Simple concatenated uint16 tokens.
    Creates train.bin and val.bin.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    
    print(f"Writing dataset to {output_dir}/")
    
    train_file = open(train_path, 'wb')
    val_file = open(val_path, 'wb')
    
    num_sequences = 0
    tokens_written = 0
    val_tokens = int(total_tokens * val_ratio)
    train_tokens = total_tokens - val_tokens
    
    hasher = hashlib.sha256()
    
    for seq in sequences:
        if tokens_written < train_tokens:
            train_file.write(seq.tobytes())
        else:
            val_file.write(seq.tobytes())
        
        hasher.update(seq.tobytes())
        tokens_written += len(seq)
        num_sequences += 1
        
        if tokens_written >= total_tokens:
            break
    
    train_file.close()
    val_file.close()
    
    # Create metadata
    metadata = DatasetMetadata(
        name=name,
        source=source,
        total_tokens=tokens_written,
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_sequences=num_sequences,
        created_at=datetime.now().isoformat(),
        sha256=hasher.hexdigest()[:16],
        tokenizer="gpt2" if vocab_size >= 32000 else "byte",
        languages=languages
    )
    metadata.save(output_dir)
    
    print(f"Created dataset:")
    print(f"  Total tokens: {tokens_written:,}")
    print(f"  Train tokens: {train_tokens:,}")
    print(f"  Val tokens: {tokens_written - train_tokens:,}")
    print(f"  Sequences: {num_sequences:,}")
    print(f"  SHA256: {metadata.sha256}")
    
    return metadata


def prepare_fineweb(tokens: int, output_dir: Path, vocab_size: int = 32000):
    """Prepare FineWeb-Edu dataset."""
    print(f"\n{'='*60}")
    print(f"Preparing FineWeb-Edu ({tokens/1e6:.0f}M tokens)")
    print(f"{'='*60}\n")
    
    sequences = stream_fineweb_edu(tokens, vocab_size)
    return write_binary_dataset(
        sequences=sequences,
        output_dir=output_dir,
        name=f"fineweb-edu-{tokens//1_000_000}m",
        source="HuggingFaceFW/fineweb-edu",
        total_tokens=tokens,
        vocab_size=vocab_size,
        seq_length=512
    )


def prepare_stack(tokens: int, output_dir: Path, vocab_size: int = 32000, languages: list = None):
    """Prepare The Stack v2 dataset."""
    if languages is None:
        # Systems/Embedded focus for TriX code model
        languages = [
            'python',       # ML/AI, PyTorch
            'c',            # Embedded, systems
            'cpp',          # Systems, performance
            'rust',         # Modern systems
            'swift',        # iOS mobile
            'kotlin',       # Android mobile
            'java',         # Android, embedded
            'objective-c',  # iOS legacy
            'cuda',         # NVIDIA GPU
            'assembly',     # Embedded, optimization
        ]
    
    print(f"\n{'='*60}")
    print(f"Preparing The Stack v2 ({tokens/1e6:.0f}M tokens)")
    print(f"Languages: {languages}")
    print(f"{'='*60}\n")
    
    sequences = stream_stack_v2(tokens, vocab_size, languages)
    return write_binary_dataset(
        sequences=sequences,
        output_dir=output_dir,
        name=f"stack-v2-{tokens//1_000_000}m",
        source="bigcode/the-stack-v2",
        total_tokens=tokens,
        vocab_size=vocab_size,
        seq_length=512,
        languages=languages
    )


def prepare_all():
    """Prepare all datasets for the experiment."""
    base_dir = Path(__file__).parent / "data"
    
    datasets = [
        ("fineweb", 5_000_000, 8192, base_dir / "fineweb_5m"),
        ("fineweb", 50_000_000, 16384, base_dir / "fineweb_50m"),
        ("fineweb", 500_000_000, 32000, base_dir / "fineweb_500m"),
        ("stack", 250_000_000, 32000, base_dir / "stack_250m"),
    ]
    
    results = []
    for dataset_type, tokens, vocab_size, output_dir in datasets:
        if dataset_type == "fineweb":
            meta = prepare_fineweb(tokens, output_dir, vocab_size)
        else:
            meta = prepare_stack(tokens, output_dir, vocab_size)
        results.append(meta)
    
    # Summary
    print(f"\n{'='*60}")
    print("All datasets prepared:")
    print(f"{'='*60}")
    for meta in results:
        print(f"  {meta.name}: {meta.total_tokens:,} tokens")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for TriX training")
    parser.add_argument("--dataset", choices=["fineweb", "stack", "synthetic"], 
                       help="Dataset to prepare")
    parser.add_argument("--tokens", type=str, default="5M",
                       help="Number of tokens (e.g., 5M, 250M, 1B)")
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Vocabulary size")
    parser.add_argument("--output", type=Path, required=False,
                       help="Output directory")
    parser.add_argument("--all", action="store_true",
                       help="Prepare all datasets for the experiment")
    parser.add_argument("--languages", nargs="+", default=None,
                       help="Languages for Stack dataset")
    
    args = parser.parse_args()
    
    if args.all:
        prepare_all()
        return
    
    if not args.dataset:
        parser.error("--dataset or --all required")
    
    tokens = parse_tokens(args.tokens)
    output_dir = args.output or Path(f"experiments/data/{args.dataset}_{args.tokens.lower()}")
    
    if args.dataset == "fineweb":
        prepare_fineweb(tokens, output_dir, args.vocab_size)
    elif args.dataset == "stack":
        prepare_stack(tokens, output_dir, args.vocab_size, args.languages)
    else:
        # Synthetic
        print(f"Preparing synthetic dataset ({tokens/1e6:.0f}M tokens)")
        sequences = generate_synthetic_data(tokens, args.vocab_size)
        write_binary_dataset(
            sequences=sequences,
            output_dir=output_dir,
            name=f"synthetic-{tokens//1_000_000}m",
            source="synthetic",
            total_tokens=tokens,
            vocab_size=args.vocab_size,
            seq_length=512
        )


if __name__ == "__main__":
    main()
