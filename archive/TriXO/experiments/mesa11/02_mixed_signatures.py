#!/usr/bin/env python3
"""
Mesa 11 Validation Experiment 2: Mixed Signatures

HYPOTHESIS: Signatures can blend positional and feature dimensions,
            enabling architectures that route on BOTH sequence position
            AND content simultaneously.

PROOF STRUCTURE:
    1. Extend TriX signatures to [position_dims | feature_dims]
    2. Train on a task requiring BOTH positional and content awareness
    3. Analyze learned signature weights - do both dimensions get used?
    4. Compare against position-only and content-only baselines

TASK: Position-Dependent Classification
    - Input: sequence of vectors
    - Rule: classify based on BOTH position AND content
    - Example: "positive sentiment at position 0-2, negative at 3+"
    - Requires blended addressing to solve optimally

Author: Droid (Mesa 11 Exploration)
Date: 2024-12-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import math


@dataclass
class ExperimentResult:
    """Results from mixed signature experiment."""
    task_accuracy: float
    position_weight: float  # Learned weight on position dims
    content_weight: float   # Learned weight on content dims
    baseline_position_only: float
    baseline_content_only: float
    signature_analysis: Dict


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]


class MixedSignatureTile(nn.Module):
    """
    A tile with mixed positional + content signature.
    
    Signature = [position_signature | content_signature]
    
    The tile learns to weight these dimensions for routing.
    """
    
    def __init__(self, d_position: int, d_content: int, d_hidden: int):
        super().__init__()
        self.d_position = d_position
        self.d_content = d_content
        self.d_total = d_position + d_content
        
        # Learnable ternary signatures for each dimension
        self.position_sig = nn.Parameter(torch.randn(d_position))
        self.content_sig = nn.Parameter(torch.randn(d_content))
        
        # Tile computation (simple MLP)
        self.fc1 = nn.Linear(d_content, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_content)
        
    @property
    def signature(self) -> torch.Tensor:
        """Combined signature [position | content], ternarized."""
        pos_ternary = self.position_sig.sign()
        content_ternary = self.content_sig.sign()
        return torch.cat([pos_ternary, content_ternary])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input (content only, position handled by routing)."""
        h = F.gelu(self.fc1(x))
        return self.fc2(h)


class MixedAddressRouter(nn.Module):
    """
    Router that matches inputs against mixed signatures.
    
    Input representation: [positional_encoding | content_features]
    Signature: [position_signature | content_signature]
    
    Learns to weight position vs content for optimal routing.
    """
    
    def __init__(self, d_position: int, d_content: int, num_tiles: int):
        super().__init__()
        self.d_position = d_position
        self.d_content = d_content
        self.d_total = d_position + d_content
        self.num_tiles = num_tiles
        
        # Learnable weighting between position and content
        # Initialized to equal weight
        self.position_weight = nn.Parameter(torch.tensor(0.5))
        self.content_weight = nn.Parameter(torch.tensor(0.5))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_position)
        
    def compute_address(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute mixed address for input.
        
        Args:
            x: content features (batch, seq, d_content)
            positions: sequence positions (batch, seq)
            
        Returns:
            address: (batch, seq, d_position + d_content)
        """
        batch, seq, _ = x.shape
        
        # Get positional encoding
        pos_enc = self.pos_encoding(seq).unsqueeze(0).expand(batch, -1, -1)
        
        # Weight and combine
        pos_weight = torch.sigmoid(self.position_weight)
        content_weight = torch.sigmoid(self.content_weight)
        
        # Normalize to sum to 1
        total = pos_weight + content_weight
        pos_weight = pos_weight / total
        content_weight = content_weight / total
        
        # Weighted address
        address = torch.cat([
            pos_weight * pos_enc,
            content_weight * x
        ], dim=-1)
        
        return address
    
    def route(self, address: torch.Tensor, signatures: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route based on address-signature matching.
        
        Args:
            address: (batch, seq, d_total)
            signatures: (num_tiles, d_total)
            
        Returns:
            indices: winning tile indices (batch, seq)
            scores: routing scores (batch, seq, num_tiles)
        """
        # Dot product similarity
        scores = torch.einsum('bsd,td->bst', address, signatures)
        
        # Winner-take-all
        indices = scores.argmax(dim=-1)
        
        return indices, scores


class MixedSignatureFFN(nn.Module):
    """
    FFN with mixed positional + content routing.
    
    This is the core test: can a network learn to use BOTH
    position and content for routing decisions?
    """
    
    def __init__(
        self,
        d_content: int,
        d_position: int = 32,
        num_tiles: int = 8,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.d_content = d_content
        self.d_position = d_position
        self.num_tiles = num_tiles
        
        # Router
        self.router = MixedAddressRouter(d_position, d_content, num_tiles)
        
        # Tiles
        self.tiles = nn.ModuleList([
            MixedSignatureTile(d_position, d_content, d_hidden)
            for _ in range(num_tiles)
        ])
        
    def get_signatures(self) -> torch.Tensor:
        """Stack all tile signatures."""
        return torch.stack([t.signature for t in self.tiles])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with mixed routing.
        
        Args:
            x: input (batch, seq, d_content)
            
        Returns:
            output: transformed (batch, seq, d_content)
            info: routing information
        """
        batch, seq, _ = x.shape
        positions = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, -1)
        
        # Compute mixed address
        address = self.router.compute_address(x, positions)
        
        # Get signatures and route
        signatures = self.get_signatures()
        indices, scores = self.router.route(address, signatures)
        
        # Execute winning tiles
        output = torch.zeros_like(x)
        for b in range(batch):
            for s in range(seq):
                tile_idx = indices[b, s].item()
                output[b, s] = self.tiles[tile_idx](x[b, s])
        
        # Compute position/content weights for analysis
        pos_w = torch.sigmoid(self.router.position_weight).item()
        cont_w = torch.sigmoid(self.router.content_weight).item()
        total = pos_w + cont_w
        
        info = {
            'indices': indices,
            'scores': scores,
            'position_weight': pos_w / total,
            'content_weight': cont_w / total,
        }
        
        return output, info


class PositionDependentTask:
    """
    Task that requires both positional and content awareness.
    
    Rule: 
    - Positions 0-2: classify by content feature 0 (threshold 0)
    - Positions 3+: classify by content feature 1 (threshold 0)
    
    This cannot be solved optimally with position-only or content-only routing.
    """
    
    def __init__(self, d_content: int = 16, seq_len: int = 6):
        self.d_content = d_content
        self.seq_len = seq_len
        
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of (input, label) pairs."""
        # Random content features
        x = torch.randn(batch_size, self.seq_len, self.d_content)
        
        # Labels based on position-dependent rule
        labels = torch.zeros(batch_size, self.seq_len, dtype=torch.long)
        
        for pos in range(self.seq_len):
            if pos < 3:
                # Use feature 0
                labels[:, pos] = (x[:, pos, 0] > 0).long()
            else:
                # Use feature 1
                labels[:, pos] = (x[:, pos, 1] > 0).long()
        
        return x, labels
    
    def theoretical_accuracy(self) -> float:
        """Theoretical max accuracy for this task."""
        return 1.0  # Deterministic rule, should be 100%


def train_model(
    model: nn.Module,
    task: PositionDependentTask,
    n_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> List[float]:
    """Train model on task, return accuracy history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    classifier = nn.Linear(task.d_content, 2)
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    accuracies = []
    
    for step in range(n_steps):
        x, labels = task.generate_batch(batch_size)
        
        # Forward
        if hasattr(model, 'forward'):
            if isinstance(model, MixedSignatureFFN):
                output, info = model(x)
            else:
                output = model(x)
        else:
            output = x  # Baseline: no transformation
        
        # Classify
        logits = classifier(output)
        loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
        
        # Backward
        optimizer.zero_grad()
        optimizer_cls.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_cls.step()
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        accuracies.append(acc)
        
        if step % 200 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc:.3f}")
    
    return accuracies


class PositionOnlyRouter(nn.Module):
    """Baseline: route only on position."""
    
    def __init__(self, d_content: int, d_position: int = 32, num_tiles: int = 8):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_position)
        self.tile_sigs = nn.Parameter(torch.randn(num_tiles, d_position))
        self.transforms = nn.ModuleList([
            nn.Linear(d_content, d_content) for _ in range(num_tiles)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d = x.shape
        pos_enc = self.pos_encoding(seq).unsqueeze(0).expand(batch, -1, -1)
        
        # Route on position only
        scores = torch.einsum('bsp,tp->bst', pos_enc, self.tile_sigs.sign())
        indices = scores.argmax(dim=-1)
        
        output = torch.zeros_like(x)
        for b in range(batch):
            for s in range(seq):
                tile_idx = indices[b, s].item()
                output[b, s] = self.transforms[tile_idx](x[b, s])
        
        return output


class ContentOnlyRouter(nn.Module):
    """Baseline: route only on content."""
    
    def __init__(self, d_content: int, num_tiles: int = 8):
        super().__init__()
        self.tile_sigs = nn.Parameter(torch.randn(num_tiles, d_content))
        self.transforms = nn.ModuleList([
            nn.Linear(d_content, d_content) for _ in range(num_tiles)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d = x.shape
        
        # Route on content only
        scores = torch.einsum('bsd,td->bst', x, self.tile_sigs.sign())
        indices = scores.argmax(dim=-1)
        
        output = torch.zeros_like(x)
        for b in range(batch):
            for s in range(seq):
                tile_idx = indices[b, s].item()
                output[b, s] = self.transforms[tile_idx](x[b, s])
        
        return output


def analyze_signatures(model: MixedSignatureFFN) -> Dict:
    """Analyze learned signature structure."""
    signatures = model.get_signatures().detach()
    
    # Split into position and content parts
    pos_sigs = signatures[:, :model.d_position]
    content_sigs = signatures[:, model.d_position:]
    
    # Measure diversity in each part
    def signature_diversity(sigs):
        n = sigs.shape[0]
        total_dist = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = (sigs[i] != sigs[j]).float().mean().item()
                total_dist += dist
                count += 1
        return total_dist / count if count > 0 else 0
    
    return {
        'position_diversity': signature_diversity(pos_sigs),
        'content_diversity': signature_diversity(content_sigs),
        'position_sparsity': (pos_sigs == 0).float().mean().item(),
        'content_sparsity': (content_sigs == 0).float().mean().item(),
    }


def run_experiment(
    d_content: int = 16,
    d_position: int = 32,
    num_tiles: int = 8,
    n_steps: int = 1000,
) -> ExperimentResult:
    """Run the mixed signature experiment."""
    
    print("=" * 70)
    print("Mesa 11 Experiment 2: Mixed Signatures")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Networks can learn to blend positional and content")
    print("            addressing when the task requires both.")
    print()
    
    task = PositionDependentTask(d_content=d_content)
    
    # Train mixed model
    print("Training Mixed Signature Model...")
    torch.manual_seed(42)
    mixed_model = MixedSignatureFFN(
        d_content=d_content,
        d_position=d_position,
        num_tiles=num_tiles,
    )
    mixed_accs = train_model(mixed_model, task, n_steps=n_steps)
    mixed_final = sum(mixed_accs[-100:]) / 100
    
    # Get learned weights
    _, info = mixed_model(torch.randn(1, task.seq_len, d_content))
    pos_weight = info['position_weight']
    cont_weight = info['content_weight']
    
    print(f"\n  Final accuracy: {mixed_final:.3f}")
    print(f"  Learned position weight: {pos_weight:.3f}")
    print(f"  Learned content weight: {cont_weight:.3f}")
    
    # Train position-only baseline
    print("\nTraining Position-Only Baseline...")
    torch.manual_seed(42)
    pos_model = PositionOnlyRouter(d_content, d_position, num_tiles)
    pos_accs = train_model(pos_model, task, n_steps=n_steps)
    pos_final = sum(pos_accs[-100:]) / 100
    print(f"  Final accuracy: {pos_final:.3f}")
    
    # Train content-only baseline
    print("\nTraining Content-Only Baseline...")
    torch.manual_seed(42)
    cont_model = ContentOnlyRouter(d_content, num_tiles)
    cont_accs = train_model(cont_model, task, n_steps=n_steps)
    cont_final = sum(cont_accs[-100:]) / 100
    print(f"  Final accuracy: {cont_final:.3f}")
    
    # Analyze signatures
    sig_analysis = analyze_signatures(mixed_model)
    
    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Mixed Model:        {mixed_final:.3f} accuracy")
    print(f"  Position-Only:      {pos_final:.3f} accuracy")
    print(f"  Content-Only:       {cont_final:.3f} accuracy")
    print(f"\n  Learned Weights:")
    print(f"    Position: {pos_weight:.3f}")
    print(f"    Content:  {cont_weight:.3f}")
    print(f"\n  Signature Analysis:")
    print(f"    Position diversity: {sig_analysis['position_diversity']:.3f}")
    print(f"    Content diversity:  {sig_analysis['content_diversity']:.3f}")
    
    # Evaluate hypothesis
    mixed_wins = mixed_final > max(pos_final, cont_final) + 0.02
    both_used = 0.2 < pos_weight < 0.8 and 0.2 < cont_weight < 0.8
    
    print("\n" + "=" * 70)
    if mixed_wins and both_used:
        print("HYPOTHESIS CONFIRMED: Mixed addressing outperforms single-mode")
        print("                      Both position and content dimensions used")
    elif mixed_wins:
        print("PARTIAL CONFIRMATION: Mixed wins but weighting is extreme")
    else:
        print("HYPOTHESIS NOT CONFIRMED: See results above")
    print("=" * 70)
    
    return ExperimentResult(
        task_accuracy=mixed_final,
        position_weight=pos_weight,
        content_weight=cont_weight,
        baseline_position_only=pos_final,
        baseline_content_only=cont_final,
        signature_analysis=sig_analysis,
    )


if __name__ == "__main__":
    result = run_experiment(
        d_content=16,
        d_position=32,
        num_tiles=8,
        n_steps=1500,
    )
    
    print("\n" + "=" * 70)
    print("MESA 11 EXPERIMENT 2: COMPLETE")
    print("=" * 70)
    print(f"\nMixed model accuracy: {result.task_accuracy:.3f}")
    print(f"Improvement over best baseline: {result.task_accuracy - max(result.baseline_position_only, result.baseline_content_only):.3f}")
    print(f"\nLearned address weighting:")
    print(f"  Position: {result.position_weight:.1%}")
    print(f"  Content:  {result.content_weight:.1%}")
