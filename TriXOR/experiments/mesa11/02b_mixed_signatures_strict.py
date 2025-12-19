#!/usr/bin/env python3
"""
Mesa 11 Experiment 2b: Mixed Signatures (Strict Task)

LESSON FROM 2a: Position-only routing can compensate by learning
                position-specific transforms. We need a stricter test.

NEW TASK: Position-Content Joint Routing
    - 8 tiles, each specialized for ONE (position, content_class) pair
    - Position: {early, late} (2 classes)
    - Content: {A, B, C, D} (4 classes based on input quadrant)
    - Correct tile = position_class * 4 + content_class
    
    The ROUTING DECISION ITSELF must be correct - the tiles only output
    their ID. If routing is wrong, output is wrong.

This forces the network to use both position AND content for routing,
not just for downstream computation.

Author: Droid (Mesa 11 Exploration)
Date: 2024-12-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass 
class StrictExperimentResult:
    """Results from strict mixed signature experiment."""
    mixed_routing_accuracy: float
    position_only_routing_accuracy: float
    content_only_routing_accuracy: float
    learned_position_weight: float
    learned_content_weight: float
    theoretical_max: float


class StrictMixedRouter(nn.Module):
    """
    Router where correct routing requires BOTH position and content.
    
    8 tiles for 2 position classes × 4 content classes.
    Tile i is the ONLY correct tile for inputs with:
        position_class = i // 4
        content_class = i % 4
    """
    
    def __init__(self, d_content: int = 16, d_position: int = 16):
        super().__init__()
        self.d_content = d_content
        self.d_position = d_position
        self.num_tiles = 8  # 2 positions × 4 content classes
        
        # Learnable signatures: [position | content]
        self.position_sigs = nn.Parameter(torch.randn(self.num_tiles, d_position) * 0.1)
        self.content_sigs = nn.Parameter(torch.randn(self.num_tiles, d_content) * 0.1)
        
        # Learnable weights for blending
        self.position_logit = nn.Parameter(torch.tensor(0.0))
        self.content_logit = nn.Parameter(torch.tensor(0.0))
        
        # Position encoding (simple: early vs late)
        self.register_buffer('pos_early', torch.randn(d_position))
        self.register_buffer('pos_late', torch.randn(d_position))
        
        # Content class prototypes (4 quadrants)
        self.register_buffer('content_prototypes', torch.randn(4, d_content))
        
    def get_position_encoding(self, positions: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Map positions to early/late encoding."""
        # Early = positions < seq_len/2, Late = positions >= seq_len/2
        is_early = positions < (seq_len / 2)
        pos_enc = torch.where(
            is_early.unsqueeze(-1),
            self.pos_early.unsqueeze(0).unsqueeze(0).expand(positions.shape[0], positions.shape[1], -1),
            self.pos_late.unsqueeze(0).unsqueeze(0).expand(positions.shape[0], positions.shape[1], -1),
        )
        return pos_enc
    
    def get_content_class(self, x: torch.Tensor) -> torch.Tensor:
        """Classify content into 4 classes based on first 2 features."""
        # Quadrant based on sign of features 0 and 1
        f0_pos = x[..., 0] > 0  # (batch, seq)
        f1_pos = x[..., 1] > 0
        content_class = f0_pos.long() * 2 + f1_pos.long()  # 0,1,2,3
        return content_class
    
    def get_target_tile(self, positions: torch.Tensor, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute ground truth tile for each input."""
        is_early = positions < (seq_len / 2)
        position_class = (~is_early).long()  # 0 for early, 1 for late
        content_class = self.get_content_class(x)
        target_tile = position_class * 4 + content_class
        return target_tile
    
    @property
    def position_weight(self):
        w = torch.sigmoid(self.position_logit)
        return w / (w + torch.sigmoid(self.content_logit))
    
    @property
    def content_weight(self):
        w = torch.sigmoid(self.content_logit)
        return w / (torch.sigmoid(self.position_logit) + w)
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Route inputs to tiles.
        
        Returns:
            selected_tiles: (batch, seq) - which tile was selected
            target_tiles: (batch, seq) - which tile SHOULD be selected
            info: routing information
        """
        batch, seq, _ = x.shape
        
        # Get encodings
        pos_enc = self.get_position_encoding(positions, seq_len)  # (batch, seq, d_pos)
        
        # Compute routing scores
        # Position component
        pos_scores = torch.einsum('bsp,tp->bst', pos_enc, self.position_sigs.tanh())
        
        # Content component  
        content_scores = torch.einsum('bsc,tc->bst', x, self.content_sigs.tanh())
        
        # Blend based on learned weights
        pos_w = self.position_weight
        cont_w = self.content_weight
        
        combined_scores = pos_w * pos_scores + cont_w * content_scores
        
        # Select tiles
        selected_tiles = combined_scores.argmax(dim=-1)
        
        # Get targets
        target_tiles = self.get_target_tile(positions, x, seq_len)
        
        info = {
            'position_weight': pos_w.item(),
            'content_weight': cont_w.item(),
            'position_scores': pos_scores,
            'content_scores': content_scores,
            'combined_scores': combined_scores,
        }
        
        return selected_tiles, target_tiles, info


class PositionOnlyStrictRouter(nn.Module):
    """Baseline: route only on position."""
    
    def __init__(self, d_position: int = 16):
        super().__init__()
        self.num_tiles = 8
        self.d_position = d_position
        
        self.position_sigs = nn.Parameter(torch.randn(self.num_tiles, d_position) * 0.1)
        self.register_buffer('pos_early', torch.randn(d_position))
        self.register_buffer('pos_late', torch.randn(d_position))
        
    def get_position_encoding(self, positions: torch.Tensor, seq_len: int) -> torch.Tensor:
        is_early = positions < (seq_len / 2)
        pos_enc = torch.where(
            is_early.unsqueeze(-1),
            self.pos_early.unsqueeze(0).unsqueeze(0).expand(positions.shape[0], positions.shape[1], -1),
            self.pos_late.unsqueeze(0).unsqueeze(0).expand(positions.shape[0], positions.shape[1], -1),
        )
        return pos_enc
    
    def get_content_class(self, x: torch.Tensor) -> torch.Tensor:
        f0_pos = x[..., 0] > 0
        f1_pos = x[..., 1] > 0
        return f0_pos.long() * 2 + f1_pos.long()
    
    def get_target_tile(self, positions: torch.Tensor, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        is_early = positions < (seq_len / 2)
        position_class = (~is_early).long()
        content_class = self.get_content_class(x)
        return position_class * 4 + content_class
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        pos_enc = self.get_position_encoding(positions, seq_len)
        scores = torch.einsum('bsp,tp->bst', pos_enc, self.position_sigs.tanh())
        selected = scores.argmax(dim=-1)
        targets = self.get_target_tile(positions, x, seq_len)
        return selected, targets, {'position_weight': 1.0, 'content_weight': 0.0}


class ContentOnlyStrictRouter(nn.Module):
    """Baseline: route only on content."""
    
    def __init__(self, d_content: int = 16):
        super().__init__()
        self.num_tiles = 8
        self.d_content = d_content
        
        self.content_sigs = nn.Parameter(torch.randn(self.num_tiles, d_content) * 0.1)
        
    def get_content_class(self, x: torch.Tensor) -> torch.Tensor:
        f0_pos = x[..., 0] > 0
        f1_pos = x[..., 1] > 0
        return f0_pos.long() * 2 + f1_pos.long()
    
    def get_target_tile(self, positions: torch.Tensor, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        is_early = positions < (seq_len / 2)
        position_class = (~is_early).long()
        content_class = self.get_content_class(x)
        return position_class * 4 + content_class
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        scores = torch.einsum('bsc,tc->bst', x, self.content_sigs.tanh())
        selected = scores.argmax(dim=-1)
        targets = self.get_target_tile(positions, x, seq_len)
        return selected, targets, {'position_weight': 0.0, 'content_weight': 1.0}


def train_router(router: nn.Module, n_steps: int = 2000, batch_size: int = 64, seq_len: int = 8) -> List[float]:
    """Train router to maximize routing accuracy."""
    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)
    accuracies = []
    
    for step in range(n_steps):
        # Generate random inputs
        x = torch.randn(batch_size, seq_len, 16)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Forward
        selected, targets, info = router(x, positions, seq_len)
        
        # Loss: cross-entropy on tile selection
        # We need scores for CE loss
        if hasattr(router, 'position_sigs') and hasattr(router, 'content_sigs'):
            # Mixed router - recompute scores
            pos_enc = router.get_position_encoding(positions, seq_len)
            pos_scores = torch.einsum('bsp,tp->bst', pos_enc, router.position_sigs.tanh())
            content_scores = torch.einsum('bsc,tc->bst', x, router.content_sigs.tanh())
            scores = router.position_weight * pos_scores + router.content_weight * content_scores
        elif hasattr(router, 'position_sigs'):
            pos_enc = router.get_position_encoding(positions, seq_len)
            scores = torch.einsum('bsp,tp->bst', pos_enc, router.position_sigs.tanh())
        else:
            scores = torch.einsum('bsc,tc->bst', x, router.content_sigs.tanh())
        
        loss = F.cross_entropy(scores.view(-1, 8), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accuracy
        acc = (selected == targets).float().mean().item()
        accuracies.append(acc)
        
        if step % 400 == 0:
            pw = info.get('position_weight', 'N/A')
            cw = info.get('content_weight', 'N/A')
            if isinstance(pw, float):
                print(f"  Step {step}: loss={loss.item():.4f}, acc={acc:.3f}, pos_w={pw:.3f}, cont_w={cw:.3f}")
            else:
                print(f"  Step {step}: loss={loss.item():.4f}, acc={acc:.3f}")
    
    return accuracies


def compute_theoretical_max():
    """
    Compute theoretical maximum for each routing strategy.
    
    - Mixed: Can achieve 100% (can represent all 8 tile mappings)
    - Position-only: Can achieve 25% (can only distinguish 2 position classes, 
                     random among 4 content classes within each)
    - Content-only: Can achieve 50% (can distinguish 4 content classes,
                    random among 2 position classes for each)
    """
    return {
        'mixed': 1.0,
        'position_only': 0.25,  # 2 positions / 8 tiles
        'content_only': 0.50,   # 4 contents / 8 tiles
    }


def run_strict_experiment(n_steps: int = 2000) -> StrictExperimentResult:
    """Run the strict mixed signature experiment."""
    
    print("=" * 70)
    print("Mesa 11 Experiment 2b: Mixed Signatures (Strict Task)")
    print("=" * 70)
    print()
    print("TASK: Route to 1 of 8 tiles based on (position_class, content_class)")
    print("      position_class ∈ {early, late}")
    print("      content_class ∈ {0, 1, 2, 3} (based on feature signs)")
    print("      Correct tile = position_class * 4 + content_class")
    print()
    print("THEORETICAL LIMITS:")
    theoretical = compute_theoretical_max()
    print(f"  Mixed (pos+content): {theoretical['mixed']:.0%}")
    print(f"  Position-only:       {theoretical['position_only']:.0%}")
    print(f"  Content-only:        {theoretical['content_only']:.0%}")
    print()
    
    # Train mixed router
    print("Training Mixed Router...")
    torch.manual_seed(42)
    mixed = StrictMixedRouter()
    mixed_accs = train_router(mixed, n_steps=n_steps)
    mixed_final = sum(mixed_accs[-100:]) / 100
    
    # Get final weights
    _, _, info = mixed(torch.randn(1, 8, 16), torch.arange(8).unsqueeze(0), 8)
    pos_weight = info['position_weight']
    cont_weight = info['content_weight']
    
    # Train position-only
    print("\nTraining Position-Only Router...")
    torch.manual_seed(42)
    pos_only = PositionOnlyStrictRouter()
    pos_accs = train_router(pos_only, n_steps=n_steps)
    pos_final = sum(pos_accs[-100:]) / 100
    
    # Train content-only
    print("\nTraining Content-Only Router...")
    torch.manual_seed(42)
    cont_only = ContentOnlyStrictRouter()
    cont_accs = train_router(cont_only, n_steps=n_steps)
    cont_final = sum(cont_accs[-100:]) / 100
    
    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Routing Accuracy:")
    print(f"    Mixed:         {mixed_final:.3f}  (theoretical max: {theoretical['mixed']:.3f})")
    print(f"    Position-only: {pos_final:.3f}  (theoretical max: {theoretical['position_only']:.3f})")
    print(f"    Content-only:  {cont_final:.3f}  (theoretical max: {theoretical['content_only']:.3f})")
    print(f"\n  Learned Weights (Mixed Router):")
    print(f"    Position: {pos_weight:.3f}")
    print(f"    Content:  {cont_weight:.3f}")
    
    # Evaluate
    mixed_beats_both = mixed_final > pos_final + 0.05 and mixed_final > cont_final + 0.05
    mixed_near_optimal = mixed_final > 0.8
    both_used = 0.2 < pos_weight < 0.8
    
    print("\n" + "=" * 70)
    if mixed_beats_both and mixed_near_optimal:
        print("HYPOTHESIS CONFIRMED:")
        print("  - Mixed routing significantly outperforms single-mode baselines")
        print("  - Mixed routing approaches theoretical optimum")
        if both_used:
            print("  - Both position and content dimensions are used")
    elif mixed_beats_both:
        print("PARTIAL CONFIRMATION:")
        print("  - Mixed beats baselines but not near optimal")
    else:
        print("HYPOTHESIS NOT CONFIRMED")
    print("=" * 70)
    
    return StrictExperimentResult(
        mixed_routing_accuracy=mixed_final,
        position_only_routing_accuracy=pos_final,
        content_only_routing_accuracy=cont_final,
        learned_position_weight=pos_weight,
        learned_content_weight=cont_weight,
        theoretical_max=theoretical['mixed'],
    )


if __name__ == "__main__":
    result = run_strict_experiment(n_steps=2000)
    
    print("\n" + "=" * 70)
    print("MESA 11 EXPERIMENT 2b: COMPLETE")
    print("=" * 70)
    print(f"\nMixed routing accuracy: {result.mixed_routing_accuracy:.3f}")
    print(f"vs Position-only:       +{result.mixed_routing_accuracy - result.position_only_routing_accuracy:.3f}")
    print(f"vs Content-only:        +{result.mixed_routing_accuracy - result.content_only_routing_accuracy:.3f}")
    print(f"\nLearned weights: pos={result.learned_position_weight:.2f}, cont={result.learned_content_weight:.2f}")
