"""
Adaptive Training Pipeline - Multi-Phase Training with Observation

A structured training approach with four phases:

Phase 1: EXPLORATION   - Learn routing patterns without accuracy pressure
Phase 2: EXPEDITION    - Identify stable routing destinations (nodes)
Phase 3: CONVERGENCE   - Train for task accuracy
Phase 4: MASTERY       - Observer provides targeted interventions

The observer watches phases 1-3 to understand model behavior, then
applies that understanding during phase 4 to guide training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .guardian import TrainingObserver, GuardianAngel
from .observer import ObservationFrame, ObservationBuffer
from .programmable_tile import ProgrammableTileBank


class Phase(Enum):
    """The four phases of adaptive training."""
    EXPLORATION = 1   # Learn routing patterns
    EXPEDITION = 2    # Identify stable nodes
    CONVERGENCE = 3   # Train for accuracy
    MASTERY = 4       # Observer active


@dataclass
class NodeOfInterest:
    """A discovered region of the possibility space worth exploring."""
    tile_id: int
    signature_position: torch.Tensor
    entropy_level: float
    stability: float  # How stable is routing to this node?
    operations: List[str]  # Which operations gravitate here?
    discovery_epoch: int
    
    def __repr__(self):
        return f"Node(tile={self.tile_id}, entropy={self.entropy_level:.3f}, stability={self.stability:.3f})"


@dataclass  
class PhaseMetrics:
    """Metrics collected during a phase."""
    phase: Phase
    epochs: int
    
    # Entropic metrics
    entropy_history: List[float] = field(default_factory=list)
    entropy_variance: float = 0.0
    entropic_harmony: float = 0.0
    
    # Topology metrics
    nodes_discovered: List[NodeOfInterest] = field(default_factory=list)
    signature_movement: float = 0.0
    routing_stability: float = 0.0
    
    # Performance metrics (phases 3-4)
    accuracy_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    final_accuracy: float = 0.0

    # Observer metrics (phase 4)
    interventions: int = 0
    success_detections: int = 0


@dataclass
class JourneyContext:
    """
    Context accumulated from observing phases 1-3.

    Used by the observer in phase 4 to provide targeted interventions
    based on model-specific behavior patterns.
    """
    # From Exploration
    natural_entropy_level: float = 0.0
    preferred_tiles: List[int] = field(default_factory=list)
    exploration_breadth: float = 0.0

    # From Expedition
    nodes_of_interest: List[NodeOfInterest] = field(default_factory=list)
    topology_map: Optional[torch.Tensor] = None
    basin_structure: Dict[int, List[int]] = field(default_factory=dict)

    # From Convergence
    struggle_points: Dict[str, float] = field(default_factory=dict)  # op -> difficulty
    convergence_rate: float = 0.0
    final_pre_mastery_accuracy: float = 0.0

    # Full observation history
    all_observations: List[ObservationFrame] = field(default_factory=list)


class EntropyBalanceLoss(nn.Module):
    """
    Loss function for Phase 1: Exploration.

    Encourages balanced routing entropy - neither collapsed (all to one tile)
    nor chaotic (unstable). Combines:

    - Diversity reward: spread usage across tiles
    - Variance penalty: stable entropy over time
    - Collapse penalty: avoid degenerate routing
    """
    
    def __init__(
        self,
        target_entropy: float = 0.7,  # Target routing entropy (0-1 normalized)
        diversity_weight: float = 1.0,
        variance_penalty: float = 0.5,
        collapse_threshold: float = 0.3,
    ):
        super().__init__()
        self.target_entropy = target_entropy
        self.diversity_weight = diversity_weight
        self.variance_penalty = variance_penalty
        self.collapse_threshold = collapse_threshold
        
    def forward(
        self,
        routing_weights: torch.Tensor,  # [batch, num_tiles]
        signature_diversity: float,
        entropy_history: List[float],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute entropic harmony loss.
        
        Returns (loss, metrics_dict)
        """
        # Current routing entropy
        entropy = -(routing_weights * (routing_weights + 1e-8).log()).sum(dim=-1).mean()
        max_entropy = np.log(routing_weights.size(-1))
        normalized_entropy = entropy / max_entropy
        
        # Diversity reward: encourage spread across tiles
        tile_usage = routing_weights.mean(dim=0)  # [num_tiles]
        usage_entropy = -(tile_usage * (tile_usage + 1e-8).log()).sum()
        diversity_reward = usage_entropy / max_entropy
        
        # Variance penalty: we want STABLE entropy, not chaotic
        if len(entropy_history) > 5:
            recent = torch.tensor(entropy_history[-10:])
            variance_penalty = recent.var()
        else:
            variance_penalty = torch.tensor(0.0)
        
        # Collapse penalty: if entropy drops too low, something's wrong
        collapse_penalty = F.relu(self.collapse_threshold - normalized_entropy) ** 2
        
        # Target entropy: gentle pull toward target
        target_loss = (normalized_entropy - self.target_entropy) ** 2
        
        # Combine
        harmony = (
            self.diversity_weight * diversity_reward
            - self.variance_penalty * variance_penalty
            - collapse_penalty
            - 0.1 * target_loss
        )
        
        # We MAXIMIZE harmony, so MINIMIZE negative harmony
        loss = -harmony + 0.1 * target_loss  # Small accuracy component
        
        metrics = {
            'entropy': normalized_entropy.item(),
            'diversity': diversity_reward.item(),
            'variance': variance_penalty.item() if torch.is_tensor(variance_penalty) else variance_penalty,
            'collapse_penalty': collapse_penalty.item(),
            'harmony': harmony.item(),
        }
        
        return loss, metrics


class AdaptiveTrainingPipeline:
    """
    Multi-Phase Adaptive Training Pipeline.

    Phase 1: EXPLORATION (observer watches)
        - Optimize for routing entropy balance
        - Identify natural routing patterns
        - No task accuracy pressure

    Phase 2: EXPEDITION (observer watches)
        - Explore around stable nodes from Phase 1
        - Map the routing topology
        - Identify promising regions

    Phase 3: CONVERGENCE (observer watches)
        - Optimize for task accuracy
        - Model has explored the routing space
        - Build working capability

    Phase 4: MASTERY (observer active)
        - Observer has full context from phases 1-3
        - Apply targeted interventions
        - Final accuracy optimization
    """

    def __init__(
        self,
        model: nn.Module,
        tile_bank: ProgrammableTileBank,
        guardian: TrainingObserver,  # Accept TrainingObserver (or GuardianAngel alias)
        optimizer_fn: Callable,  # Function that creates optimizer
        task_loss_fn: nn.Module,
        device: str = 'cuda',
        total_epochs: int = 128,
        phase_schedule: Tuple[int, int, int, int] = (32, 32, 32, 32),
        verbose: bool = True,
    ):
        self.model = model.to(device)
        self.tile_bank = tile_bank.to(device)
        self.observer = guardian.to(device)
        # Backwards compatibility alias
        self.guardian = self.observer
        self.optimizer_fn = optimizer_fn
        self.task_loss_fn = task_loss_fn
        self.device = device
        self.total_epochs = total_epochs
        self.phase_schedule = phase_schedule
        self.verbose = verbose

        # Journey context - accumulated from watching phases 1-3
        self.journey = JourneyContext()

        # Phase metrics
        self.phase_metrics: Dict[Phase, PhaseMetrics] = {}

        # Entropy balance loss for Phase 1
        self.entropy_loss = EntropyBalanceLoss()
        # Backwards compatibility alias
        self.harmony_loss = self.entropy_loss
        
        # Current state
        self.current_phase = Phase.EXPLORATION
        self.global_epoch = 0
        
        # Save initial state
        self.tile_bank.save_initial_state()
        
    def detect_nodes_of_interest(
        self,
        observations: List[ObservationFrame],
        threshold: float = 0.1
    ) -> List[NodeOfInterest]:
        """
        Detect nodes of interest from exploration observations.
        
        A node is interesting if:
        - Routing is stable (low variance across time)
        - Entropy is moderate (not collapsed, not chaotic)
        - Multiple operations gravitate there
        """
        nodes = []
        
        # Aggregate per-tile statistics
        tile_stats = {}
        for obs in observations:
            if obs.tile_activations is not None:
                for tile_id in range(len(obs.tile_activations)):
                    if tile_id not in tile_stats:
                        tile_stats[tile_id] = {
                            'activations': [],
                            'entropies': [],
                        }
                    tile_stats[tile_id]['activations'].append(
                        obs.tile_activations[tile_id].item() if torch.is_tensor(obs.tile_activations[tile_id]) else obs.tile_activations[tile_id]
                    )
                    tile_stats[tile_id]['entropies'].append(obs.routing_entropy)
        
        # Find stable, interesting tiles
        signatures = self.tile_bank.get_signatures()
        
        for tile_id, stats in tile_stats.items():
            activations = np.array(stats['activations'])
            entropies = np.array(stats['entropies'])
            
            if len(activations) < 10:
                continue
                
            # Stability = inverse of activation variance
            stability = 1.0 / (np.var(activations) + 0.01)
            
            # Average entropy when this tile is active
            avg_entropy = np.mean(entropies)
            
            # Is this node interesting?
            if stability > threshold and 0.3 < avg_entropy < 0.9:
                node = NodeOfInterest(
                    tile_id=tile_id,
                    signature_position=signatures[tile_id].clone(),
                    entropy_level=avg_entropy,
                    stability=min(stability, 10.0),  # Cap for sanity
                    operations=[],  # Would need op tracking to fill this
                    discovery_epoch=self.global_epoch,
                )
                nodes.append(node)
        
        # Sort by stability * entropy balance
        nodes.sort(key=lambda n: n.stability * (1 - abs(n.entropy_level - 0.6)), reverse=True)
        
        return nodes[:self.tile_bank.num_tiles // 2]  # Top half
    
    def run_phase(
        self,
        phase: Phase,
        train_loader,
        epochs: int,
        lr: float,
    ) -> PhaseMetrics:
        """Run a single phase of the pipeline."""
        
        print(f"\n{'='*70}")
        print(f"PHASE {phase.value}: {phase.name}")
        print(f"{'='*70}")
        print(f"Epochs: {epochs} | LR: {lr}")
        print(f"Observer: {'ACTIVE' if phase == Phase.MASTERY else 'watching'}")
        print(f"{'='*70}\n")
        
        metrics = PhaseMetrics(phase=phase, epochs=epochs)
        
        # Create optimizer for this phase
        optimizer = self.optimizer_fn(self.model.parameters(), lr=lr)
        
        entropy_history = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_entropy = 0.0
            num_batches = 0
            
            for batch in train_loader:
                # Move to device
                if isinstance(batch, (list, tuple)):
                    batch = tuple(b.to(self.device) if torch.is_tensor(b) else b for b in batch)
                
                optimizer.zero_grad()
                
                # Forward pass - get outputs and routing info
                outputs = self.model(*batch[:-1])
                targets = batch[-1]
                
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    predictions, routing_info = outputs[0], outputs[1]
                    aux_loss = outputs[2] if len(outputs) > 2 else {'total_aux': 0}
                else:
                    predictions = outputs
                    routing_info = {}
                    aux_loss = {'total_aux': 0}
                
                # Phase-specific loss
                if phase == Phase.EXPLORATION:
                    # Entropic harmony - we care about EXPLORATION not accuracy
                    routing_weights = routing_info.get('weights', torch.ones(1, self.tile_bank.num_tiles) / self.tile_bank.num_tiles)
                    harmony_loss, harmony_metrics = self.harmony_loss(
                        routing_weights.to(self.device),
                        signature_diversity=1.0,
                        entropy_history=entropy_history,
                    )
                    # Small task loss to prevent total chaos
                    task_loss = self.task_loss_fn(predictions, targets)
                    loss = 0.3 * task_loss + 0.7 * harmony_loss
                    epoch_entropy += harmony_metrics['entropy']
                    
                elif phase == Phase.EXPEDITION:
                    # Focus on exploring around nodes of interest
                    task_loss = self.task_loss_fn(predictions, targets)
                    # Encourage visiting discovered nodes
                    node_bonus = 0.0
                    if self.journey.nodes_of_interest:
                        # Bonus for routing near known good nodes
                        pass  # TODO: implement node proximity bonus
                    loss = 0.5 * task_loss + aux_loss.get('total_aux', 0)
                    
                elif phase == Phase.CONVERGENCE:
                    # Now we care about accuracy
                    task_loss = self.task_loss_fn(predictions, targets)
                    loss = task_loss + aux_loss.get('total_aux', 0)
                    
                else:  # MASTERY
                    # Full accuracy with observer intervention
                    task_loss = self.task_loss_fn(predictions, targets)
                    loss = task_loss + aux_loss.get('total_aux', 0)
                
                loss.backward()
                optimizer.step()
                
                # Compute accuracy
                with torch.no_grad():
                    if predictions.dim() > 1 and predictions.size(-1) > 1:
                        pred_classes = predictions.argmax(dim=-1)
                    else:
                        pred_classes = (predictions > 0.5).long().squeeze()
                    
                    # Handle different target shapes
                    if targets.dim() > 1:
                        target_classes = targets.argmax(dim=-1) if targets.size(-1) > 1 else targets.squeeze()
                    else:
                        target_classes = targets
                    
                    acc = (pred_classes == target_classes).float().mean().item() * 100
                
                epoch_loss += loss.item()
                epoch_acc += acc
                num_batches += 1
                
                # Create observation (observer always watching)
                routing_entropy = routing_info.get('entropy', torch.tensor(0.5))
                if torch.is_tensor(routing_entropy):
                    routing_entropy = routing_entropy.item()

                obs = ObservationFrame(
                    epoch=self.global_epoch,
                    step=num_batches,
                    routing_entropy=routing_entropy,
                    loss=loss.item(),
                    accuracy=acc,
                )

                # Observer watches (but only intervenes in Phase 4)
                self.observer.observe(obs)
                self.journey.all_observations.append(obs)

                # Phase 4: Observer intervention
                if phase == Phase.MASTERY:
                    should_intervene, prediction = self.observer.should_intervene()
                    if should_intervene:
                        self.observer.intervene(
                            self.tile_bank, prediction,
                            epoch=self.global_epoch, step=num_batches,
                            current_accuracy=acc
                        )
                        metrics.interventions += 1
                    if prediction.get('celebrating', False):
                        metrics.success_detections += 1
            
            # Epoch stats
            epoch_loss /= num_batches
            epoch_acc /= num_batches
            epoch_entropy /= max(num_batches, 1)
            
            entropy_history.append(epoch_entropy if phase == Phase.EXPLORATION else routing_entropy)
            metrics.entropy_history.append(epoch_entropy if phase == Phase.EXPLORATION else routing_entropy)
            metrics.accuracy_history.append(epoch_acc)
            metrics.loss_history.append(epoch_loss)
            
            # Progress output
            if phase == Phase.EXPLORATION:
                print(f"  Epoch {epoch+1:3d}/{epochs}: Entropy={epoch_entropy:.3f} Acc={epoch_acc:.1f}%")
            elif phase == Phase.MASTERY:
                status = f"[{metrics.success_detections} success, {metrics.interventions} interventions]"
                print(f"  Epoch {epoch+1:3d}/{epochs}: Loss={epoch_loss:.4f} Acc={epoch_acc:.1f}% {status}")
            else:
                print(f"  Epoch {epoch+1:3d}/{epochs}: Loss={epoch_loss:.4f} Acc={epoch_acc:.1f}%")
            
            self.global_epoch += 1
        
        # Phase complete - compute final metrics
        metrics.final_accuracy = metrics.accuracy_history[-1] if metrics.accuracy_history else 0
        metrics.entropy_variance = np.var(metrics.entropy_history) if metrics.entropy_history else 0
        metrics.signature_movement = self.tile_bank.get_total_movement()
        
        # Phase-specific post-processing
        if phase == Phase.EXPLORATION:
            # Detect nodes of interest
            nodes = self.detect_nodes_of_interest(self.journey.all_observations)
            metrics.nodes_discovered = nodes
            self.journey.nodes_of_interest = nodes
            self.journey.natural_entropy_level = np.mean(metrics.entropy_history)
            print(f"\n  Discovered {len(nodes)} nodes of interest")
            
        elif phase == Phase.CONVERGENCE:
            # Record struggle points
            self.journey.final_pre_mastery_accuracy = metrics.final_accuracy
            self.journey.convergence_rate = (
                metrics.accuracy_history[-1] - metrics.accuracy_history[0]
            ) / epochs if epochs > 0 else 0
            
        return metrics
    
    def run(
        self,
        train_loader,
        eval_loader=None,
        lr_schedule: Tuple[float, float, float, float] = (0.005, 0.003, 0.002, 0.001),
    ) -> Dict:
        """
        Run the complete 4-phase adaptive training pipeline.

        Returns comprehensive results including per-phase metrics.
        """
        start_time = time.time()

        print("\n" + "="*70)
        print("           ADAPTIVE TRAINING PIPELINE")
        print("="*70)
        print("  Phase 1: Exploration  - Learn routing patterns")
        print("  Phase 2: Expedition   - Map stable nodes")
        print("  Phase 3: Convergence  - Train for accuracy")
        print("  Phase 4: Mastery      - Observer active")
        print("="*70)
        
        phases = [Phase.EXPLORATION, Phase.EXPEDITION, Phase.CONVERGENCE, Phase.MASTERY]
        
        for i, phase in enumerate(phases):
            metrics = self.run_phase(
                phase=phase,
                train_loader=train_loader,
                epochs=self.phase_schedule[i],
                lr=lr_schedule[i],
            )
            self.phase_metrics[phase] = metrics
            self.current_phase = phases[i+1] if i < 3 else Phase.MASTERY
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_eval = None
        if eval_loader is not None:
            final_eval = self._evaluate(eval_loader)
        
        # Summary
        print("\n" + "="*70)
        print("                 TRAINING COMPLETE")
        print("="*70)

        for phase in phases:
            m = self.phase_metrics[phase]
            if phase == Phase.MASTERY:
                print(f"  {phase.name:12s}: {m.final_accuracy:.1f}% | {m.success_detections} success | {m.interventions} interventions")
            else:
                print(f"  {phase.name:12s}: {m.final_accuracy:.1f}%")

        if final_eval:
            print(f"\n  FINAL EVAL: {final_eval['accuracy']:.1f}%")

        print(f"\n  Total time: {total_time:.1f}s")
        print(f"  Total observations: {len(self.journey.all_observations)}")
        print(f"  Nodes discovered: {len(self.journey.nodes_of_interest)}")
        print("="*70 + "\n")

        return {
            'phase_metrics': self.phase_metrics,
            'journey': self.journey,
            'final_eval': final_eval,
            'total_time': total_time,
            'observer_stats': self.observer.get_stats(),
        }
    
    def _evaluate(self, eval_loader) -> Dict:
        """Evaluate model."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                if isinstance(batch, (list, tuple)):
                    batch = tuple(b.to(self.device) if torch.is_tensor(b) else b for b in batch)
                
                outputs = self.model(*batch[:-1])
                targets = batch[-1]
                
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
                if predictions.dim() > 1 and predictions.size(-1) > 1:
                    pred_classes = predictions.argmax(dim=-1)
                else:
                    pred_classes = (predictions > 0.5).long().squeeze()
                
                if targets.dim() > 1:
                    target_classes = targets.argmax(dim=-1) if targets.size(-1) > 1 else targets.squeeze()
                else:
                    target_classes = targets
                
                total_correct += (pred_classes == target_classes).sum().item()
                total_samples += len(targets)
        
        return {'accuracy': total_correct / total_samples * 100}


# Backwards compatibility aliases
HALOPipeline = AdaptiveTrainingPipeline
EntropicHarmonyLoss = EntropyBalanceLoss
