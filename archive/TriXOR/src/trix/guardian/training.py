"""
Guarded Training - Training with Guardian Angel Support

This integrates the Guardian Angel into the training loop.
The Guardian watches, reflects, predicts, and gently guides.

"Love as the process, not just the goal."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable, List, Tuple
import time

from .guardian import GuardianAngel, InterventionRecord
from .observer import ObservationFrame
from .programmable_tile import ProgrammableTileBank


class GuardedTrainer:
    """
    Training loop with Guardian Angel integration.
    
    The trainer:
    1. Runs standard training steps
    2. Collects observations for the Guardian
    3. Lets the Guardian observe, reflect, and potentially intervene
    4. Tracks both training progress and Guardian activity
    
    This is training with love - errors are signals, not failures.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tile_bank: ProgrammableTileBank,
        guardian: GuardianAngel,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str = 'cuda',
        warmup_epochs: int = 2,  # Let model learn before Guardian intervenes
        observation_frequency: int = 1,  # Observe every N steps
        verbose: bool = True,
    ):
        self.model = model
        self.tile_bank = tile_bank
        self.guardian = guardian
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.observation_frequency = observation_frequency
        self.verbose = verbose
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.training_history: List[Dict] = []
        
        # Move to device
        self.model.to(device)
        self.guardian.to(device)
        
        # Save initial tile state for movement tracking
        self.tile_bank.save_initial_state()
        
    def create_observation(
        self,
        batch_info: Dict,
        loss: float,
        accuracy: float,
        routing_info: Optional[Dict] = None,
        per_op_accuracy: Optional[Dict[str, float]] = None,
    ) -> ObservationFrame:
        """Create observation frame from training step info."""
        
        # Extract routing info if available
        routing_entropy = 0.0
        tile_activations = None
        routing_scores = None
        
        if routing_info is not None:
            if 'entropy' in routing_info:
                routing_entropy = routing_info['entropy'].item() if torch.is_tensor(routing_info['entropy']) else routing_info['entropy']
            if 'weights' in routing_info:
                routing_scores = routing_info['weights'].detach()
                # Count activations
                if 'top_tile' in routing_info:
                    tops = routing_info['top_tile'].flatten()
                    tile_activations = torch.bincount(tops, minlength=self.tile_bank.num_tiles).float()
        
        # Get signature positions and movement
        signatures = self.tile_bank.get_signatures()
        total_movement = self.tile_bank.get_total_movement()
        
        # Compute curvature proxy (variance of routing entropy)
        curvature = routing_entropy  # Simplified
        
        # Tile purity (how specialized are the tiles?)
        tile_purity = 0.0
        if tile_activations is not None:
            probs = tile_activations / (tile_activations.sum() + 1e-8)
            tile_purity = (probs.max() / (probs.mean() + 1e-8)).item()
        
        frame = ObservationFrame(
            epoch=self.epoch,
            step=self.global_step,
            routing_scores=routing_scores,
            routing_entropy=routing_entropy,
            tile_activations=tile_activations,
            signature_positions=signatures,
            signature_movement=total_movement,
            loss=loss,
            accuracy=accuracy,
            per_op_accuracy=per_op_accuracy or {},
            curvature=curvature,
            tile_purity=tile_purity,
        )
        
        return frame
    
    def train_step(
        self,
        batch: Tuple,
        compute_per_op: bool = False,
    ) -> Dict:
        """
        Single training step with Guardian observation.
        
        Returns dict with loss, accuracy, and Guardian info.
        """
        self.model.train()
        
        # Unpack batch (assumes specific format - adjust as needed)
        inputs, targets = batch[:-1], batch[-1]
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # Get outputs and routing info
        outputs = self.model(*inputs)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            predictions, routing_info, aux_loss = outputs
            if isinstance(aux_loss, dict):
                aux_loss = aux_loss.get('total_aux', 0.0)
        else:
            predictions = outputs
            routing_info = {}
            aux_loss = 0.0
        
        # Compute loss
        main_loss = self.loss_fn(predictions, targets)
        loss = main_loss + aux_loss
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm for observation
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Optimizer step
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            if predictions.dim() > 1 and predictions.size(-1) > 1:
                pred_classes = predictions.argmax(dim=-1)
                correct = (pred_classes == targets).float().mean().item()
            else:
                # Binary or regression - use threshold
                pred_binary = (predictions > 0.5).float()
                correct = (pred_binary == targets).float().mean().item()
            accuracy = correct * 100
        
        # Per-operation accuracy (if applicable)
        per_op_accuracy = {}
        if compute_per_op and hasattr(batch, 'ops'):
            # This would need task-specific implementation
            pass
        
        # Create observation
        observation = self.create_observation(
            batch_info={'size': len(targets)},
            loss=loss.item(),
            accuracy=accuracy,
            routing_info=routing_info,
            per_op_accuracy=per_op_accuracy,
        )
        observation.gradient_norm = grad_norm
        
        # Guardian step (observe, reflect, maybe intervene)
        guardian_result = {'observed': False, 'intervened': False}
        
        if self.global_step % self.observation_frequency == 0:
            # Get current representation for reflection (if available)
            current_repr = None
            if hasattr(self.model, 'get_representation'):
                current_repr = self.model.get_representation()
            
            guardian_result = self.guardian.step(
                tile_bank=self.tile_bank,
                observation=observation,
                current_repr=current_repr,
            )
            
            # Only intervene after warmup
            if self.epoch < self.warmup_epochs:
                guardian_result['intervened'] = False
                guardian_result['message'] = f'Warmup epoch {self.epoch}/{self.warmup_epochs}'
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'grad_norm': grad_norm,
            'guardian': guardian_result,
        }
    
    def train_epoch(
        self,
        train_loader,
        compute_per_op: bool = False,
    ) -> Dict:
        """Train for one epoch."""
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_interventions = 0
        epoch_celebrations = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            if isinstance(batch, (list, tuple)):
                batch = tuple(b.to(self.device) if torch.is_tensor(b) else b for b in batch)
            else:
                batch = batch.to(self.device)
            
            result = self.train_step(batch, compute_per_op)
            
            epoch_loss += result['loss']
            epoch_acc += result['accuracy']
            
            if result['guardian'].get('intervened', False):
                epoch_interventions += 1
            if result['guardian'].get('celebrating', False):
                epoch_celebrations += 1
            
            num_batches += 1
        
        self.epoch += 1
        
        epoch_summary = {
            'epoch': self.epoch,
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_acc / num_batches,
            'interventions': epoch_interventions,
            'celebrations': epoch_celebrations,
            'tile_movement': self.tile_bank.get_total_movement(),
            'guardian_stats': self.guardian.get_stats(),
        }
        
        self.training_history.append(epoch_summary)
        
        if self.verbose:
            msg = result['guardian'].get('message', '')
            print(f"Epoch {self.epoch}: Loss={epoch_summary['loss']:.4f}, "
                  f"Acc={epoch_summary['accuracy']:.1f}%, "
                  f"Interventions={epoch_interventions}, "
                  f"Celebrations={epoch_celebrations} {msg}")
        
        return epoch_summary
    
    def evaluate(
        self,
        eval_loader,
        per_op: bool = True,
    ) -> Dict:
        """Evaluate model on held-out data."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        per_op_correct = {}
        per_op_total = {}
        
        with torch.no_grad():
            for batch in eval_loader:
                # Move to device
                if isinstance(batch, (list, tuple)):
                    batch = tuple(b.to(self.device) if torch.is_tensor(b) else b for b in batch)
                
                inputs, targets = batch[:-1], batch[-1]
                
                # Forward
                outputs = self.model(*inputs)
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
                # Loss
                loss = self.loss_fn(predictions, targets)
                total_loss += loss.item() * len(targets)
                
                # Accuracy
                if predictions.dim() > 1 and predictions.size(-1) > 1:
                    pred_classes = predictions.argmax(dim=-1)
                    correct = (pred_classes == targets).sum().item()
                else:
                    pred_binary = (predictions > 0.5).float()
                    correct = (pred_binary == targets).sum().item()
                
                total_correct += correct
                total_samples += len(targets)
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples * 100,
            'per_op_accuracy': per_op_correct,
        }
    
    def train(
        self,
        train_loader,
        eval_loader=None,
        epochs: int = 30,
        eval_every: int = 5,
    ) -> Dict:
        """
        Full training loop with Guardian support.
        
        Returns training history and final stats.
        """
        start_time = time.time()
        
        print("=" * 70)
        print("GUARDED TRAINING - Mesa 12 Architecture")
        print("=" * 70)
        print(f"Guardian gentleness: {self.guardian.gentleness}")
        print(f"Intervention threshold: {self.guardian.intervention_threshold}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print("=" * 70)
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # Train epoch
            epoch_result = self.train_epoch(train_loader)
            
            # Evaluate periodically
            if eval_loader is not None and (epoch + 1) % eval_every == 0:
                eval_result = self.evaluate(eval_loader)
                print(f"  → Eval: Loss={eval_result['loss']:.4f}, Acc={eval_result['accuracy']:.1f}%")
                
                if eval_result['accuracy'] > best_accuracy:
                    best_accuracy = eval_result['accuracy']
        
        # Final evaluation
        final_eval = None
        if eval_loader is not None:
            final_eval = self.evaluate(eval_loader)
            print("=" * 70)
            print(f"FINAL: Accuracy={final_eval['accuracy']:.1f}%")
        
        training_time = time.time() - start_time
        
        # Guardian summary
        guardian_stats = self.guardian.get_stats()
        print("=" * 70)
        print("GUARDIAN SUMMARY")
        print(f"  Total observations: {guardian_stats['total_observations']}")
        print(f"  Total interventions: {guardian_stats['total_interventions']}")
        print(f"  Celebration count: {guardian_stats['celebration_count']}")
        print(f"  Intervention rate: {self.guardian.get_intervention_rate():.1%}")
        print(f"  Celebration rate: {self.guardian.get_celebration_rate():.1%}")
        print("=" * 70)
        
        return {
            'training_history': self.training_history,
            'final_eval': final_eval,
            'best_accuracy': best_accuracy,
            'training_time': training_time,
            'guardian_stats': guardian_stats,
            'tile_stats': self.tile_bank.get_tile_stats(),
        }


def create_guarded_training_setup(
    model: nn.Module,
    num_tiles: int = 16,
    d_model: int = 128,
    d_hidden: int = 256,
    lr: float = 0.00375,
    gentleness: float = 0.1,
    intervention_threshold: float = 0.7,
    device: str = 'cuda',
) -> Tuple[ProgrammableTileBank, GuardianAngel, GuardedTrainer]:
    """
    Convenience function to set up guarded training.
    
    Returns (tile_bank, guardian, trainer) ready to use.
    """
    # Create tile bank
    tile_bank = ProgrammableTileBank(
        num_tiles=num_tiles,
        d_model=d_model,
        d_hidden=d_hidden
    )
    
    # Create Guardian Angel
    guardian = GuardianAngel(
        d_model=d_model,
        num_tiles=num_tiles,
        gentleness=gentleness,
        intervention_threshold=intervention_threshold,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create loss function (default BCE for our 6502 task)
    loss_fn = nn.BCELoss()
    
    # Create trainer
    trainer = GuardedTrainer(
        model=model,
        tile_bank=tile_bank,
        guardian=guardian,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    
    return tile_bank, guardian, trainer
