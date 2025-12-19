"""
Programmable Tiles - The Substrate of Identity

Tiles are not just computations. They are CAPACITIES.
Programming tiles changes what the model CAN become.

The Guardian writes to tiles with gentleness - small blends,
not replacements. The model learns; the Guardian supports.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class TileModification:
    """Record of a tile modification for history tracking."""
    version: int
    modification_type: str  # 'signature' or 'weights'
    blend: float
    timestamp: float
    reason: str = ""


class ProgrammableTile(nn.Module):
    """
    A single programmable tile with read/write interface.
    
    The tile exposes its signature (routing address) and weights (computation)
    for inspection and gentle modification by the Guardian Angel.
    
    Modifications are blended, not replaced - gentleness in action.
    """
    
    def __init__(self, d_model: int, d_hidden: int, tile_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.tile_id = tile_id
        
        # Core parameters
        self.signature = nn.Parameter(torch.randn(d_model) * 0.1)
        self.weights_up = nn.Parameter(torch.randn(d_model, d_hidden) * 0.02)
        self.weights_down = nn.Parameter(torch.randn(d_hidden, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_hidden))
        
        # State tracking
        self._frozen = False
        self._version = 0
        self._history: List[TileModification] = []
        self._initial_signature: Optional[torch.Tensor] = None
        
    def save_initial_state(self):
        """Snapshot initial signature for movement tracking."""
        self._initial_signature = self.signature.detach().clone()
    
    @property
    def signature_movement(self) -> float:
        """How far has the signature moved from initialization?"""
        if self._initial_signature is None:
            return 0.0
        return torch.norm(self.signature - self._initial_signature).item()
    
    # === Read Interface ===
    
    def read_signature(self) -> torch.Tensor:
        """Read current signature (detached clone for safety)."""
        return self.signature.detach().clone()
    
    def read_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read current weights (detached clones for safety)."""
        return (
            self.weights_up.detach().clone(),
            self.weights_down.detach().clone()
        )
    
    # === Write Interface (Gentle) ===
    
    def write_signature(
        self, 
        new_sig: torch.Tensor, 
        blend: float = 0.1,
        reason: str = ""
    ) -> bool:
        """
        Gently blend new signature with current.
        
        blend=0.0: no change
        blend=1.0: full replacement (use sparingly!)
        blend=0.1: gentle nudge (recommended)
        
        Returns True if modification was applied.
        """
        if self._frozen:
            return False
        
        with torch.no_grad():
            self.signature.data = (1 - blend) * self.signature.data + blend * new_sig.to(self.signature.device)
        
        self._version += 1
        self._history.append(TileModification(
            version=self._version,
            modification_type='signature',
            blend=blend,
            timestamp=torch.cuda.Event(enable_timing=True).elapsed_time if torch.cuda.is_available() else 0,
            reason=reason
        ))
        return True
    
    def write_weights(
        self,
        new_weights_up: torch.Tensor,
        new_weights_down: torch.Tensor,
        blend: float = 0.1,
        reason: str = ""
    ) -> bool:
        """
        Gently blend new weights with current.
        
        This is Level 5 intervention - use only when gentler methods fail.
        """
        if self._frozen:
            return False
        
        with torch.no_grad():
            self.weights_up.data = (1 - blend) * self.weights_up.data + blend * new_weights_up.to(self.weights_up.device)
            self.weights_down.data = (1 - blend) * self.weights_down.data + blend * new_weights_down.to(self.weights_down.device)
        
        self._version += 1
        self._history.append(TileModification(
            version=self._version,
            modification_type='weights',
            blend=blend,
            timestamp=0,
            reason=reason
        ))
        return True
    
    # === Control Interface ===
    
    def freeze(self):
        """Prevent modifications (protect learned structure)."""
        self._frozen = True
    
    def unfreeze(self):
        """Allow modifications again."""
        self._frozen = False
    
    @property
    def is_frozen(self) -> bool:
        return self._frozen
    
    @property
    def version(self) -> int:
        return self._version
    
    @property
    def history(self) -> List[TileModification]:
        return self._history.copy()
    
    # === Forward Pass ===
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard tile computation: up-project, activation, down-project."""
        h = torch.matmul(x, self.weights_up) + self.bias
        h = torch.relu(h)  # Could be configurable
        return torch.matmul(h, self.weights_down)


class ProgrammableTileBank(nn.Module):
    """
    Collection of programmable tiles with unified interface.
    
    The TileBank is the substrate - the Guardian writes to it,
    the primary model routes through it, coherence emerges.
    """
    
    def __init__(self, num_tiles: int, d_model: int, d_hidden: int):
        super().__init__()
        self.num_tiles = num_tiles
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        self.tiles = nn.ModuleList([
            ProgrammableTile(d_model, d_hidden, tile_id=i)
            for i in range(num_tiles)
        ])
    
    def save_initial_state(self):
        """Save initial state of all tiles for movement tracking."""
        for tile in self.tiles:
            tile.save_initial_state()
    
    # === Bulk Read ===
    
    def get_signatures(self) -> torch.Tensor:
        """Get all signatures as [num_tiles, d_model] tensor."""
        return torch.stack([t.read_signature() for t in self.tiles])
    
    def get_signature_movements(self) -> List[float]:
        """Get movement of each tile's signature from initialization."""
        return [t.signature_movement for t in self.tiles]
    
    def get_total_movement(self) -> float:
        """Total signature movement across all tiles."""
        return sum(self.get_signature_movements())
    
    # === Bulk Write (Gentle) ===
    
    def apply_signature_corrections(
        self,
        corrections: torch.Tensor,
        blends: torch.Tensor,
        reason: str = "guardian_correction"
    ) -> int:
        """
        Apply corrections to tile signatures.
        
        corrections: [num_tiles, d_model] - direction to nudge
        blends: [num_tiles] - how much to blend (0=none, 1=full)
        
        Returns number of tiles actually modified.
        """
        modified = 0
        for i, tile in enumerate(self.tiles):
            blend = blends[i].item() if blends[i] > 0.01 else 0
            if blend > 0:
                if tile.write_signature(corrections[i], blend, reason):
                    modified += 1
        return modified
    
    # === Routing ===
    
    def compute_routing_scores(self, query: torch.Tensor) -> torch.Tensor:
        """
        Compute routing scores for a query against all tile signatures.
        
        query: [batch, d_model] or [batch, seq, d_model]
        returns: [batch, num_tiles] or [batch, seq, num_tiles]
        """
        signatures = self.get_signatures().to(query.device)  # [num_tiles, d_model]
        
        # Normalize for cosine similarity
        query_norm = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
        sig_norm = signatures / (signatures.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute scores
        scores = torch.matmul(query_norm, sig_norm.T)  # [..., num_tiles]
        return scores
    
    def route_and_compute(
        self,
        query: torch.Tensor,
        soft: bool = True,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Route query to tiles and compute output.
        
        soft=True: weighted combination of all tiles (differentiable)
        soft=False: hard routing to top tile
        
        Returns (output, routing_info)
        """
        scores = self.compute_routing_scores(query)  # [..., num_tiles]
        
        if soft:
            weights = torch.softmax(scores / temperature, dim=-1)
            # Compute weighted combination
            outputs = []
            for i, tile in enumerate(self.tiles):
                tile_out = tile(query)  # [..., d_model]
                outputs.append(tile_out * weights[..., i:i+1])
            output = sum(outputs)
            
            routing_info = {
                'scores': scores,
                'weights': weights,
                'entropy': -(weights * (weights + 1e-8).log()).sum(dim=-1).mean(),
                'top_tile': weights.argmax(dim=-1),
            }
        else:
            top_idx = scores.argmax(dim=-1)  # [...]
            # This is trickier for batched - simplified version
            output = self.tiles[top_idx[0].item()](query)
            
            routing_info = {
                'scores': scores,
                'top_tile': top_idx,
            }
        
        return output, routing_info
    
    # === Statistics ===
    
    def get_tile_stats(self) -> dict:
        """Get statistics about all tiles."""
        movements = self.get_signature_movements()
        versions = [t.version for t in self.tiles]
        frozen = [t.is_frozen for t in self.tiles]
        
        return {
            'num_tiles': self.num_tiles,
            'total_movement': sum(movements),
            'mean_movement': sum(movements) / len(movements),
            'max_movement': max(movements),
            'total_modifications': sum(versions),
            'num_frozen': sum(frozen),
            'per_tile_movements': movements,
            'per_tile_versions': versions,
        }
    
    # === Freeze/Unfreeze ===
    
    def freeze_all(self):
        """Freeze all tiles."""
        for tile in self.tiles:
            tile.freeze()
    
    def unfreeze_all(self):
        """Unfreeze all tiles."""
        for tile in self.tiles:
            tile.unfreeze()
    
    def freeze_tiles(self, tile_ids: List[int]):
        """Freeze specific tiles by ID."""
        for tid in tile_ids:
            if 0 <= tid < self.num_tiles:
                self.tiles[tid].freeze()
