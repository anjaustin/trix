"""
Sparse Training Components for TriX (Option B)

Train with gated computation from the start.
Each tile learns to stand alone.
"""

import sys
from pathlib import Path

# Handle both import and direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from trix.kernel import TriXLinear, STESign


class SparseTriXFFN(nn.Module):
    """
    FFN that trains with sparse (gated) computation.
    
    Unlike dense training where all tiles compute on all inputs,
    sparse training routes each input to ONE tile. Only that tile
    computes and receives gradients.
    
    This forces tiles to learn independent, complete representations.
    
    Training: Uses PyTorch sparse forward with STE for gradients.
    Inference: Can use NEON kernel for 4x speedup (call pack() first).
    
    Args:
        d_model: Input/output dimension
        expansion: FFN expansion factor (default: 4)
        num_tiles: Number of routing tiles (default: 4)
        dropout: Dropout rate
        balance_weight: Weight for load balancing loss
        diversity_weight: Weight for signature diversity loss
    """
    
    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        num_tiles: int = 4,
        dropout: float = 0.1,
        balance_weight: float = 0.01,
        diversity_weight: float = 0.001,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_model * expansion
        self.num_tiles = num_tiles
        self.balance_weight = balance_weight
        self.diversity_weight = diversity_weight
        
        # Tile size for gated computation
        self.tile_size = self.d_ff // num_tiles
        
        # Core layers
        self.up_proj = TriXLinear(d_model, self.d_ff, num_tiles)
        self.down_proj = TriXLinear(self.d_ff, d_model, num_tiles)
        self.dropout = nn.Dropout(dropout)
        
        # EMA signatures for stable routing
        self.register_buffer('ema_signatures', None)
        self.ema_decay = 0.99
        
        # Packed state for inference
        self._packed = False
    
    def get_tile_signatures(self) -> torch.Tensor:
        """Get tile signatures from up_proj weights."""
        return self.up_proj.get_tile_signatures()
    
    def pack(self):
        """Pack weights for NEON-accelerated inference."""
        self.up_proj.pack()
        self.down_proj.pack()
        self._packed = True
    
    def unpack(self):
        """Unpack weights for training."""
        self.up_proj.unpack()
        self.down_proj.unpack()
        self._packed = False
    
    @property
    def is_packed(self) -> bool:
        """Check if model is in packed (inference) mode."""
        return self._packed
    
    def compute_routing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute emergent routing from signatures.
        
        Args:
            x: Input tensor [batch, d_model]
            
        Returns:
            One-hot gate tensor [batch, num_tiles]
        """
        # Get current signatures
        signatures = self.get_tile_signatures()
        
        # Update EMA signatures for stability
        if self.training:
            if self.ema_signatures is None:
                self.ema_signatures = signatures.detach().clone()
            else:
                self.ema_signatures = (
                    self.ema_decay * self.ema_signatures + 
                    (1 - self.ema_decay) * signatures.detach()
                )
            # Use EMA for routing during training
            routing_sigs = self.ema_signatures
        else:
            routing_sigs = signatures
        
        # Compute alignment scores
        scores = x @ routing_sigs.T  # [batch, num_tiles]
        
        # Hard routing (one-hot)
        winner = scores.argmax(dim=-1, keepdim=True)
        gate = torch.zeros(x.shape[0], self.num_tiles, device=x.device)
        gate.scatter_(-1, winner, 1.0)
        
        return gate
    
    def gated_forward(
        self, 
        x: torch.Tensor, 
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with true sparse computation.
        
        Only computes tiles that are active in the gate.
        """
        batch = x.shape[0]
        device = x.device
        
        # Quantize weights (STE)
        up_w = STESign.apply(self.up_proj.weight)
        down_w = STESign.apply(self.down_proj.weight)
        
        # Output accumulator
        out = torch.zeros(batch, self.d_model, device=device)
        
        # Process each tile separately
        for t in range(self.num_tiles):
            # Find inputs routed to this tile
            tile_mask = gate[:, t] > 0  # [batch]
            
            if tile_mask.sum() == 0:
                continue  # No inputs for this tile
            
            # Get inputs for this tile
            tile_x = x[tile_mask]  # [tile_batch, d_model]
            
            # Tile weight slices
            up_start = t * self.tile_size
            up_end = up_start + self.tile_size
            
            tile_up_w = up_w[up_start:up_end, :]  # [tile_size, d_model]
            tile_up_scale = self.up_proj.scales[up_start:up_end]
            
            # Up projection for this tile
            hidden = torch.mm(tile_x, tile_up_w.t()) * tile_up_scale
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
            
            # Down projection - this tile's hidden to full output
            # Each tile contributes to a slice of down_proj
            down_start = t * (self.d_model // self.num_tiles)
            down_end = down_start + (self.d_model // self.num_tiles)
            
            # Actually, for proper sparse: each tile should produce full d_model output
            # Let's use the tile's corresponding down_proj weights
            tile_down_w = down_w[:, up_start:up_end]  # [d_model, tile_size]
            tile_down_scale = self.down_proj.scales
            
            tile_out = torch.mm(hidden, tile_down_w.t()) * tile_down_scale
            
            # Place outputs back
            out[tile_mask] = tile_out
        
        return out
    
    def compute_balance_loss(self, gate: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to prevent tile collapse.
        
        Penalizes uneven tile usage.
        """
        # Average usage per tile
        usage = gate.mean(dim=0)  # [num_tiles]
        
        # Coefficient of variation (std / mean)
        # Lower is better (more balanced)
        cv = usage.std() / (usage.mean() + 1e-8)
        
        return cv * self.balance_weight
    
    def compute_diversity_loss(self) -> torch.Tensor:
        """
        Compute signature diversity loss to prevent signature collapse.
        
        Encourages tiles to have different signatures.
        """
        signatures = self.get_tile_signatures()  # [num_tiles, d_model]
        
        # Pairwise similarity
        # Want signatures to be different (low similarity)
        similarity = torch.mm(signatures, signatures.t())  # [num_tiles, num_tiles]
        
        # Mask diagonal (self-similarity)
        mask = 1 - torch.eye(self.num_tiles, device=signatures.device)
        
        # Average off-diagonal similarity (want this low)
        avg_similarity = (similarity * mask).sum() / (mask.sum() + 1e-8)
        
        return avg_similarity * self.diversity_weight
    
    def neon_forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        NEON-accelerated forward pass using packed weights.
        
        Requires pack() to be called first.
        """
        from trix.kernel import trix_forward
        
        # Up projection with NEON
        hidden = trix_forward(
            x, self.up_proj.packed_weight, self.up_proj.scales,
            gate, self.d_ff, self.num_tiles
        )
        hidden = F.relu(hidden)
        
        # Down projection with NEON
        out = trix_forward(
            hidden, self.down_proj.packed_weight, self.down_proj.scales,
            gate, self.d_model, self.num_tiles
        )
        
        return out
    
    def forward(
        self, 
        x: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with sparse computation.
        
        Args:
            x: Input tensor [batch, d_model] or [batch, seq, d_model]
            gate: Optional pre-computed gate (for inference)
            
        Returns:
            output: Output tensor (same shape as input)
            gate: Routing decisions
            aux_losses: Dictionary of auxiliary losses
        """
        orig_shape = x.shape
        
        # Flatten if 3D
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
        
        # Compute routing if not provided
        if gate is None:
            gate = self.compute_routing(x)
        
        # Choose forward implementation
        if self._packed and not self.training:
            # Use NEON kernel for inference
            out = self.neon_forward(x.cpu(), gate.cpu())
            if orig_shape[-1] != out.shape[-1]:  # Handle device mismatch
                out = out.to(x.device)
        else:
            # Use PyTorch sparse forward for training
            out = self.gated_forward(x, gate)
        
        # Compute auxiliary losses
        aux_losses = {}
        if self.training:
            aux_losses['balance'] = self.compute_balance_loss(gate)
            aux_losses['diversity'] = self.compute_diversity_loss()
            aux_losses['total_aux'] = aux_losses['balance'] + aux_losses['diversity']
        
        # Restore shape
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)
            gate = gate.view(orig_shape[0], orig_shape[1], -1)
        
        return out, gate, aux_losses
    
    def get_routing_stats(self, x: torch.Tensor) -> Dict:
        """Get routing statistics for analysis."""
        with torch.no_grad():
            # Handle 3D input [batch, seq, d_model]
            if x.dim() == 3:
                B, T, C = x.shape
                x = x.reshape(B * T, C)
            
            gate = self.compute_routing(x)
            usage = gate.mean(dim=0).cpu().tolist()
            
            # Entropy (higher = more balanced)
            usage_t = torch.tensor(usage) + 1e-8
            entropy = -(usage_t * usage_t.log()).sum().item()
            
            return {
                'tile_usage': usage,
                'entropy': entropy,
                'dominant_tile': usage.index(max(usage)),
                'balance': 1 - (max(usage) - min(usage)),
            }
    
    def get_signature_diversity(self) -> float:
        """Compute signature diversity (0-1, higher = more diverse)."""
        with torch.no_grad():
            sigs = self.get_tile_signatures()
            total_diff = 0.0
            pairs = 0
            for i in range(self.num_tiles):
                for j in range(i + 1, self.num_tiles):
                    diff = (sigs[i] != sigs[j]).float().mean().item()
                    total_diff += diff
                    pairs += 1
            return total_diff / pairs if pairs > 0 else 0.0


class SparseTriXBlock(nn.Module):
    """Transformer block with sparse FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_tiles: int = 4,
        expansion: int = 4,
        dropout: float = 0.1,
        balance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = SparseTriXFFN(
            d_model, expansion=expansion, num_tiles=num_tiles,
            dropout=dropout, balance_weight=balance_weight
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        is_causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Self-attention
        normed = self.norm1(x)
        if is_causal:
            T = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask, is_causal=True)
        else:
            attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # Sparse FFN
        ffn_out, gate, aux_losses = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x, gate, aux_losses


if __name__ == "__main__":
    # Quick test
    print("Testing SparseTriXFFN...")
    
    ffn = SparseTriXFFN(d_model=64, num_tiles=4)
    x = torch.randn(8, 64)
    
    out, gate, aux = ffn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Gate shape: {gate.shape}")
    print(f"Gate sum (should be 1): {gate.sum(dim=-1)}")
    print(f"Aux losses: {aux}")
    print(f"Routing stats: {ffn.get_routing_stats(x)}")
    print(f"Signature diversity: {ffn.get_signature_diversity():.3f}")
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print(f"Gradients flow: {ffn.up_proj.weight.grad is not None}")
    
    print("\nSparseTriXFFN: OK")
