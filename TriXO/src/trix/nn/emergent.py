"""
Emergent Routing for TriX

Routing decisions emerge from tile weight structure - no learned gate network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from ..kernel import TriXLinear, STESign


class EmergentGatedFFN(nn.Module):
    """
    Feed-forward network with emergent routing.
    
    Instead of a learned gate network, routing emerges from tile weight signatures.
    Each tile's signature = sign(sum of its weight rows) - a ternary summary of
    what input patterns the tile "prefers."
    
    Routing = argmax(input @ signatures)
    
    Args:
        d_model: Model dimension
        expansion: FFN expansion factor (default: 4)
        num_tiles: Number of routing tiles (default: 4)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        num_tiles: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_model * expansion
        self.num_tiles = num_tiles
        
        # TriX layers with tiled weights
        self.up_proj = TriXLinear(d_model, self.d_ff, num_tiles)
        self.down_proj = TriXLinear(self.d_ff, d_model, num_tiles)
        self.dropout = nn.Dropout(dropout)
        
        # No gate_proj - routing emerges from up_proj weights
    
    def get_tile_signatures(self) -> torch.Tensor:
        """
        Extract tile signatures from up_proj weights.
        
        Each tile's signature is the sign of summed preferences across its outputs.
        
        Returns:
            Tensor of shape [num_tiles, d_model] with ternary values
        """
        # up_proj.weight is [d_ff, d_model]
        # Split into tiles: [num_tiles, d_ff/num_tiles, d_model]
        weight = self.up_proj.weight.data
        tile_size = self.d_ff // self.num_tiles
        
        signatures = []
        for t in range(self.num_tiles):
            start = t * tile_size
            end = start + tile_size
            tile_weight = weight[start:end, :]  # [tile_size, d_model]
            
            # Signature: what does this tile care about overall?
            preference = tile_weight.sum(dim=0)  # [d_model]
            signature = torch.sign(preference)   # Ternary
            signatures.append(signature)
        
        return torch.stack(signatures)  # [num_tiles, d_model]
    
    def route(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute routing decisions from input-signature alignment.
        
        Args:
            x: Input tensor [batch, d_model] or [batch, seq, d_model]
            
        Returns:
            One-hot routing tensor [batch, num_tiles] or [batch, seq, num_tiles]
        """
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.d_model)
        
        signatures = self.get_tile_signatures()  # [num_tiles, d_model]
        
        # Score = how well does input align with each tile's signature?
        scores = x @ signatures.T  # [batch, num_tiles]
        
        # Hard routing via argmax
        winner = scores.argmax(dim=-1, keepdim=True)  # [batch, 1]
        gate = torch.zeros_like(scores).scatter_(-1, winner, 1.0)
        
        if len(orig_shape) == 3:
            gate = gate.view(orig_shape[0], orig_shape[1], self.num_tiles)
        
        return gate
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with emergent routing.
        
        Args:
            x: Input tensor [batch, d_model] or [batch, seq, d_model]
            
        Returns:
            output: Transformed tensor (same shape as input)
            gate: Routing decisions for analysis
        """
        orig_shape = x.shape
        
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
        
        # Emergent routing - no learned gate!
        gate = self.route(x)
        
        # FFN with routing
        hidden = F.relu(self.up_proj(x, gate))
        hidden = self.dropout(hidden)
        out = self.down_proj(hidden, gate)
        
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)
            gate = gate.view(orig_shape[0], orig_shape[1], -1)
        
        return out, gate
    
    def get_routing_stats(self, x: torch.Tensor) -> dict:
        """Analyze routing behavior for a batch of inputs."""
        with torch.no_grad():
            gate = self.route(x)
            if gate.dim() == 3:
                gate = gate.view(-1, self.num_tiles)
            
            usage = gate.sum(dim=0) / gate.shape[0]
            
            return {
                'tile_usage': usage.tolist(),
                'entropy': -(usage * (usage + 1e-8).log()).sum().item(),
                'max_tile': usage.argmax().item(),
                'balance': usage.min().item() / (usage.max().item() + 1e-8),
            }


class EmergentTransformerBlock(nn.Module):
    """Transformer block with emergent routing FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_tiles: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = EmergentGatedFFN(d_model, num_tiles=num_tiles, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        normed = self.norm1(x)
        if is_causal:
            T = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask, is_causal=True)
        else:
            attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # Emergent FFN
        ffn_out, gate = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x, gate
