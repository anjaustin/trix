"""
TriX Native Layers

Production-ready neural network components with signature-based emergent routing.
This is the recommended way to use TriX.

Key insight: Routing decisions emerge from weight structure - no learned gate network needed.
Each tile's signature (sum of its weight preferences) determines which inputs it handles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..kernel import TriXLinear


class TriXFFN(nn.Module):
    """
    Feed-forward network with signature-based emergent routing.
    
    Instead of a learned gate network, routing emerges from tile weight signatures.
    Each tile's signature = sign(sum of its weight rows) - a ternary summary of
    what input patterns the tile "prefers."
    
    Routing = argmax(input @ signatures.T)
    
    This achieves:
    - Zero routing parameters
    - Consistent routing for similar inputs
    - Discriminative routing for different inputs
    - Automatic adaptation as weights train
    
    Args:
        d_model: Model dimension
        expansion: FFN expansion factor (default: 4)
        num_tiles: Number of routing tiles (default: 4)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> ffn = TriXFFN(512, num_tiles=4)
        >>> output, routing = ffn(input)
        >>> ffn.get_routing_stats(input)  # Analyze routing behavior
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
        
        # TriX layers - routing emerges from up_proj signatures
        self.up_proj = TriXLinear(d_model, self.d_ff, num_tiles)
        self.down_proj = TriXLinear(self.d_ff, d_model, num_tiles)
        self.dropout = nn.Dropout(dropout)
    
    def get_tile_signatures(self) -> torch.Tensor:
        """
        Get tile signatures from up_proj weights.
        
        Returns:
            Tensor [num_tiles, d_model] with ternary values {-1, 0, +1}
        """
        return self.up_proj.get_tile_signatures()
    
    def compute_routing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute routing decisions via input-signature alignment.
        
        Args:
            x: Input tensor [batch, d_model] or [batch, seq, d_model]
            
        Returns:
            One-hot routing tensor [batch, num_tiles] or [batch, seq, num_tiles]
        """
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.d_model)
        
        signatures = self.get_tile_signatures()  # [num_tiles, d_model]
        scores = x @ signatures.T  # [batch, num_tiles]
        
        # Hard routing via argmax - winner takes all
        winner = scores.argmax(dim=-1, keepdim=True)
        gate = torch.zeros_like(scores).scatter_(-1, winner, 1.0)
        
        if len(orig_shape) == 3:
            gate = gate.view(orig_shape[0], orig_shape[1], self.num_tiles)
        
        return gate
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with emergent routing.
        
        Args:
            x: Input [batch, d_model] or [batch, seq, d_model]
            
        Returns:
            output: Transformed tensor (same shape as input)
            routing: Routing decisions [batch, num_tiles] or [batch, seq, num_tiles]
        """
        orig_shape = x.shape
        
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
        
        # Emergent routing from signatures
        gate = self.compute_routing(x)
        
        # FFN computation
        hidden = F.relu(self.up_proj(x, gate))
        hidden = self.dropout(hidden)
        out = self.down_proj(hidden, gate)
        
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)
            gate = gate.view(orig_shape[0], orig_shape[1], -1)
        
        return out, gate
    
    def get_routing_stats(self, x: torch.Tensor) -> dict:
        """
        Analyze routing behavior for a batch of inputs.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with:
            - tile_usage: List of usage fractions per tile
            - balance: Ratio of min to max usage (1.0 = perfect balance)
            - entropy: Routing entropy (higher = more balanced)
            - dominant_tile: Most frequently used tile index
        """
        with torch.no_grad():
            gate = self.compute_routing(x)
            if gate.dim() == 3:
                gate = gate.view(-1, self.num_tiles)
            
            usage = gate.mean(dim=0)
            
            return {
                'tile_usage': usage.tolist(),
                'balance': (usage.min() / (usage.max() + 1e-8)).item(),
                'entropy': -(usage * (usage + 1e-8).log()).sum().item(),
                'dominant_tile': usage.argmax().item(),
            }
    
    def get_signature_diversity(self) -> float:
        """
        Measure pairwise diversity between tile signatures.
        
        Returns:
            Float in [0, 1] - fraction of differing signature positions
            Higher = more diverse tiles, Lower = potential collapse
        """
        sigs = self.get_tile_signatures()
        total_diff = 0.0
        pairs = 0
        
        for i in range(self.num_tiles):
            for j in range(i + 1, self.num_tiles):
                diff = (sigs[i] != sigs[j]).float().mean().item()
                total_diff += diff
                pairs += 1
        
        return total_diff / pairs if pairs > 0 else 0.0
    
    def pack(self):
        """Prepare for fast inference - pack weights and cache signatures."""
        self.up_proj.pack()
        self.down_proj.pack()
    
    def unpack(self):
        """Return to training mode - unpack weights and clear caches."""
        self.up_proj.unpack()
        self.down_proj.unpack()


class TriXBlock(nn.Module):
    """
    Transformer block with TriX sparse FFN and emergent routing.
    
    Standard transformer architecture:
    - Multi-head self-attention
    - TriXFFN with signature-based routing
    - Pre-norm residual connections
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        num_tiles: Number of routing tiles (default: 4)
        expansion: FFN expansion factor (default: 4)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> block = TriXBlock(512, n_heads=8, num_tiles=4)
        >>> output, routing = block(input)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_tiles: int = 4,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = TriXFFN(d_model, expansion=expansion, num_tiles=num_tiles, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Optional attention mask
            is_causal: Whether to use causal masking
            
        Returns:
            output: Transformed tensor [batch, seq_len, d_model]
            routing: FFN routing decisions [batch, seq_len, num_tiles]
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)
        
        if is_causal:
            T = x.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        attn_out, _ = self.attention(
            normed, normed, normed,
            attn_mask=attn_mask,
            is_causal=is_causal
        )
        x = x + self.dropout(attn_out)
        
        # FFN with emergent routing
        ffn_out, routing = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x, routing
    
    def get_routing_stats(self, x: torch.Tensor) -> dict:
        """Get FFN routing statistics."""
        return self.ffn.get_routing_stats(x)
    
    def get_signature_diversity(self) -> float:
        """Get FFN signature diversity."""
        return self.ffn.get_signature_diversity()
    
    def pack(self):
        """Prepare for fast inference."""
        self.ffn.pack()
    
    def unpack(self):
        """Return to training mode."""
        self.ffn.unpack()


class TriXStack(nn.Module):
    """
    Stack of TriX transformer blocks.
    
    Convenience class for building transformer models with emergent routing.
    
    Args:
        n_layers: Number of transformer blocks
        d_model: Model dimension
        n_heads: Number of attention heads
        num_tiles: Number of routing tiles (default: 4)
        expansion: FFN expansion factor (default: 4)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        num_tiles: int = 4,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            TriXBlock(d_model, n_heads, num_tiles, expansion, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through all blocks.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            is_causal: Whether to use causal masking
            
        Returns:
            output: Transformed tensor
            all_routing: List of routing tensors from each block
        """
        all_routing = []
        
        for block in self.blocks:
            x, routing = block(x, is_causal=is_causal)
            all_routing.append(routing)
        
        x = self.norm(x)
        return x, all_routing
    
    def get_all_routing_stats(self, x: torch.Tensor) -> list:
        """Get routing statistics for all blocks."""
        stats = []
        for i, block in enumerate(self.blocks):
            block_stats = block.get_routing_stats(x)
            block_stats['block'] = i
            stats.append(block_stats)
        return stats
    
    def pack(self):
        """Prepare all blocks for fast inference."""
        for block in self.blocks:
            block.pack()
    
    def unpack(self):
        """Return all blocks to training mode."""
        for block in self.blocks:
            block.unpack()
