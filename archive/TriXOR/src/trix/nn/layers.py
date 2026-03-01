"""
TriX Neural Network Layers

High-level components for building sparse neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..kernel import TriXLinear


class Top1Gate(torch.autograd.Function):
    """
    Hard top-1 gating with straight-through gradient.
    
    Forward: produces one-hot vector from argmax
    Backward: passes gradient through unchanged
    """
    
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        idx = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, idx, 1.0)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class GatedFFN(nn.Module):
    """
    Feed-forward network with TriX sparse routing.
    
    Uses tile-based gating to skip computation for inactive tiles,
    achieving linear speedup proportional to sparsity.
    
    Args:
        d_model: Model dimension
        expansion: FFN expansion factor (default: 4)
        num_tiles: Number of routing tiles (default: 4)
        noise_scale: Gate noise for load balancing during training
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        num_tiles: int = 4,
        noise_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_model * expansion
        self.num_tiles = num_tiles
        self.noise_scale = noise_scale
        
        self.up_proj = TriXLinear(d_model, self.d_ff, num_tiles)
        self.down_proj = TriXLinear(self.d_ff, d_model, num_tiles)
        self.gate_proj = nn.Linear(d_model, num_tiles)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, d_model] or [batch, d_model]
            
        Returns:
            output: Transformed tensor (same shape as input)
            gate: Gate activations for analysis
        """
        orig_shape = x.shape
        
        # Flatten to 2D for processing
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
        
        # Compute gate
        gate_logits = self.gate_proj(x)
        
        # Add noise during training for load balancing
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(gate_logits) * self.noise_scale
            gate_logits = gate_logits + noise
        
        gate = Top1Gate.apply(gate_logits)
        
        # FFN with sparse routing
        hidden = F.relu(self.up_proj(x, gate))
        hidden = self.dropout(hidden)
        out = self.down_proj(hidden, gate)
        
        # Restore original shape
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)
            gate = gate.view(orig_shape[0], orig_shape[1], -1)
        
        return out, gate
    
    def get_gate_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Get gate usage distribution for analysis."""
        with torch.no_grad():
            _, gate = self.forward(x)
            return gate.mean(dim=0)


class TriXTransformerBlock(nn.Module):
    """
    Transformer block with TriX sparse FFN.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        num_tiles: Number of routing tiles
        noise_scale: Gate noise for load balancing
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_tiles: int = 4,
        noise_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = GatedFFN(d_model, num_tiles=num_tiles, noise_scale=noise_scale, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Optional attention mask
            is_causal: Whether to use causal masking
            
        Returns:
            output: Transformed tensor
            gate: Gate activations
        """
        # Self-attention
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
        
        # Feed-forward
        ffn_out, gate = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x, gate
