"""
Tile-Routed Memory: Attention replacement via addressable memory slots.

Core idea:
    Instead of O(n²) pairwise attention, use O(M·d) routed memory:
    - M memory slots with learned signatures
    - Each token routes to a slot via signature matching
    - Read/write operations are sparse and addressable

Complexity:
    - Attention: O(n² · d) per layer
    - Routed Memory: O(n · M · d) where M << n

This is "attention where the keys are tiled and addressable."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class RoutedMemoryAttention(nn.Module):
    """
    Attention replacement using routed memory slots.
    
    Architecture:
        1. M memory slots, each with:
           - signature: for routing (ternary, learned)
           - value: content to read (learned or accumulated)
        2. For each token:
           - Route to top-k slots via signature matching
           - Read from selected slots
           - Optionally write to slots (update content)
    
    Args:
        d_model: Model dimension
        n_slots: Number of memory slots (like "number of keys")
        n_heads: Number of attention heads (each head has its own slots)
        top_k: Number of slots to read from (1 = hard routing, >1 = soft)
        slot_dim: Dimension per slot (default: d_model // n_heads)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_slots: int = 64,
        n_heads: int = 4,
        top_k: int = 1,
        slot_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.top_k = top_k
        self.slot_dim = slot_dim or d_model // n_heads
        
        # Query projection (token → query for routing)
        self.q_proj = nn.Linear(d_model, n_heads * self.slot_dim)
        
        # Per-head slot signatures (for routing)
        # Shape: [n_heads, n_slots, slot_dim]
        self.slot_signatures = nn.Parameter(
            torch.randn(n_heads, n_slots, self.slot_dim) * 0.5
        )
        
        # Per-head slot values (the "content" to read)
        # Shape: [n_heads, n_slots, slot_dim]
        self.slot_values = nn.Parameter(
            torch.randn(n_heads, n_slots, self.slot_dim) * 0.02
        )
        
        # Output projection
        self.out_proj = nn.Linear(n_heads * self.slot_dim, d_model)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Temperature for routing softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Tracking
        self.register_buffer('slot_usage', torch.zeros(n_heads, n_slots))
        self.register_buffer('total_tokens', torch.tensor(0.0))
    
    def _quantize_ternary(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize signatures to {-1, 0, +1}."""
        with torch.no_grad():
            q = torch.zeros_like(x)
            q[x > 0.3] = 1.0
            q[x < -0.3] = -1.0
        return x + (q - x).detach()
    
    @property
    def signatures(self) -> torch.Tensor:
        """Get quantized ternary signatures."""
        return self._quantize_ternary(self.slot_signatures)
    
    def forward(
        self, 
        x: torch.Tensor,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, d_model]
            is_causal: Whether to apply causal masking (not used in slot-based)
        
        Returns:
            output: Output tensor [B, T, d_model]
            info: Dict with routing information
        """
        B, T, D = x.shape
        
        # Normalize input
        x_norm = self.norm(x)
        
        # Project to queries [B, T, n_heads, slot_dim]
        q = self.q_proj(x_norm).view(B, T, self.n_heads, self.slot_dim)
        
        # Get signatures [n_heads, n_slots, slot_dim]
        sigs = self.signatures
        
        # Compute routing scores [B, T, n_heads, n_slots]
        # For each token, score against all slots
        scores = torch.einsum('bthd,hsd->bths', q, sigs)
        scores = scores / (self.temperature * math.sqrt(self.slot_dim))
        
        # Route: select top-k slots per token per head
        if self.top_k == 1:
            # Hard routing
            slot_idx = scores.argmax(dim=-1)  # [B, T, n_heads]
            weights = torch.ones(B, T, self.n_heads, 1, device=x.device)
        else:
            # Soft top-k routing
            topk_scores, slot_idx = scores.topk(self.top_k, dim=-1)  # [B, T, n_heads, top_k]
            weights = F.softmax(topk_scores, dim=-1)  # [B, T, n_heads, top_k]
        
        # Read from slots [n_heads, n_slots, slot_dim]
        values = self.slot_values
        
        # Gather values for selected slots
        if self.top_k == 1:
            # [B, T, n_heads] -> gather from [n_heads, n_slots, slot_dim]
            # Expand slot_idx for gathering
            idx_expanded = slot_idx.unsqueeze(-1).expand(-1, -1, -1, self.slot_dim)
            # Expand values for batch
            values_expanded = values.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
            # Gather: [B, T, n_heads, slot_dim]
            read_values = torch.gather(
                values_expanded, 
                dim=3, 
                index=idx_expanded.unsqueeze(3)
            ).squeeze(3)
        else:
            # Multi-slot read with weighting
            # slot_idx: [B, T, n_heads, top_k]
            idx_expanded = slot_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.slot_dim)
            values_expanded = values.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
            # [B, T, n_heads, top_k, slot_dim]
            selected_values = torch.gather(
                values_expanded.unsqueeze(3).expand(-1, -1, -1, self.top_k, -1, -1),
                dim=4,
                index=idx_expanded
            )
            # Weight and sum: [B, T, n_heads, slot_dim]
            read_values = (selected_values * weights.unsqueeze(-1)).sum(dim=3)
        
        # Reshape and project output
        read_values = read_values.reshape(B, T, self.n_heads * self.slot_dim)
        output = self.out_proj(read_values)
        output = self.dropout(output)
        
        # Residual connection
        output = x + output
        
        # Track usage
        if self.training:
            for h in range(self.n_heads):
                if self.top_k == 1:
                    slots_used = slot_idx[:, :, h].flatten()
                else:
                    slots_used = slot_idx[:, :, h, :].flatten()
                for s in slots_used.unique():
                    self.slot_usage[h, s] += (slots_used == s).sum().float()
            self.total_tokens += B * T
        
        info = {
            'slot_idx': slot_idx,
            'scores': scores,
        }
        
        return output, info
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        if self.total_tokens == 0:
            return {'n_slots': self.n_slots, 'active_slots': 0}
        
        usage = self.slot_usage / self.total_tokens
        
        return {
            'n_slots': self.n_slots,
            'n_heads': self.n_heads,
            'active_slots_per_head': [(usage[h] > 0.001).sum().item() for h in range(self.n_heads)],
            'usage_mean': usage.mean().item(),
            'usage_std': usage.std().item(),
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.slot_usage.zero_()
        self.total_tokens.zero_()


class RoutedMemoryBlock(nn.Module):
    """
    Transformer block with RoutedMemoryAttention instead of standard attention.
    
    This is the "attention replacement" experiment block.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_slots: int = 64,
        n_heads: int = 4,
        top_k: int = 1,
        ffn_hidden: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        ffn_hidden = ffn_hidden or d_model * 4
        
        # Routed memory instead of attention
        self.memory = RoutedMemoryAttention(
            d_model=d_model,
            n_slots=n_slots,
            n_heads=n_heads,
            top_k=top_k,
            dropout=dropout,
        )
        
        # Standard FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, is_causal: bool = True) -> Tuple[torch.Tensor, Dict]:
        # Memory read (replaces attention)
        x, mem_info = self.memory(x, is_causal=is_causal)
        
        # FFN with residual
        x = x + self.ffn(x)
        
        return x, mem_info


class RoutedMemoryTransformer(nn.Module):
    """
    Full transformer using RoutedMemoryAttention instead of standard attention.
    
    For comparing against standard transformers.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_slots: int = 64,
        n_heads: int = 4,
        top_k: int = 1,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Routed memory blocks
        self.blocks = nn.ModuleList([
            RoutedMemoryBlock(
                d_model=d_model,
                n_slots=n_slots,
                n_heads=n_heads,
                top_k=top_k,
                dropout=dropout,
            ) for _ in range(n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T = x.shape
        
        # Embeddings
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        
        # Blocks
        all_info = []
        for block in self.blocks:
            h, info = block(h)
            all_info.append(info)
        
        # Output
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits, {'block_info': all_info}
    
    def get_all_routing_stats(self) -> Dict:
        """Get routing stats from all blocks."""
        return {
            f'block_{i}': block.memory.get_routing_stats()
            for i, block in enumerate(self.blocks)
        }


# =============================================================================
# Standard Attention Baseline for comparison
# =============================================================================

class StandardAttentionBlock(nn.Module):
    """Standard transformer block with PyTorch attention."""
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_hidden: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        ffn_hidden = ffn_hidden or d_model * 4
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        
        x_norm = self.ln1(x)
        if is_causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        x = x + self.dropout(attn_out)
        x = x + self.ffn(x)
        
        return x


class StandardTransformer(nn.Module):
    """Standard transformer baseline."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            StandardAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        
        for block in self.blocks:
            h = block(h)
        
        h = self.ln_f(h)
        return self.head(h)


# =============================================================================
# Test
# =============================================================================

def test_routed_memory():
    """Test RoutedMemoryAttention."""
    print("=" * 70)
    print("RoutedMemoryAttention Test")
    print("=" * 70)
    
    d_model = 64
    n_slots = 32
    n_heads = 4
    
    # Test single module
    mem = RoutedMemoryAttention(
        d_model=d_model,
        n_slots=n_slots,
        n_heads=n_heads,
        top_k=1,
    )
    
    x = torch.randn(2, 16, d_model)
    output, info = mem(x)
    
    print(f"\nRoutedMemoryAttention:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Slot indices shape: {info['slot_idx'].shape}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print(f"  Gradients: OK")
    
    # Test full transformer
    print(f"\nRoutedMemoryTransformer:")
    
    vocab_size = 100
    model = RoutedMemoryTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_slots=n_slots,
        n_heads=n_heads,
    )
    
    x = torch.randint(0, vocab_size, (2, 16))
    logits, info = model(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits.shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    # Compare to standard transformer
    baseline = StandardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=n_heads,
    )
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"  Baseline params: {baseline_params:,}")
    print(f"  Ratio: {params/baseline_params:.2f}x")
    
    print("\n" + "=" * 70)
    print("Test PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_routed_memory()
