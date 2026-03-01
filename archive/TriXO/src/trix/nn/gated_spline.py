"""
Gated Spline Unit (GSU) - VGem's Recommendation

The insight: Splines are great at shape, but sometimes you need to
shut a neuron off completely. A linear gate helps the spline focus
on the active manifold.

y = (W_up · φ(W_down · x)) ⊙ σ(W_gate · x)

This is:
- Non-Linear LoRA (rotation → spline → rotation)
- With gating (focus on active manifold)
- Multi-head option (multiple hypotheses per feature)

Architecture for a Compressible Brain:
- Learn big (Hybrid/GSU)
- Deploy small (Spline-6502)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SplineActivation(nn.Module):
    """
    Learnable 1D spline activation function.
    
    Replaces fixed activations (ReLU, GELU) with learned piecewise linear.
    """
    
    def __init__(self, features: int, grid_size: int = 8):
        super().__init__()
        self.features = features
        self.grid_size = grid_size
        
        # Per-feature spline: bases and slopes for each grid cell
        self.bases = nn.Parameter(torch.zeros(features, grid_size))
        self.slopes = nn.Parameter(torch.randn(features, grid_size) * 0.1)
        
        # Initialize to approximate ReLU
        with torch.no_grad():
            for i in range(grid_size):
                # Left half: zero (like ReLU negative region)
                # Right half: linear (like ReLU positive region)
                if i < grid_size // 2:
                    self.slopes[:, i] = 0.01  # Small leak
                else:
                    self.slopes[:, i] = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spline activation."""
        # x: [..., features]
        shape = x.shape
        x_flat = x.view(-1, self.features)
        batch_size = x_flat.shape[0]
        
        # Normalize to [0, 1) for grid indexing
        x_norm = torch.sigmoid(x_flat)
        
        # Grid indices
        idx = (x_norm * self.grid_size).long().clamp(0, self.grid_size - 1)
        
        # Gather coefficients
        bases = self.bases.unsqueeze(0).expand(batch_size, -1, -1)
        slopes = self.slopes.unsqueeze(0).expand(batch_size, -1, -1)
        
        idx_exp = idx.unsqueeze(-1)
        b = bases.gather(-1, idx_exp).squeeze(-1)
        s = slopes.gather(-1, idx_exp).squeeze(-1)
        
        # Local offset within cell
        cell_size = 1.0 / self.grid_size
        local = (x_norm - idx.float() * cell_size) / cell_size
        
        # Spline evaluation
        out = b + s * local * x_flat  # Scale by original value for gradient flow
        
        return out.view(shape)


class GatedSplineUnit(nn.Module):
    """
    Gated Spline Unit (GSU) - VGem's architecture.
    
    y = (W_up · φ(W_down · x)) ⊙ σ(W_gate · x)
    
    Components:
    - Down projection: Rotate to aligned space
    - Spline activation: Learn the nonlinearity
    - Up projection: Rotate back
    - Gate: Focus on active manifold
    """
    
    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        grid_size: int = 8,
        num_heads: int = 1,  # Multi-head splines
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model // 4
        self.grid_size = grid_size
        self.num_heads = num_heads
        
        # Total hidden = d_hidden * num_heads
        total_hidden = self.d_hidden * num_heads
        
        # Down projection (rotation to aligned space)
        self.down = nn.Linear(d_model, total_hidden, bias=False)
        
        # Spline activation (learnable nonlinearity)
        self.spline = SplineActivation(total_hidden, grid_size)
        
        # Up projection (rotation back)
        self.up = nn.Linear(total_hidden, d_model, bias=False)
        
        # Gate (focus mechanism)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        
        # Output scale
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for stable training."""
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.gate.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gating.
        
        x: [batch, seq, d_model]
        """
        # Down project (rotate to aligned space)
        h = self.down(x)
        
        # Spline activation (learn the nonlinearity)
        h = self.spline(h)
        
        # Up project (rotate back)
        h = self.up(h)
        
        # Gate (sigmoid to focus)
        g = torch.sigmoid(self.gate(x))
        
        # Gated output
        out = h * g * self.scale
        
        return out


class GSU_FFN(nn.Module):
    """
    FFN using Gated Spline Units.
    
    Drop-in replacement for transformer FFN.
    """
    
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 1,  # Can use multiple GSUs
        d_hidden: int = None,
        grid_size: int = 8,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        
        self.gsus = nn.ModuleList([
            GatedSplineUnit(d_model, d_hidden, grid_size, num_heads)
            for _ in range(num_tiles)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual."""
        h = self.norm(x)
        
        # Sum contributions from all GSUs
        out = sum(gsu(h) for gsu in self.gsus)
        
        out = self.dropout(out)
        
        return x + out
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def test_gsu():
    """Test Gated Spline Unit."""
    print("=" * 60)
    print("GATED SPLINE UNIT (GSU) TEST")
    print("=" * 60)
    
    d_model = 256
    
    # Create GSU FFN
    gsu_ffn = GSU_FFN(
        d_model=d_model,
        num_tiles=4,
        d_hidden=32,
        grid_size=8,
        num_heads=2,
    )
    
    # MLP baseline
    mlp_ffn = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model * 4, d_model),
    )
    
    gsu_params = gsu_ffn.num_parameters()
    mlp_params = sum(p.numel() for p in mlp_ffn.parameters())
    
    print(f"\nParameters:")
    print(f"  GSU FFN: {gsu_params:,}")
    print(f"  MLP FFN: {mlp_params:,}")
    print(f"  Ratio: {gsu_params / mlp_params:.1%}")
    
    # Forward pass
    x = torch.randn(4, 32, d_model)
    
    gsu_out = gsu_ffn(x)
    mlp_out = x + mlp_ffn(x)
    
    print(f"\nForward:")
    print(f"  Input: {x.shape}")
    print(f"  GSU output: {gsu_out.shape}")
    print(f"  MLP output: {mlp_out.shape}")
    
    # Gradient check
    loss = gsu_out.sum()
    loss.backward()
    
    has_grads = all(p.grad is not None for p in gsu_ffn.parameters() if p.requires_grad)
    print(f"  Gradients: {'OK' if has_grads else 'FAIL'}")
    
    # Quick learning test
    print("\nLearning test...")
    
    from torch.utils.data import DataLoader, TensorDataset
    
    X = torch.randn(500, 16, d_model)
    Y = X.roll(1, dims=-1) + 0.1 * torch.randn_like(X)
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gsu_ffn = gsu_ffn.to(device)
    mlp_ffn = mlp_ffn.to(device)
    
    def train(model, loader, epochs=15, residual_wrapper=None):
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for e in range(epochs):
            total = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                if residual_wrapper:
                    out = xb + model(xb)
                else:
                    out = model(xb)
                loss = F.mse_loss(out, yb)
                loss.backward()
                opt.step()
                total += loss.item()
            if e % 5 == 0:
                print(f"  Epoch {e}: {total/len(loader):.4f}")
        return total / len(loader)
    
    print("\nGSU:")
    gsu_loss = train(gsu_ffn, loader, residual_wrapper=None)
    
    print("\nMLP:")
    mlp_loss = train(mlp_ffn, loader, residual_wrapper=True)
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"GSU loss: {gsu_loss:.4f} ({gsu_params:,} params)")
    print(f"MLP loss: {mlp_loss:.4f} ({mlp_params:,} params)")
    print(f"GSU is {gsu_params/mlp_params:.1%} the size of MLP")
    
    if gsu_loss < mlp_loss:
        print(f"\n*** GSU WINS! ***")
    else:
        gap = (gsu_loss - mlp_loss) / gsu_loss * 100
        print(f"\nMLP wins by {gap:.1f}%")
    
    return gsu_loss, mlp_loss


if __name__ == '__main__':
    test_gsu()
