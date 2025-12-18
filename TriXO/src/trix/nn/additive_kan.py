"""
Track B: Additive KAN for High-Dimensional Inputs

VGem's critical insight:
- 768-dim grid = 16^768 cells = Heat Death
- 768 × 1D splines = 768 × 16 cells = 12,288 cells = Doable

Kolmogorov-Arnold representation:
    f(x1, x2, ..., xn) = Σ φᵢ(xᵢ)

Each φᵢ is a 1D spline (piecewise linear function).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Spline1D(nn.Module):
    """
    Learnable 1D piecewise linear function.
    
    Divides input range into grid_size intervals.
    Each interval has a linear function: y = base + slope * (x - x_start)
    """
    
    def __init__(self, grid_size: int = 16, input_range: tuple = (-1, 1)):
        super().__init__()
        
        self.grid_size = grid_size
        self.input_min, self.input_max = input_range
        self.input_range = self.input_max - self.input_min
        
        # Learnable parameters: base and slope for each interval
        self.bases = nn.Parameter(torch.zeros(grid_size))
        self.slopes = nn.Parameter(torch.ones(grid_size) * 0.1)
        
    def forward(self, x):
        """
        Evaluate spline at points x.
        
        Args:
            x: Input tensor [..., 1] or [...]
        
        Returns:
            y: Output tensor, same shape as x
        """
        # Normalize to [0, 1]
        x_norm = (x - self.input_min) / self.input_range
        x_norm = x_norm.clamp(0, 1 - 1e-6)
        
        # Get interval index
        idx = (x_norm * self.grid_size).long()
        idx = idx.clamp(0, self.grid_size - 1)
        
        # Get local offset within interval
        interval_size = 1.0 / self.grid_size
        x_local = (x_norm - idx.float() * interval_size) / interval_size
        
        # Lookup coefficients
        base = self.bases[idx]
        slope = self.slopes[idx]
        
        # Evaluate: y = base + slope * x_local
        y = base + slope * x_local
        
        return y


class AdditiveKAN(nn.Module):
    """
    Additive Kolmogorov-Arnold Network.
    
    For input dimension d, uses d separate 1D splines and sums outputs.
    
    f(x) = Σᵢ φᵢ(xᵢ) + bias
    
    Complexity: O(d × grid_size) instead of O(grid_size^d)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        grid_size: int = 16,
        input_range: tuple = (-1, 1),
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        
        # One 1D spline per input dimension, per output dimension
        self.splines = nn.ModuleList([
            nn.ModuleList([
                Spline1D(grid_size, input_range)
                for _ in range(input_dim)
            ])
            for _ in range(output_dim)
        ])
        
        # Output bias
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, input_dim]
        
        Returns:
            y: Output tensor [batch, output_dim]
        """
        batch_size = x.shape[0]
        outputs = []
        
        for out_idx in range(self.output_dim):
            # Sum contributions from each input dimension
            total = torch.zeros(batch_size, device=x.device)
            for in_idx in range(self.input_dim):
                total = total + self.splines[out_idx][in_idx](x[:, in_idx])
            
            outputs.append(total + self.bias[out_idx])
        
        return torch.stack(outputs, dim=1)
    
    def num_parameters(self):
        """Count parameters."""
        # Each spline: grid_size bases + grid_size slopes
        spline_params = 2 * self.grid_size
        total_splines = self.input_dim * self.output_dim
        bias_params = self.output_dim
        
        return spline_params * total_splines + bias_params
    
    def memory_bytes(self, dtype_bytes=4):
        """Estimate memory usage."""
        return self.num_parameters() * dtype_bytes


class ProductSplineKAN(nn.Module):
    """
    Product-Space KAN: Groups dimensions into pairs, uses 2D splines.
    
    VGem's alternative suggestion:
    - Group 768 dims into 384 pairs
    - Use 2D spline for each pair
    - Sum results
    
    Slightly more expressive than pure additive, still tractable.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        grid_size: int = 16,
        input_range: tuple = (-1, 1),
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.num_pairs = input_dim // 2
        self.has_odd = input_dim % 2 == 1
        
        # 2D splines for pairs
        # Each has grid_size × grid_size × 3 parameters (base, slope_a, slope_b)
        self.pair_coeffs = nn.ParameterList([
            nn.Parameter(torch.zeros(output_dim, grid_size, grid_size, 3))
            for _ in range(self.num_pairs)
        ])
        
        # Initialize slopes to small values
        for coeffs in self.pair_coeffs:
            nn.init.zeros_(coeffs[..., 0])  # base
            nn.init.normal_(coeffs[..., 1], mean=0, std=0.1)  # slope_a
            nn.init.normal_(coeffs[..., 2], mean=0, std=0.1)  # slope_b
        
        # 1D spline for odd dimension if needed
        if self.has_odd:
            self.odd_spline = nn.ModuleList([
                Spline1D(grid_size, input_range)
                for _ in range(output_dim)
            ])
        
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.input_range = input_range
        
    def forward(self, x):
        """Forward pass with 2D spline pairs."""
        batch_size = x.shape[0]
        device = x.device
        
        # Normalize input
        x_norm = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        x_norm = x_norm.clamp(0, 1 - 1e-6)
        
        outputs = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Process pairs
        for pair_idx in range(self.num_pairs):
            dim_a = pair_idx * 2
            dim_b = pair_idx * 2 + 1
            
            a = x_norm[:, dim_a]
            b = x_norm[:, dim_b]
            
            # Grid indices
            idx_a = (a * self.grid_size).long().clamp(0, self.grid_size - 1)
            idx_b = (b * self.grid_size).long().clamp(0, self.grid_size - 1)
            
            # Lookup coefficients for each output dim
            coeffs = self.pair_coeffs[pair_idx]  # [output_dim, grid, grid, 3]
            
            for out_idx in range(self.output_dim):
                c = coeffs[out_idx, idx_a, idx_b]  # [batch, 3]
                
                # Evaluate: base + slope_a * a + slope_b * b
                result = c[:, 0] + c[:, 1] * a + c[:, 2] * b
                outputs[:, out_idx] = outputs[:, out_idx] + result
        
        # Handle odd dimension
        if self.has_odd:
            odd_dim = self.input_dim - 1
            for out_idx in range(self.output_dim):
                outputs[:, out_idx] = outputs[:, out_idx] + \
                    self.odd_spline[out_idx](x[:, odd_dim])
        
        return outputs + self.bias
    
    def num_parameters(self):
        """Count parameters."""
        # Each 2D spline: grid × grid × 3 × output_dim
        pair_params = self.grid_size * self.grid_size * 3 * self.output_dim
        total_pair_params = pair_params * self.num_pairs
        
        # 1D spline for odd: 2 × grid × output_dim
        odd_params = 0
        if self.has_odd:
            odd_params = 2 * self.grid_size * self.output_dim
        
        return total_pair_params + odd_params + self.output_dim


class KANTile(nn.Module):
    """
    A TriX Tile implemented as an Additive KAN.
    
    Replaces the dense MLP tile with sparse KAN computation.
    This is the integration point for PQH architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int = 16,
        use_product: bool = False,
    ):
        super().__init__()
        
        if use_product:
            self.kan = ProductSplineKAN(input_dim, output_dim, grid_size)
        else:
            self.kan = AdditiveKAN(input_dim, output_dim, grid_size)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        Forward with residual connection (VGem's Residual Bus).
        
        Output = Input + KAN(Input)
        """
        # If dimensions match, use residual
        if self.input_dim == self.output_dim:
            return x + self.kan(x)
        else:
            return self.kan(x)


class SharedAdditiveKAN(nn.Module):
    """
    Efficient Additive KAN with shared splines across output dimensions.
    
    Instead of separate splines for each (input, output) pair,
    share splines across outputs with learned mixing weights.
    
    f(x)_j = Σᵢ wᵢⱼ × φᵢ(xᵢ)
    
    where φᵢ is shared across all outputs.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int = 16,
        input_range: tuple = (-1, 1),
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        
        # Shared 1D splines - one per input dimension
        self.splines = nn.ModuleList([
            Spline1D(grid_size, input_range)
            for _ in range(input_dim)
        ])
        
        # Mixing weights: how each spline contributes to each output
        self.mix = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [batch, input_dim]
        Returns:
            y: [batch, output_dim]
        """
        batch_size = x.shape[0]
        
        # Evaluate all splines: [batch, input_dim]
        spline_outputs = torch.stack([
            self.splines[i](x[:, i])
            for i in range(self.input_dim)
        ], dim=1)
        
        # Mix to get outputs: [batch, output_dim]
        y = torch.matmul(spline_outputs, self.mix.t()) + self.bias
        
        return y
    
    def num_parameters(self):
        """Count parameters."""
        spline_params = 2 * self.grid_size * self.input_dim
        mix_params = self.output_dim * self.input_dim
        bias_params = self.output_dim
        return spline_params + mix_params + bias_params


def test_additive_kan():
    """Test AdditiveKAN."""
    print("=" * 60)
    print("ADDITIVE KAN TEST")
    print("=" * 60)
    
    # Test 1: Simple function approximation
    print("\n1. Approximating f(x,y) = x + y")
    
    kan = AdditiveKAN(input_dim=2, output_dim=1, grid_size=16)
    
    # Generate training data
    x = torch.rand(1000, 2) * 2 - 1  # [-1, 1]
    y_true = x[:, 0:1] + x[:, 1:2]
    
    # Train
    optimizer = torch.optim.Adam(kan.parameters(), lr=0.01)
    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = kan(x)
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # Test
    x_test = torch.rand(100, 2) * 2 - 1
    y_test = x_test[:, 0:1] + x_test[:, 1:2]
    y_pred = kan(x_test)
    mse = F.mse_loss(y_pred, y_test).item()
    print(f"  Final test MSE: {mse:.6f}")
    
    # Test 2: Efficient SharedAdditiveKAN
    print("\n2. SharedAdditiveKAN for 768 dimensions")
    
    # This is the RIGHT architecture for transformer-scale
    shared_kan = SharedAdditiveKAN(input_dim=768, output_dim=768, grid_size=16)
    
    print(f"  Input dim: 768")
    print(f"  Output dim: 768")
    print(f"  Grid size: 16")
    print(f"  Parameters: {shared_kan.num_parameters():,}")
    
    # Compare to dense layer and MLP
    dense_params = 768 * 768 + 768  # 590,592
    mlp_params = 768 * 3072 * 2 + 3072 + 768  # 4,722,432
    
    print(f"\n  Comparison:")
    print(f"    Dense (768→768):     {dense_params:,} params")
    print(f"    MLP (768→3072→768):  {mlp_params:,} params")
    print(f"    SharedKAN:           {shared_kan.num_parameters():,} params")
    print(f"\n  SharedKAN / Dense: {shared_kan.num_parameters() / dense_params:.1%}")
    print(f"  SharedKAN / MLP:   {shared_kan.num_parameters() / mlp_params:.1%}")
    
    # Test 3: KAN as TriX Tile replacement
    print("\n3. KAN Tile for TriX (single tile)")
    
    # A single TriX tile might handle 64-dim slices
    tile_kan = SharedAdditiveKAN(input_dim=64, output_dim=64, grid_size=16)
    tile_mlp_params = 64 * 256 * 2 + 256 + 64  # Small MLP
    
    print(f"  Tile input/output: 64")
    print(f"  KAN Tile params: {tile_kan.num_parameters():,}")
    print(f"  MLP Tile params: {tile_mlp_params:,}")
    print(f"  Savings: {(1 - tile_kan.num_parameters() / tile_mlp_params) * 100:.1f}%")
    
    # Memory with 2-bit quantization
    kan_2bit = tile_kan.num_parameters() * 0.25  # 2-bit = 0.25 bytes
    print(f"\n  KAN Tile (2-bit): {kan_2bit:,.0f} bytes")
    
    return True


if __name__ == '__main__':
    test_additive_kan()
