"""
Spline2D: Piecewise Linear Kolmogorov-Arnold Network

The architecture that bridges Deep Learning and 6502 Assembly.

Instead of: y = σ(Wx)  [touches all weights]
We use:     y = Σ φᵢ(xᵢ) [lookup + simple math]

For ADC (A + B):
- 256x256 = 65,536 possible inputs
- 16x16 grid = 256 cells
- 3 coefficients per cell = 768 total parameters
- 99% compression vs brute force lookup

The Math:
    Result ≈ Base + (Slope_A × A_local) + (Slope_B × B_local)

Where:
    - Grid cell determined by high nibbles (A>>4, B>>4)
    - Local offsets are low nibbles (A & 0xF, B & 0xF)
    - Base, Slope_A, Slope_B are learned per cell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class Spline2D(nn.Module):
    """
    2D Piecewise Linear Spline for binary operations.
    
    This is a Kolmogorov-Arnold Network (KAN) specialized for 8-bit arithmetic.
    
    Args:
        grid_size: Number of grid cells per dimension (16 = 16x16 grid)
        input_range: Range of input values (256 for 8-bit)
        output_features: Number of outputs (1 for result, more for flags)
    """
    
    def __init__(
        self,
        grid_size: int = 16,
        input_range: int = 256,
        output_features: int = 1,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.input_range = input_range
        self.output_features = output_features
        self.stride = input_range // grid_size  # 16 for 256/16
        
        # Learnable coefficients: [grid_y, grid_x, output_features, 3]
        # The 3 coefficients are: (Base, Slope_A, Slope_B)
        self.coeffs = nn.Parameter(
            torch.zeros(grid_size, grid_size, output_features, 3)
        )
        
        # Initialize with sensible defaults for addition:
        # Base ≈ grid_center, Slopes ≈ 1
        self._init_for_addition()
        
    def _init_for_addition(self):
        """Initialize coefficients to approximate addition."""
        with torch.no_grad():
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Center of this grid cell
                    a_center = i * self.stride + self.stride // 2
                    b_center = j * self.stride + self.stride // 2
                    
                    # For addition: result = a + b
                    # At cell center: base = a_center + b_center
                    # Slopes should be 1.0 for both
                    base = (a_center + b_center) % 256
                    
                    for k in range(self.output_features):
                        self.coeffs[i, j, k, 0] = base  # Base
                        self.coeffs[i, j, k, 1] = 1.0   # Slope_A
                        self.coeffs[i, j, k, 2] = 1.0   # Slope_B
    
    def forward(
        self, 
        a: torch.Tensor, 
        b: torch.Tensor,
        return_indices: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the spline.
        
        Args:
            a: First input tensor [batch] or [batch, 1], values 0-255
            b: Second input tensor [batch] or [batch, 1], values 0-255
            return_indices: If True, also return grid indices (for debugging)
            
        Returns:
            result: Output tensor [batch, output_features]
        """
        # Flatten inputs
        a = a.view(-1).float()
        b = b.view(-1).float()
        batch_size = a.shape[0]
        
        # 1. Determine Grid Index (Coarse Routing)
        idx_a = (a / self.stride).long().clamp(0, self.grid_size - 1)
        idx_b = (b / self.stride).long().clamp(0, self.grid_size - 1)
        
        # 2. Determine Local Offset (Fine Detail)
        # Offset from the START of the grid cell (not center)
        off_a = (a % self.stride)
        off_b = (b % self.stride)
        
        # 3. Fetch Coefficients (Sparse Retrieval)
        # coeffs shape: [grid_size, grid_size, output_features, 3]
        # We need to gather [batch, output_features, 3]
        c = self.coeffs[idx_a, idx_b]  # [batch, output_features, 3]
        
        # 4. Compute Spline (Conditional Computation)
        # Result = Base + (Slope_A × off_a) + (Slope_B × off_b)
        base = c[:, :, 0]      # [batch, output_features]
        slope_a = c[:, :, 1]   # [batch, output_features]
        slope_b = c[:, :, 2]   # [batch, output_features]
        
        # Expand offsets for broadcasting
        off_a = off_a.unsqueeze(1)  # [batch, 1]
        off_b = off_b.unsqueeze(1)  # [batch, 1]
        
        result = base + (slope_a * off_a) + (slope_b * off_b)
        
        if return_indices:
            return result, (idx_a, idx_b, off_a.squeeze(), off_b.squeeze())
        
        return result
    
    def export_coefficients(self) -> np.ndarray:
        """
        Export coefficients for 6502 implementation.
        
        Returns:
            numpy array of shape [grid_size, grid_size, output_features, 3]
            quantized to integers
        """
        coeffs = self.coeffs.detach().cpu().numpy()
        
        # Quantize: Base to uint8, Slopes to int8
        # For clean 6502 code, we want slopes to be small integers
        coeffs_int = np.round(coeffs).astype(np.int16)
        coeffs_int[:, :, :, 0] = np.clip(coeffs_int[:, :, :, 0], 0, 255)  # Base: 0-255
        coeffs_int[:, :, :, 1] = np.clip(coeffs_int[:, :, :, 1], -8, 8)   # Slope_A: small
        coeffs_int[:, :, :, 2] = np.clip(coeffs_int[:, :, :, 2], -8, 8)   # Slope_B: small
        
        return coeffs_int
    
    def generate_6502_lookup(self) -> bytes:
        """
        Generate binary lookup table for 6502.
        
        Format: For each grid cell (row-major):
            - 1 byte: Base
            - 1 byte: Slope_A (signed)
            - 1 byte: Slope_B (signed)
        
        Total: grid_size × grid_size × 3 bytes
        """
        coeffs = self.export_coefficients()
        
        data = bytearray()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Assuming single output for now
                base = int(coeffs[i, j, 0, 0]) & 0xFF
                slope_a = int(coeffs[i, j, 0, 1]) & 0xFF
                slope_b = int(coeffs[i, j, 0, 2]) & 0xFF
                data.extend([base, slope_a, slope_b])
        
        return bytes(data)


class Spline2DWithCarry(nn.Module):
    """
    Spline2D that also handles carry input and outputs flags.
    
    For ADC: Result, C_out, V_out, Z, N
    """
    
    def __init__(self, grid_size: int = 16):
        super().__init__()
        
        # Separate splines for C_in=0 and C_in=1
        self.spline_no_carry = Spline2D(grid_size=grid_size, output_features=1)
        self.spline_with_carry = Spline2D(grid_size=grid_size, output_features=1)
        
        # Flag predictors (simple linear from result)
        self.carry_predictor = nn.Linear(3, 1)  # From A, B, base_result
        self.overflow_predictor = nn.Linear(3, 1)
        
    def forward(self, a, b, c_in):
        """
        Args:
            a: First operand [batch]
            b: Second operand [batch]
            c_in: Carry in [batch], 0 or 1
        """
        # Get results from both splines
        result_no_carry = self.spline_no_carry(a, b)
        result_with_carry = self.spline_with_carry(a, b)
        
        # Select based on carry
        c_in = c_in.view(-1, 1).float()
        result = result_no_carry * (1 - c_in) + result_with_carry * c_in
        
        # Compute flags
        # Z: result == 0
        z = (result.round() % 256 == 0).float()
        # N: result & 0x80 != 0
        n = (result.round() % 256 >= 128).float()
        
        return result, z, n


def train_spline_adc(grid_size=16, epochs=1000, lr=0.1):
    """
    Train a Spline2D to compute ADC (without carry for simplicity).
    """
    print(f"Training Spline2D for ADC (grid_size={grid_size})")
    print("=" * 60)
    
    # Create model
    model = Spline2D(grid_size=grid_size, output_features=1)
    
    # Generate ALL training data (exhaustive)
    a_vals = torch.arange(256).float()
    b_vals = torch.arange(256).float()
    
    # Create meshgrid
    a_grid, b_grid = torch.meshgrid(a_vals, b_vals, indexing='ij')
    a_flat = a_grid.flatten()
    b_flat = b_grid.flatten()
    
    # Ground truth: (a + b) % 256
    y_true = ((a_flat + b_flat) % 256).unsqueeze(1)
    
    print(f"Training on {len(a_flat)} samples (full 256x256 grid)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_pred = model(a_flat, b_flat)
        
        # Loss: MSE
        loss = F.mse_loss(y_pred, y_true)
        
        # Regularization: encourage integer slopes
        slopes = model.coeffs[:, :, :, 1:]  # [grid, grid, out, 2]
        slope_reg = ((slopes - slopes.round()) ** 2).mean()
        
        total_loss = loss + 0.1 * slope_reg
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            # Check accuracy
            with torch.no_grad():
                y_rounded = y_pred.round() % 256
                y_target = y_true % 256
                accuracy = (y_rounded == y_target).float().mean()
                
            print(f"  Epoch {epoch}: Loss={loss.item():.4f}, Acc={accuracy.item()*100:.2f}%")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(a_flat, b_flat)
        y_rounded = y_pred.round() % 256
        y_target = y_true % 256
        
        errors = (y_rounded != y_target).sum().item()
        accuracy = 1 - errors / len(a_flat)
        
    print(f"\nFinal: {errors} errors out of {len(a_flat)} ({accuracy*100:.4f}% accuracy)")
    
    # Show coefficient statistics
    coeffs = model.coeffs.detach()
    print(f"\nCoefficient statistics:")
    print(f"  Base: min={coeffs[:,:,:,0].min():.1f}, max={coeffs[:,:,:,0].max():.1f}")
    print(f"  Slope_A: min={coeffs[:,:,:,1].min():.2f}, max={coeffs[:,:,:,1].max():.2f}")
    print(f"  Slope_B: min={coeffs[:,:,:,2].min():.2f}, max={coeffs[:,:,:,2].max():.2f}")
    
    # Show a few sample cells
    print(f"\nSample grid cells:")
    for i, j in [(0, 0), (8, 8), (15, 15)]:
        c = coeffs[i, j, 0]
        print(f"  Cell ({i},{j}): Base={c[0]:.1f}, Slope_A={c[1]:.2f}, Slope_B={c[2]:.2f}")
    
    return model, accuracy


def test_spline_trace():
    """
    Trace through VGem's example: 37 + 26 = 63
    """
    print("\n" + "=" * 60)
    print("TRACE: ADC $25 + $1A (37 + 26 = 63)")
    print("=" * 60)
    
    # Create and train model
    model = Spline2D(grid_size=16, output_features=1)
    
    # Quick training
    a_vals = torch.arange(256).float()
    b_vals = torch.arange(256).float()
    a_grid, b_grid = torch.meshgrid(a_vals, b_vals, indexing='ij')
    a_flat = a_grid.flatten()
    b_flat = b_grid.flatten()
    y_true = ((a_flat + b_flat) % 256).unsqueeze(1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(500):
        optimizer.zero_grad()
        loss = F.mse_loss(model(a_flat, b_flat), y_true)
        loss.backward()
        optimizer.step()
    
    # Now trace 37 + 26
    a = torch.tensor([37.0])
    b = torch.tensor([26.0])
    
    result, (idx_a, idx_b, off_a, off_b) = model(a, b, return_indices=True)
    
    coeffs = model.coeffs.detach()
    cell_coeffs = coeffs[idx_a[0], idx_b[0], 0]
    
    print(f"\n1. Inputs: A={int(a.item())} (0x{int(a.item()):02X}), B={int(b.item())} (0x{int(b.item()):02X})")
    print(f"\n2. Grid Index: A>>4 = {idx_a[0].item()}, B>>4 = {idx_b[0].item()}")
    print(f"   → Grid Cell ({idx_a[0].item()}, {idx_b[0].item()})")
    print(f"\n3. Local Offsets: A&0xF = {off_a.item():.0f}, B&0xF = {off_b.item():.0f}")
    print(f"\n4. Spline Coefficients for Cell ({idx_a[0].item()}, {idx_b[0].item()}):")
    print(f"   Base    = {cell_coeffs[0].item():.1f}")
    print(f"   Slope_A = {cell_coeffs[1].item():.2f}")
    print(f"   Slope_B = {cell_coeffs[2].item():.2f}")
    print(f"\n5. Computation:")
    print(f"   Result = Base + (Slope_A × off_a) + (Slope_B × off_b)")
    print(f"   Result = {cell_coeffs[0].item():.1f} + ({cell_coeffs[1].item():.2f} × {off_a.item():.0f}) + ({cell_coeffs[2].item():.2f} × {off_b.item():.0f})")
    computed = cell_coeffs[0].item() + cell_coeffs[1].item() * off_a.item() + cell_coeffs[2].item() * off_b.item()
    print(f"   Result = {computed:.1f}")
    print(f"\n6. Final: {int(round(computed))} (expected: 63)")
    
    if abs(round(computed) - 63) < 1:
        print("\n✓ CORRECT!")
    else:
        print(f"\n✗ Off by {abs(round(computed) - 63)}")


if __name__ == '__main__':
    # Test the trace first
    test_spline_trace()
    
    # Then full training
    print("\n")
    model, accuracy = train_spline_adc(grid_size=16, epochs=500)
    
    if accuracy > 0.99:
        print("\n" + "=" * 60)
        print("SUCCESS! Spline2D achieves >99% accuracy on ADC")
        print("Ready for 6502 deployment.")
        print("=" * 60)
