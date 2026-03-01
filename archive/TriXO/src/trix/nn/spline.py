"""
Lookup-Spline Neural Networks

Piecewise linear functions that enable neural inference on 6502-class hardware.

The key insight (from NVIDIA):
- Splines are sparse: only the relevant interval's coefficients are used
- Splines show where computation is important
- On 6502: ~12 cycles vs ~10,000 for MLP

This is TriX made explicit:
- Intervals = Tiles
- Coefficients = Weights  
- Interval lookup = Routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SplineLayer(nn.Module):
    """
    Piecewise linear spline layer.
    
    For each input feature:
    1. Determine which interval the value falls into
    2. Load slope and intercept for that interval
    3. Compute: y = slope * x + intercept
    
    This is mathematically equivalent to an MLP but:
    - Only touches relevant coefficients (sparse)
    - Can be implemented with lookup tables (6502-friendly)
    - Training learns the optimal piecewise approximation
    
    Args:
        in_features: Number of input features
        out_features: Number of output features  
        num_intervals: Number of piecewise intervals (more = higher accuracy)
        input_range: (min, max) range of expected inputs
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_intervals: int = 16,
        input_range: Tuple[float, float] = (0.0, 255.0),
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_intervals = num_intervals
        self.input_min, self.input_max = input_range
        
        # Learnable coefficients: [out_features, in_features, num_intervals]
        # For each output, for each input feature, for each interval
        self.slopes = nn.Parameter(
            torch.randn(out_features, in_features, num_intervals) * 0.1
        )
        self.intercepts = nn.Parameter(
            torch.zeros(out_features, in_features, num_intervals)
        )
        
        # Output bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Interval boundaries (fixed, uniform)
        # Register as buffer (not a parameter)
        boundaries = torch.linspace(
            self.input_min, 
            self.input_max, 
            num_intervals + 1
        )
        self.register_buffer('boundaries', boundaries)
        
    def _get_interval_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determine which interval each input value falls into.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            indices: Interval indices [..., in_features] in range [0, num_intervals-1]
        """
        # Normalize to [0, num_intervals]
        x_norm = (x - self.input_min) / (self.input_max - self.input_min)
        x_norm = x_norm * self.num_intervals
        
        # Clamp to valid range and convert to indices
        indices = x_norm.floor().long()
        indices = indices.clamp(0, self.num_intervals - 1)
        
        return indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spline layer.
        
        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]
            
        Returns:
            y: Output tensor [batch, out_features] or [batch, seq, out_features]
        """
        original_shape = x.shape
        
        # Flatten to [batch, in_features]
        if x.dim() == 3:
            batch, seq, features = x.shape
            x = x.reshape(batch * seq, features)
        
        batch_size = x.shape[0]
        
        # Get interval indices: [batch, in_features]
        indices = self._get_interval_indices(x)
        
        # Gather slopes and intercepts for each input
        # slopes: [out_features, in_features, num_intervals]
        # indices: [batch, in_features]
        # We need: [batch, out_features, in_features]
        
        # Expand indices for gathering
        # [batch, 1, in_features, 1] -> broadcast over out_features
        idx_expanded = indices.unsqueeze(1).unsqueeze(-1)  # [batch, 1, in_features, 1]
        idx_expanded = idx_expanded.expand(batch_size, self.out_features, self.in_features, 1)
        
        # Expand slopes/intercepts for batch
        slopes_expanded = self.slopes.unsqueeze(0).expand(batch_size, -1, -1, -1)
        intercepts_expanded = self.intercepts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Gather: [batch, out_features, in_features]
        gathered_slopes = slopes_expanded.gather(-1, idx_expanded).squeeze(-1)
        gathered_intercepts = intercepts_expanded.gather(-1, idx_expanded).squeeze(-1)
        
        # Evaluate spline: y = slope * x + intercept
        # x: [batch, in_features] -> [batch, 1, in_features]
        x_expanded = x.unsqueeze(1)
        
        # Per-feature contribution: [batch, out_features, in_features]
        contributions = gathered_slopes * x_expanded + gathered_intercepts
        
        # Sum over input features: [batch, out_features]
        output = contributions.sum(dim=-1) + self.bias
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            output = output.reshape(batch, seq, self.out_features)
            
        return output
    
    def export_tables(self) -> dict:
        """
        Export coefficients as lookup tables for 6502 implementation.
        
        Returns:
            dict with 'slopes' and 'intercepts' as numpy arrays,
            quantized to 8-bit fixed-point format.
        """
        import numpy as np
        
        # Convert to numpy
        slopes_np = self.slopes.detach().cpu().numpy()
        intercepts_np = self.intercepts.detach().cpu().numpy()
        
        # Quantize to 8-bit signed (-128 to 127)
        # Scale to fit range
        slope_max = max(abs(slopes_np.min()), abs(slopes_np.max()))
        intercept_max = max(abs(intercepts_np.min()), abs(intercepts_np.max()))
        
        if slope_max > 0:
            slopes_q = (slopes_np / slope_max * 127).astype(np.int8)
        else:
            slopes_q = np.zeros_like(slopes_np, dtype=np.int8)
            
        if intercept_max > 0:
            intercepts_q = (intercepts_np / intercept_max * 127).astype(np.int8)
        else:
            intercepts_q = np.zeros_like(intercepts_np, dtype=np.int8)
        
        return {
            'slopes': slopes_q,
            'intercepts': intercepts_q,
            'slope_scale': slope_max,
            'intercept_scale': intercept_max,
            'num_intervals': self.num_intervals,
            'input_range': (self.input_min, self.input_max),
        }


class SplineActivation(nn.Module):
    """
    Learnable activation function using splines.
    
    Instead of fixed ReLU/GELU, learn the optimal activation shape
    as a piecewise linear function.
    
    This is related to KAN (Kolmogorov-Arnold Networks) but simpler.
    """
    
    def __init__(
        self,
        num_intervals: int = 16,
        input_range: Tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__()
        
        self.num_intervals = num_intervals
        self.input_min, self.input_max = input_range
        
        # Initialize to approximate ReLU
        self.slopes = nn.Parameter(torch.ones(num_intervals))
        self.intercepts = nn.Parameter(torch.zeros(num_intervals))
        
        # Make left half (negative inputs) have slope 0 like ReLU
        with torch.no_grad():
            self.slopes[:num_intervals//2] = 0.01  # Small leak like LeakyReLU
            
        boundaries = torch.linspace(input_range[0], input_range[1], num_intervals + 1)
        self.register_buffer('boundaries', boundaries)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned activation."""
        # Get interval indices
        x_norm = (x - self.input_min) / (self.input_max - self.input_min)
        x_norm = x_norm * self.num_intervals
        indices = x_norm.floor().long().clamp(0, self.num_intervals - 1)
        
        # Gather coefficients
        slopes = self.slopes[indices]
        intercepts = self.intercepts[indices]
        
        # Evaluate
        return slopes * x + intercepts


def test_spline_doubling():
    """
    Test: Can a spline learn y = 2x?
    
    This is the simplest possible test - if this fails, nothing will work.
    """
    print("Testing SplineLayer on y = 2x")
    print("=" * 50)
    
    # Create spline
    spline = SplineLayer(
        in_features=1,
        out_features=1,
        num_intervals=16,
        input_range=(0.0, 255.0),
    )
    
    # Training data: y = 2x for x in [0, 127] (stays in 8-bit range)
    x_train = torch.arange(0, 128, dtype=torch.float32).unsqueeze(1)
    y_train = 2 * x_train
    
    # Optimizer
    optimizer = torch.optim.Adam(spline.parameters(), lr=0.1)
    
    # Train
    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = spline(x_train)
        loss = F.mse_loss(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # Evaluate
    spline.eval()
    with torch.no_grad():
        y_pred = spline(x_train)
        
        # Check accuracy
        errors = (y_pred - y_train).abs()
        max_error = errors.max().item()
        mean_error = errors.mean().item()
        
        # Count exact matches (within 0.5 for rounding)
        exact = (errors < 0.5).sum().item()
        
    print(f"\nResults:")
    print(f"  Max error: {max_error:.4f}")
    print(f"  Mean error: {mean_error:.4f}")
    print(f"  Exact matches: {exact}/{len(x_train)} ({100*exact/len(x_train):.1f}%)")
    
    # Sample outputs
    print(f"\nSample predictions:")
    for x_val in [0, 10, 50, 100, 127]:
        x_t = torch.tensor([[float(x_val)]])
        y_p = spline(x_t).item()
        y_true = 2 * x_val
        print(f"  x={x_val}: predicted={y_p:.2f}, true={y_true}")
    
    return max_error < 1.0  # Pass if max error < 1


def test_spline_adc_simple():
    """
    Test: Can a spline learn ADC for fixed operands?
    
    Simplified: A + 50 (no carry)
    """
    print("\n" + "=" * 50)
    print("Testing SplineLayer on A + 50 (simplified ADC)")
    print("=" * 50)
    
    spline = SplineLayer(
        in_features=1,
        out_features=1,
        num_intervals=32,
        input_range=(0.0, 255.0),
    )
    
    # Training data: y = (A + 50) % 256
    x_train = torch.arange(0, 256, dtype=torch.float32).unsqueeze(1)
    y_train = ((x_train + 50) % 256).float()
    
    optimizer = torch.optim.Adam(spline.parameters(), lr=0.05)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = spline(x_train)
        loss = F.mse_loss(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # Evaluate
    spline.eval()
    with torch.no_grad():
        y_pred = spline(x_train)
        y_pred_rounded = y_pred.round()
        
        errors = (y_pred_rounded - y_train).abs()
        exact = (errors < 0.5).sum().item()
        
    print(f"\nResults:")
    print(f"  Exact matches: {exact}/256 ({100*exact/256:.1f}%)")
    
    # Show the wrap-around region
    print(f"\nWrap-around region (A=200-210):")
    for a in range(200, 211):
        x_t = torch.tensor([[float(a)]])
        y_p = spline(x_t).round().item()
        y_true = (a + 50) % 256
        status = "✓" if y_p == y_true else "✗"
        print(f"  {a} + 50 = {y_true}, predicted = {int(y_p)} {status}")
    
    return exact == 256


if __name__ == '__main__':
    print("Lookup-Spline Neural Networks")
    print("Testing on simple functions before 6502 deployment\n")
    
    test1 = test_spline_doubling()
    test2 = test_spline_adc_simple()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  y = 2x:     {'PASS' if test1 else 'FAIL'}")
    print(f"  A + 50:     {'PASS' if test2 else 'FAIL'}")
