#!/usr/bin/env python3
"""
Quantization-Aware Training Example

Demonstrates progressive quantization for ternary weight training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim

from trix import (
    TriXLinearQAT,
    QATTrainer,
    progressive_quantization_schedule,
)


class SimpleModel(nn.Module):
    """Simple model using QAT layers."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = TriXLinearQAT(input_dim, hidden_dim, quant_mode='progressive')
        self.relu = nn.ReLU()
        self.layer2 = TriXLinearQAT(hidden_dim, output_dim, quant_mode='progressive')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def main():
    print("=" * 60)
    print("TriX Quantization-Aware Training Example")
    print("=" * 60)
    
    # Create model
    model = SimpleModel(input_dim=64, hidden_dim=128, output_dim=32)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup QAT trainer
    total_epochs = 20
    qat_trainer = QATTrainer(
        model,
        total_epochs=total_epochs,
        start_temp=1.0,
        end_temp=10.0
    )
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Dummy data
    x = torch.randn(32, 64)
    target = torch.randn(32, 32)
    
    print("\nTraining with progressive quantization:")
    print("-" * 60)
    print(f"{'Epoch':<8} {'Temp':<10} {'Loss':<12} {'Sparsity':<12} {'Distribution'}")
    print("-" * 60)
    
    for epoch in range(total_epochs):
        # Update quantization temperature
        temp = qat_trainer.step_epoch()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Metrics
        sparsity = qat_trainer.get_model_sparsity()
        dist = model.layer1.get_ternary_distribution()
        
        if epoch % 2 == 0 or epoch == total_epochs - 1:
            print(f"{epoch + 1:<8} {temp:<10.2f} {loss.item():<12.4f} {sparsity:<12.1%} "
                  f"neg={dist['neg']:.1%} zero={dist['zero']:.1%} pos={dist['pos']:.1%}")
    
    # Final model analysis
    print("\n" + "-" * 60)
    print("Final Model Analysis:")
    print("-" * 60)
    
    for name, module in model.named_modules():
        if isinstance(module, TriXLinearQAT):
            dist = module.get_ternary_distribution()
            sparsity = module.get_sparsity()
            print(f"{name}:")
            print(f"  Sparsity: {sparsity:.1%}")
            print(f"  Distribution: -1={dist['neg']:.1%}, 0={dist['zero']:.1%}, +1={dist['pos']:.1%}")
    
    # Demonstrate temperature schedule
    print("\n" + "-" * 60)
    print("Temperature Schedule Visualization:")
    print("-" * 60)
    
    for e in [0, 25, 50, 75, 100]:
        temp = progressive_quantization_schedule(e, 100, 1.0, 10.0)
        bar = "#" * int(temp * 5)
        print(f"Epoch {e:3d}: temp={temp:5.2f} {bar}")
    
    print("\n" + "=" * 60)
    print("Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
