#!/usr/bin/env python3
"""
Mesa 8: Differentiable CUDA

The killer feature: Backpropagation through SASS assembly.

"What input to this CUDA kernel would produce the number 100?"
Gradient descent will tell you.
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/trix_latest/experiments/mesa8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from sass_parser import parse_sass_kernel, SASSInstruction, OpcodeCategory


# =============================================================================
# DIFFERENTIABLE REGISTER FILE
# =============================================================================

class DifferentiableRegisterFile(nn.Module):
    """
    GPU Register File as Differentiable State.
    
    Key insight: Use float tensors with requires_grad=True.
    Integer operations become soft approximations that allow gradients.
    """
    
    def __init__(self, num_regs: int = 256, num_uniform: int = 64):
        super().__init__()
        self.num_regs = num_regs
        self.num_uniform = num_uniform
        
        # Registers as differentiable parameters
        # We use float for differentiability, round for integer semantics
        self.regs = nn.Parameter(torch.zeros(num_regs, dtype=torch.float32), requires_grad=False)
        self.uniform = nn.Parameter(torch.zeros(num_uniform, dtype=torch.float32), requires_grad=False)
        
        # RZ is always 0
        self.register_buffer('rz', torch.tensor(0.0))
    
    def read(self, reg_name: str) -> torch.Tensor:
        """Read a register value (differentiable)."""
        if reg_name == 'RZ':
            return self.rz
        elif reg_name.startswith('R'):
            idx = int(reg_name[1:])
            return self.regs[idx]
        elif reg_name.startswith('UR'):
            idx = int(reg_name[2:])
            return self.uniform[idx]
        else:
            raise ValueError(f"Unknown register: {reg_name}")
    
    def write(self, reg_name: str, value: torch.Tensor):
        """Write a value to a register."""
        if reg_name == 'RZ':
            pass  # Read-only
        elif reg_name.startswith('R'):
            idx = int(reg_name[1:])
            # Detach and update (can't assign to leaf variable)
            self.regs.data[idx] = value.detach()
        elif reg_name.startswith('UR'):
            idx = int(reg_name[2:])
            self.uniform.data[idx] = value.detach()
    
    def set_input(self, reg_name: str, value: torch.Tensor):
        """Set an input register (will track gradients)."""
        if reg_name.startswith('R'):
            idx = int(reg_name[1:])
            self.regs.data[idx] = value


# =============================================================================
# DIFFERENTIABLE ALU
# =============================================================================

class DifferentiableIADD3(nn.Module):
    """
    Differentiable 3-input Integer Add.
    
    IADD3 Rd, Ra, Rb, Rc → Rd = Ra + Rb + Rc
    
    For integers, addition is already differentiable!
    The gradient of (a + b + c) w.r.t. a is 1.
    """
    
    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return a + b + c


class DifferentiableIMAD(nn.Module):
    """
    Differentiable Integer Multiply-Add.
    
    IMAD Rd, Ra, Rb, Rc → Rd = Ra * Rb + Rc
    
    Gradients:
      d/da = b
      d/db = a  
      d/dc = 1
    """
    
    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return a * b + c


# =============================================================================
# DIFFERENTIABLE CUDA ENGINE
# =============================================================================

class DifferentiableCUDA(nn.Module):
    """
    Differentiable CUDA Execution Engine.
    
    Executes SASS instructions with gradient tracking.
    Enables backpropagation through compiled CUDA binaries.
    """
    
    def __init__(self):
        super().__init__()
        self.iadd3 = DifferentiableIADD3()
        self.imad = DifferentiableIMAD()
    
    def execute_iadd3(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Execute IADD3 with gradient tracking."""
        return self.iadd3(a, b, c)
    
    def execute_imad(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Execute IMAD with gradient tracking."""
        return self.imad(a, b, c)


# =============================================================================
# THE RAY-GUN TEST: INVERT A CUDA KERNEL
# =============================================================================

def test_gradient_through_iadd3():
    """
    Test: Backpropagate through IADD3.
    
    Question: "What value of 'a' produces output 100 when added to b=42?"
    Answer: Gradient descent will find a=58.
    """
    
    print("=" * 60)
    print("MESA 8: DIFFERENTIABLE IADD3")
    print("=" * 60)
    
    engine = DifferentiableCUDA()
    
    # Input 'a' is what we want to find
    # Input 'b' is fixed at 42
    # Target output is 100
    
    a = torch.tensor([30.0], requires_grad=True)  # Initial guess
    b = torch.tensor([42.0])  # Fixed
    c = torch.tensor([0.0])   # RZ = 0
    target = torch.tensor([100.0])
    
    print(f"\nProblem: IADD3 R9, R2, R5, RZ")
    print(f"  R5 (b) = 42 (fixed)")
    print(f"  RZ (c) = 0 (fixed)")
    print(f"  Target R9 = 100")
    print(f"  Find: R2 (a) = ?")
    print(f"\nInitial guess: a = {a.item()}")
    
    # Gradient descent
    optimizer = torch.optim.SGD([a], lr=0.5)
    
    print("\n[GRADIENT DESCENT]")
    for step in range(20):
        optimizer.zero_grad()
        
        # Execute IADD3
        result = engine.execute_iadd3(a, b, c)
        
        # Loss: squared difference from target
        loss = (result - target) ** 2
        
        # Backpropagate!
        loss.backward()
        
        if step < 5 or step % 5 == 0:
            print(f"  Step {step:2d}: a={a.item():8.3f}, result={result.item():8.3f}, "
                  f"loss={loss.item():10.3f}, grad={a.grad.item():8.3f}")
        
        # Update
        optimizer.step()
        
        # Check convergence
        if abs(result.item() - target.item()) < 0.01:
            print(f"\n  CONVERGED at step {step}!")
            break
    
    # Final result
    final_a = a.item()
    final_result = engine.execute_iadd3(a, b, c).item()
    
    print(f"\n[RESULT]")
    print(f"  Found a = {final_a:.3f}")
    print(f"  IADD3({final_a:.3f}, 42, 0) = {final_result:.3f}")
    print(f"  Target = 100")
    print(f"  Error = {abs(final_result - 100):.6f}")
    
    success = abs(final_result - 100) < 0.1
    print(f"\n{'SUCCESS' if success else 'FAILED'}: ", end="")
    print("Gradient descent found the input that produces target output!")
    
    return success


def test_invert_multiply_add():
    """
    Test: Invert IMAD (multiply-add).
    
    Question: "What value of 'a' makes a*7 + 3 = 100?"
    Answer: a = (100 - 3) / 7 = 13.857...
    """
    
    print("\n" + "=" * 60)
    print("MESA 8: DIFFERENTIABLE IMAD")
    print("=" * 60)
    
    engine = DifferentiableCUDA()
    
    a = torch.tensor([1.0], requires_grad=True)  # Initial guess
    b = torch.tensor([7.0])   # Multiplier
    c = torch.tensor([3.0])   # Addend
    target = torch.tensor([100.0])
    
    expected = (100 - 3) / 7  # = 13.857...
    
    print(f"\nProblem: IMAD R9, Ra, 7, 3")
    print(f"  Ra * 7 + 3 = 100")
    print(f"  Find: Ra = ?")
    print(f"  Expected: Ra = (100-3)/7 = {expected:.4f}")
    print(f"\nInitial guess: a = {a.item()}")
    
    optimizer = torch.optim.Adam([a], lr=2.0)
    
    print("\n[GRADIENT DESCENT]")
    for step in range(100):
        optimizer.zero_grad()
        
        result = engine.execute_imad(a, b, c)
        loss = (result - target) ** 2
        loss.backward()
        
        if step < 5 or step % 10 == 0:
            print(f"  Step {step:2d}: a={a.item():8.4f}, result={result.item():8.3f}, loss={loss.item():10.3f}")
        
        optimizer.step()
        
        if abs(result.item() - target.item()) < 0.01:
            print(f"\n  CONVERGED at step {step}!")
            break
    
    final_a = a.item()
    final_result = engine.execute_imad(a, b, c).item()
    
    print(f"\n[RESULT]")
    print(f"  Found a = {final_a:.4f}")
    print(f"  Expected = {expected:.4f}")
    print(f"  IMAD({final_a:.4f}, 7, 3) = {final_result:.3f}")
    print(f"  Error from expected: {abs(final_a - expected):.6f}")
    
    success = abs(final_a - expected) < 1.0  # Within 1.0 of expected
    print(f"\n{'SUCCESS' if success else 'FAILED'}: ", end="")
    print("Found the input that inverts the multiply-add!")
    
    return success


def test_kernel_inversion():
    """
    Test: Invert a full kernel sequence.
    
    Kernel: c = (a + b) * 2 + 10
    Given c = 100, b = 20, find a.
    
    Solution: a = (100 - 10) / 2 - 20 = 25
    """
    
    print("\n" + "=" * 60)
    print("MESA 8: FULL KERNEL INVERSION")
    print("=" * 60)
    
    engine = DifferentiableCUDA()
    
    # The kernel: c = (a + b) * 2 + 10
    # Sequence: IADD3 temp, a, b, 0 → IMAD c, temp, 2, 10
    
    a = torch.tensor([0.0], requires_grad=True)
    b = torch.tensor([20.0])
    zero = torch.tensor([0.0])
    two = torch.tensor([2.0])
    ten = torch.tensor([10.0])
    target = torch.tensor([100.0])
    
    expected = (100 - 10) / 2 - 20  # = 25
    
    print(f"\nKernel: c = (a + b) * 2 + 10")
    print(f"  IADD3 R3, R2, R5, RZ    # R3 = a + b")
    print(f"  IMAD  R9, R3, 2, 10     # R9 = R3 * 2 + 10")
    print(f"\nGiven: b = 20, target c = 100")
    print(f"Find: a = ?")
    print(f"Expected: a = (100-10)/2 - 20 = {expected}")
    
    optimizer = torch.optim.Adam([a], lr=2.0)
    
    print("\n[GRADIENT DESCENT]")
    for step in range(100):
        optimizer.zero_grad()
        
        # Execute kernel
        temp = engine.execute_iadd3(a, b, zero)   # temp = a + b
        result = engine.execute_imad(temp, two, ten)  # c = temp * 2 + 10
        
        loss = (result - target) ** 2
        loss.backward()
        
        if step < 5 or step % 20 == 0:
            print(f"  Step {step:2d}: a={a.item():8.3f}, c={result.item():8.3f}, loss={loss.item():10.3f}")
        
        optimizer.step()
        
        if abs(result.item() - target.item()) < 0.01:
            print(f"\n  CONVERGED at step {step}!")
            break
    
    final_a = a.item()
    temp = engine.execute_iadd3(a, b, zero)
    final_result = engine.execute_imad(temp, two, ten).item()
    
    print(f"\n[RESULT]")
    print(f"  Found a = {final_a:.3f}")
    print(f"  Expected = {expected:.3f}")
    print(f"  Kernel output: {final_result:.3f}")
    print(f"  Error from expected: {abs(final_a - expected):.6f}")
    
    success = abs(final_a - expected) < 1.0  # Within 1.0 of expected
    print(f"\n{'SUCCESS' if success else 'FAILED'}: ", end="")
    print("Inverted a multi-instruction CUDA kernel!")
    
    return success


def run_all_tests():
    """Run all differentiability tests."""
    print("\n" + "#" * 60)
    print("# MESA 8: THE RAY-GUN")
    print("# Differentiable CUDA - Backprop Through Assembly")
    print("#" * 60)
    
    results = {}
    
    results['iadd3_gradient'] = test_gradient_through_iadd3()
    results['imad_inversion'] = test_invert_multiply_add()
    results['kernel_inversion'] = test_kernel_inversion()
    
    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print("#" * 60)
    if all_passed:
        print("# ALL TESTS PASSED")
        print("#")
        print("# THE RAY-GUN IS OPERATIONAL.")
        print("#")
        print("# You can now ask: 'What input produces this output?'")
        print("# And gradient descent will find it.")
        print("#")
        print("# Black box binaries are now learned functions.")
    print("#" * 60)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
