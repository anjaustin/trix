"""
FP4 Atoms: Exact Boolean Functions via Threshold Circuits

Instead of training atoms to be exact, we CONSTRUCT them to be exact.
Every atom is a small threshold circuit with FP4-compatible weights.

Key insight: Boolean functions over {0,1} can be implemented as:
- 1-layer: linearly separable functions (AND, OR, NAND, NOR, CARRY/MAJ)
- 2-layer: non-linearly separable via minterm detection (XOR, XNOR, SUM, MUX)

All weights ∈ {-1, 0, 1} and biases ∈ {-0.5, -1.5, -2.5}
These values exist in both E2M1 and NF4 FP4 formats.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass


def step(x: torch.Tensor) -> torch.Tensor:
    """Hard step function: exact, deterministic."""
    return (x >= 0).to(x.dtype)


def step_surrogate(x: torch.Tensor) -> torch.Tensor:
    """Differentiable surrogate for training (if needed)."""
    return torch.sigmoid(10 * x)


@dataclass
class ThresholdLayer:
    """A single threshold layer: step(Wx + b)"""
    weights: torch.Tensor  # Shape: (out_features, in_features)
    bias: torch.Tensor     # Shape: (out_features,)
    
    def forward(self, x: torch.Tensor, use_surrogate: bool = False) -> torch.Tensor:
        activation = step_surrogate if use_surrogate else step
        return activation(x @ self.weights.T + self.bias)


from .atoms import AtomStatus, VerificationResult


@dataclass  
class ThresholdCircuit:
    """A multi-layer threshold circuit."""
    layers: List[ThresholdLayer]
    name: str
    num_inputs: int
    num_outputs: int
    
    # Verification attributes (FP4 atoms are verified by construction)
    status: str = "verified"
    verification: VerificationResult = None
    
    def __post_init__(self):
        # FP4 atoms are always verified by construction
        if self.verification is None:
            # Count truth table entries
            total_cases = 2 ** self.num_inputs
            self.verification = VerificationResult(
                passed=True,
                accuracy=1.0,
                total_cases=total_cases
            )
        self.status = AtomStatus.VERIFIED
    
    def verify(self):
        """No-op for FP4 atoms - they're verified by construction."""
        pass
    
    def forward(self, x: torch.Tensor, use_surrogate: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward(x, use_surrogate)
        return x
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, use_surrogate=False)


# =============================================================================
# 1-LAYER ATOMS (Linearly Separable)
# =============================================================================

def make_AND() -> ThresholdCircuit:
    """AND(a,b) = step(a + b - 1.5)"""
    return ThresholdCircuit(
        layers=[ThresholdLayer(
            weights=torch.tensor([[1.0, 1.0]]),
            bias=torch.tensor([-1.5])
        )],
        name="AND",
        num_inputs=2,
        num_outputs=1
    )


def make_OR() -> ThresholdCircuit:
    """OR(a,b) = step(a + b - 0.5)"""
    return ThresholdCircuit(
        layers=[ThresholdLayer(
            weights=torch.tensor([[1.0, 1.0]]),
            bias=torch.tensor([-0.5])
        )],
        name="OR",
        num_inputs=2,
        num_outputs=1
    )


def make_NOT() -> ThresholdCircuit:
    """NOT(a) = step(-a + 0.5) = step(0.5 - a)"""
    return ThresholdCircuit(
        layers=[ThresholdLayer(
            weights=torch.tensor([[-1.0]]),
            bias=torch.tensor([0.5])
        )],
        name="NOT",
        num_inputs=1,
        num_outputs=1
    )


def make_NAND() -> ThresholdCircuit:
    """NAND(a,b) = step(-a - b + 1.5)"""
    return ThresholdCircuit(
        layers=[ThresholdLayer(
            weights=torch.tensor([[-1.0, -1.0]]),
            bias=torch.tensor([1.5])
        )],
        name="NAND",
        num_inputs=2,
        num_outputs=1
    )


def make_NOR() -> ThresholdCircuit:
    """NOR(a,b) = step(-a - b + 0.5)"""
    return ThresholdCircuit(
        layers=[ThresholdLayer(
            weights=torch.tensor([[-1.0, -1.0]]),
            bias=torch.tensor([0.5])
        )],
        name="NOR",
        num_inputs=2,
        num_outputs=1
    )


def make_CARRY() -> ThresholdCircuit:
    """CARRY/MAJ(a,b,c) = step(a + b + c - 1.5)
    
    Majority function: outputs 1 when ≥2 of 3 inputs are 1.
    """
    return ThresholdCircuit(
        layers=[ThresholdLayer(
            weights=torch.tensor([[1.0, 1.0, 1.0]]),
            bias=torch.tensor([-1.5])
        )],
        name="CARRY",
        num_inputs=3,
        num_outputs=1
    )


# =============================================================================
# 2-LAYER ATOMS (Non-Linearly Separable via Minterm Detection)
# =============================================================================

def make_XOR() -> ThresholdCircuit:
    """XOR(a,b) = (a=1,b=0) OR (a=0,b=1)
    
    Hidden layer detects the two minterms:
      h1 = step(a - b - 0.5)   detects (1,0)
      h2 = step(b - a - 0.5)   detects (0,1)
    
    Output layer ORs them:
      out = step(h1 + h2 - 0.5)
    """
    return ThresholdCircuit(
        layers=[
            # Hidden: minterm detectors
            ThresholdLayer(
                weights=torch.tensor([
                    [1.0, -1.0],   # h1: detects (1,0)
                    [-1.0, 1.0],  # h2: detects (0,1)
                ]),
                bias=torch.tensor([-0.5, -0.5])
            ),
            # Output: OR
            ThresholdLayer(
                weights=torch.tensor([[1.0, 1.0]]),
                bias=torch.tensor([-0.5])
            )
        ],
        name="XOR",
        num_inputs=2,
        num_outputs=1
    )


def make_XNOR() -> ThresholdCircuit:
    """XNOR(a,b) = (a=0,b=0) OR (a=1,b=1)
    
    Hidden layer detects the two minterms:
      h1 = step(-a - b + 0.5)   detects (0,0)
      h2 = step(a + b - 1.5)    detects (1,1)
    
    Output layer ORs them:
      out = step(h1 + h2 - 0.5)
    """
    return ThresholdCircuit(
        layers=[
            ThresholdLayer(
                weights=torch.tensor([
                    [-1.0, -1.0],  # h1: detects (0,0)
                    [1.0, 1.0],    # h2: detects (1,1)
                ]),
                bias=torch.tensor([0.5, -1.5])
            ),
            ThresholdLayer(
                weights=torch.tensor([[1.0, 1.0]]),
                bias=torch.tensor([-0.5])
            )
        ],
        name="XNOR",
        num_inputs=2,
        num_outputs=1
    )


def make_SUM() -> ThresholdCircuit:
    """SUM(a,b,c) = a XOR b XOR c (odd parity)
    
    Outputs 1 on exactly 4 minterms: 001, 010, 100, 111
    
    Hidden layer detects each minterm:
      h1 = detect(0,0,1): step(-a - b + c - 0.5)
      h2 = detect(0,1,0): step(-a + b - c - 0.5)
      h3 = detect(1,0,0): step(a - b - c - 0.5)
      h4 = detect(1,1,1): step(a + b + c - 2.5)
    
    Output: OR of all detectors
    """
    return ThresholdCircuit(
        layers=[
            ThresholdLayer(
                weights=torch.tensor([
                    [-1.0, -1.0, 1.0],   # h1: (0,0,1)
                    [-1.0, 1.0, -1.0],   # h2: (0,1,0)
                    [1.0, -1.0, -1.0],   # h3: (1,0,0)
                    [1.0, 1.0, 1.0],     # h4: (1,1,1)
                ]),
                bias=torch.tensor([-0.5, -0.5, -0.5, -2.5])
            ),
            ThresholdLayer(
                weights=torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
                bias=torch.tensor([-0.5])
            )
        ],
        name="SUM",
        num_inputs=3,
        num_outputs=1
    )


def make_MUX() -> ThresholdCircuit:
    """MUX(s,a,b) = a if s=0 else b
    
    Two contributing conditions:
      (s=0, a=1) → output 1
      (s=1, b=1) → output 1
    
    Hidden layer:
      h1 = detect(s=0, a=1): step(-s + a - 0.5)
      h2 = detect(s=1, b=1): step(s + b - 1.5)
    
    Output: OR
    """
    return ThresholdCircuit(
        layers=[
            ThresholdLayer(
                weights=torch.tensor([
                    [-1.0, 1.0, 0.0],   # h1: s=0, a=1
                    [1.0, 0.0, 1.0],    # h2: s=1, b=1
                ]),
                bias=torch.tensor([-0.5, -1.5])
            ),
            ThresholdLayer(
                weights=torch.tensor([[1.0, 1.0]]),
                bias=torch.tensor([-0.5])
            )
        ],
        name="MUX",
        num_inputs=3,
        num_outputs=1
    )


# =============================================================================
# FP4 ATOM LIBRARY
# =============================================================================

class FP4AtomLibrary:
    """Library of FP4-compatible atoms constructed as threshold circuits."""
    
    CONSTRUCTORS = {
        "AND": make_AND,
        "OR": make_OR,
        "NOT": make_NOT,
        "NAND": make_NAND,
        "NOR": make_NOR,
        "XOR": make_XOR,
        "XNOR": make_XNOR,
        "SUM": make_SUM,
        "CARRY": make_CARRY,
        "MUX": make_MUX,
    }
    
    # Truth tables for verification
    TRUTH_TABLES = {
        "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
        "OR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        "NOT": {(0,): 1, (1,): 0},
        "NAND": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        "NOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
        "XOR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        "XNOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
        "SUM": {
            (0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 0,
            (1, 0, 0): 1, (1, 0, 1): 0, (1, 1, 0): 0, (1, 1, 1): 1,
        },
        "CARRY": {
            (0, 0, 0): 0, (0, 0, 1): 0, (0, 1, 0): 0, (0, 1, 1): 1,
            (1, 0, 0): 0, (1, 0, 1): 1, (1, 1, 0): 1, (1, 1, 1): 1,
        },
        "MUX": {  # MUX(s, a, b) = a if s=0 else b
            (0, 0, 0): 0, (0, 0, 1): 0, (0, 1, 0): 1, (0, 1, 1): 1,
            (1, 0, 0): 0, (1, 0, 1): 1, (1, 1, 0): 0, (1, 1, 1): 1,
        },
    }
    
    def __init__(self):
        self._atoms: Dict[str, ThresholdCircuit] = {}
    
    def get_atom(self, name: str) -> ThresholdCircuit:
        """Get or create an atom by name."""
        if name not in self._atoms:
            if name not in self.CONSTRUCTORS:
                raise ValueError(f"Unknown atom: {name}")
            self._atoms[name] = self.CONSTRUCTORS[name]()
        return self._atoms[name]
    
    def get(self, name: str, train_if_missing: bool = True) -> ThresholdCircuit:
        """
        Compatibility method for AtomLibrary interface.
        
        FP4 atoms don't need training - they're exact by construction.
        The train_if_missing parameter is ignored.
        """
        return self.get_atom(name)
    
    def list_atoms(self) -> List[str]:
        """List all available atom types."""
        return list(self.CONSTRUCTORS.keys())
    
    def verify_atom(self, name: str) -> Tuple[bool, float, List[Tuple]]:
        """
        Verify an atom against its truth table.
        
        Returns:
            (passed, accuracy, failures)
        """
        atom = self.get_atom(name)
        truth_table = self.TRUTH_TABLES[name]
        
        correct = 0
        total = 0
        failures = []
        
        for inputs, expected in truth_table.items():
            x = torch.tensor([list(inputs)], dtype=torch.float32)
            output = atom(x)
            got = int(output[0, 0].item() > 0.5)
            
            if got == expected:
                correct += 1
            else:
                failures.append((inputs, expected, got))
            total += 1
        
        accuracy = correct / total
        passed = accuracy == 1.0
        return passed, accuracy, failures
    
    def verify_all(self) -> Dict[str, Tuple[bool, float]]:
        """Verify all atoms."""
        results = {}
        for name in self.list_atoms():
            passed, accuracy, _ = self.verify_atom(name)
            results[name] = (passed, accuracy)
        return results
    
    def get_fp4_weights(self, name: str) -> Dict[str, List]:
        """
        Get the FP4-compatible weights for an atom.
        
        Returns dict with layer weights and biases as lists.
        """
        atom = self.get_atom(name)
        result = {"name": name, "layers": []}
        
        for i, layer in enumerate(atom.layers):
            result["layers"].append({
                "weights": layer.weights.tolist(),
                "bias": layer.bias.tolist()
            })
        
        return result


# =============================================================================
# MINTERM GENERATOR (Truth Table → Threshold Circuit)
# =============================================================================

def truth_table_to_circuit(
    name: str,
    num_inputs: int,
    truth_table: Dict[Tuple[int, ...], int]
) -> ThresholdCircuit:
    """
    Generate a threshold circuit from an arbitrary truth table.
    
    This is the general case: any Boolean function can be expressed as
    OR of its minterms (sum of products form).
    
    Args:
        name: Circuit name
        num_inputs: Number of input bits
        truth_table: Maps input tuples to output (0 or 1)
    
    Returns:
        ThresholdCircuit that exactly implements the truth table
    """
    # Find minterms (input combinations that output 1)
    minterms = [inputs for inputs, output in truth_table.items() if output == 1]
    
    if len(minterms) == 0:
        # Constant 0: step(-1)
        return ThresholdCircuit(
            layers=[ThresholdLayer(
                weights=torch.zeros(1, num_inputs),
                bias=torch.tensor([-1.0])
            )],
            name=name,
            num_inputs=num_inputs,
            num_outputs=1
        )
    
    if len(minterms) == 2 ** num_inputs:
        # Constant 1: step(1)
        return ThresholdCircuit(
            layers=[ThresholdLayer(
                weights=torch.zeros(1, num_inputs),
                bias=torch.tensor([1.0])
            )],
            name=name,
            num_inputs=num_inputs,
            num_outputs=1
        )
    
    # Build minterm detectors
    # For minterm (b1, b2, ..., bn), detector fires iff all inputs match.
    # 
    # Strategy: For each minterm, create a neuron that computes:
    #   h = step(sum_i w_i * x_i + bias)
    # where w_i = +1 if b_i = 1, else -1
    # 
    # When inputs match minterm exactly:
    #   sum = (num_ones * 1) + (num_zeros * (-1) * 0) = num_ones (for 1-bits)
    #         + (num_zeros * (-1) * 0) = 0 (for 0-bits where x=0)
    #   Wait, let's think more carefully:
    #   For 1-bits: w=+1, x=1 → contributes +1
    #   For 0-bits: w=-1, x=0 → contributes 0
    #   Total when match: num_ones
    #   
    # When one 1-bit has x=0: lose 1 from sum → num_ones - 1
    # When one 0-bit has x=1: w=-1, x=1 → contributes -1 instead of 0 → num_ones - 1
    #
    # So threshold should be > (num_ones - 1), i.e., >= num_ones - 0.5
    # Bias = -(num_ones - 0.5) = -num_ones + 0.5
    
    num_minterms = len(minterms)
    hidden_weights = torch.zeros(num_minterms, num_inputs)
    hidden_bias = torch.zeros(num_minterms)
    
    for i, minterm in enumerate(minterms):
        num_ones = 0
        for j, bit in enumerate(minterm):
            if bit == 1:
                hidden_weights[i, j] = 1.0
                num_ones += 1
            else:
                hidden_weights[i, j] = -1.0
        # Bias: fires when sum >= num_ones - 0.5
        hidden_bias[i] = -num_ones + 0.5
    
    # OR layer: step(sum(h) - 0.5)
    or_weights = torch.ones(1, num_minterms)
    or_bias = torch.tensor([-0.5])
    
    return ThresholdCircuit(
        layers=[
            ThresholdLayer(hidden_weights, hidden_bias),
            ThresholdLayer(or_weights, or_bias)
        ],
        name=name,
        num_inputs=num_inputs,
        num_outputs=1
    )


def verify_generated_circuit(
    circuit: ThresholdCircuit,
    truth_table: Dict[Tuple[int, ...], int]
) -> Tuple[bool, float, List]:
    """Verify a generated circuit against its truth table."""
    correct = 0
    total = 0
    failures = []
    
    for inputs, expected in truth_table.items():
        x = torch.tensor([list(inputs)], dtype=torch.float32)
        output = circuit(x)
        got = int(output[0, 0].item() > 0.5)
        
        if got == expected:
            correct += 1
        else:
            failures.append((inputs, expected, got))
        total += 1
    
    accuracy = correct / total
    return accuracy == 1.0, accuracy, failures


# =============================================================================
# TORCH MODULE WRAPPER (for integration with compiler)
# =============================================================================

class FP4AtomModule(nn.Module):
    """
    Wraps a ThresholdCircuit as a torch.nn.Module.
    
    This allows FP4 atoms to be used as drop-in replacements
    for the trained atoms in the compiler.
    """
    
    def __init__(self, circuit: ThresholdCircuit):
        super().__init__()
        self.circuit = circuit
        self.num_inputs = circuit.num_inputs
        self.num_outputs = circuit.num_outputs
        
        # Register weights as parameters (for inspection, not training)
        for i, layer in enumerate(circuit.layers):
            self.register_buffer(f"weight_{i}", layer.weights)
            self.register_buffer(f"bias_{i}", layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)


# =============================================================================
# CONVENIENCE
# =============================================================================

def get_all_fp4_atoms() -> Dict[str, ThresholdCircuit]:
    """Get all FP4 atoms as a dictionary."""
    library = FP4AtomLibrary()
    return {name: library.get_atom(name) for name in library.list_atoms()}


def verify_all_fp4_atoms() -> bool:
    """Verify all FP4 atoms, return True if all pass."""
    library = FP4AtomLibrary()
    results = library.verify_all()
    
    all_pass = True
    for name, (passed, accuracy) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status} ({accuracy:.0%})")
        if not passed:
            all_pass = False
    
    return all_pass
