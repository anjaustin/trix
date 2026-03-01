"""
Atom Library - Verified Primitive Operations

An Atom is the smallest unit of exact computation.
Each atom is:
  - Trained to 100% accuracy on its bounded domain
  - Verified exhaustively
  - Immutable after verification

The AtomLibrary provides pre-verified atoms for common operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import json
from pathlib import Path


class AtomStatus(Enum):
    """Verification status of an atom"""
    UNVERIFIED = "unverified"
    TRAINING = "training"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class AtomSignature:
    """Type signature for an atom"""
    name: str
    input_bits: int
    output_bits: int
    description: str = ""
    
    def __hash__(self):
        return hash((self.name, self.input_bits, self.output_bits))


@dataclass 
class VerificationResult:
    """Result of atom verification"""
    passed: bool
    accuracy: float
    total_cases: int
    failed_cases: List[Tuple] = field(default_factory=list)
    
    def __str__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"{status}: {self.accuracy*100:.2f}% ({self.total_cases} cases)"


class AtomNetwork(nn.Module):
    """
    Neural network implementation of an atom.
    
    Architecture designed for exact discrete computation:
    - Multiple hidden layers for capacity
    - Sigmoid output for binary classification
    - Trainable to 100% on bounded domains
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return discrete 0/1 predictions"""
        with torch.no_grad():
            return (self.forward(x) > 0.5).float()


class Atom:
    """
    A verified atomic operation.
    
    An atom encapsulates:
    - A neural network that computes the operation
    - A truth table defining correct behavior
    - Verification status and results
    """
    
    def __init__(
        self,
        signature: AtomSignature,
        truth_table: Optional[Callable] = None,
        network: Optional[AtomNetwork] = None,
    ):
        self.signature = signature
        self.truth_table = truth_table
        self.network = network or AtomNetwork(
            signature.input_bits, 
            signature.output_bits
        )
        self.status = AtomStatus.UNVERIFIED
        self.verification: Optional[VerificationResult] = None
        self._training_history: List[float] = []
    
    def generate_exhaustive_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all input/output pairs for verification"""
        if self.truth_table is None:
            raise ValueError("No truth table defined for atom")
        
        n_inputs = 2 ** self.signature.input_bits
        inputs = []
        outputs = []
        
        for i in range(n_inputs):
            # Convert integer to bit vector
            bits = [(i >> b) & 1 for b in range(self.signature.input_bits)]
            input_vec = torch.tensor(bits, dtype=torch.float32)
            
            # Get expected output from truth table
            output_val = self.truth_table(*bits)
            if isinstance(output_val, (int, float)):
                output_val = [output_val]
            output_vec = torch.tensor(output_val, dtype=torch.float32)
            
            inputs.append(input_vec)
            outputs.append(output_vec)
        
        return torch.stack(inputs), torch.stack(outputs)
    
    def train(self, epochs: int = 5000, lr: float = 0.01, 
              early_stop_acc: float = 1.0) -> bool:
        """Train the atom to exactness"""
        self.status = AtomStatus.TRAINING
        
        inputs, targets = self.generate_exhaustive_data()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            self._training_history.append(loss.item())
            
            # Check accuracy
            with torch.no_grad():
                preds = (outputs > 0.5).float()
                acc = (preds == targets).float().mean().item()
                
            if acc >= early_stop_acc:
                break
        
        return self.verify().passed
    
    def verify(self) -> VerificationResult:
        """Exhaustively verify the atom"""
        inputs, targets = self.generate_exhaustive_data()
        
        predictions = self.network.predict(inputs)
        
        correct = (predictions == targets).all(dim=1)
        accuracy = correct.float().mean().item()
        
        failed_cases = []
        for i, is_correct in enumerate(correct):
            if not is_correct:
                input_bits = inputs[i].tolist()
                expected = targets[i].tolist()
                got = predictions[i].tolist()
                failed_cases.append((input_bits, expected, got))
        
        self.verification = VerificationResult(
            passed=(accuracy == 1.0),
            accuracy=accuracy,
            total_cases=len(inputs),
            failed_cases=failed_cases[:10]  # Keep first 10 failures
        )
        
        self.status = AtomStatus.VERIFIED if accuracy == 1.0 else AtomStatus.FAILED
        return self.verification
    
    def __call__(self, *bits) -> torch.Tensor:
        """Execute the atom on input bits"""
        if isinstance(bits[0], torch.Tensor):
            x = bits[0]
        else:
            x = torch.tensor(bits, dtype=torch.float32)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return self.network.predict(x)
    
    def save(self, path: Path):
        """Save atom to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save network
        torch.save(self.network.state_dict(), path / "network.pt")
        
        # Save metadata
        meta = {
            "signature": {
                "name": self.signature.name,
                "input_bits": self.signature.input_bits,
                "output_bits": self.signature.output_bits,
                "description": self.signature.description,
            },
            "status": self.status.value,
            "verification": {
                "passed": self.verification.passed,
                "accuracy": self.verification.accuracy,
                "total_cases": self.verification.total_cases,
            } if self.verification else None
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, path: Path, truth_table: Callable = None) -> "Atom":
        """Load atom from disk"""
        path = Path(path)
        
        with open(path / "meta.json") as f:
            meta = json.load(f)
        
        sig = AtomSignature(**meta["signature"])
        atom = cls(sig, truth_table)
        
        atom.network.load_state_dict(torch.load(path / "network.pt"))
        atom.status = AtomStatus(meta["status"])
        
        if meta["verification"]:
            atom.verification = VerificationResult(
                passed=meta["verification"]["passed"],
                accuracy=meta["verification"]["accuracy"],
                total_cases=meta["verification"]["total_cases"],
            )
        
        return atom


# =============================================================================
# Standard Atom Definitions (Truth Tables)
# =============================================================================

def tt_and(a, b):
    """AND gate truth table"""
    return a & b

def tt_or(a, b):
    """OR gate truth table"""
    return a | b

def tt_xor(a, b):
    """XOR gate truth table"""
    return a ^ b

def tt_not(a):
    """NOT gate truth table"""
    return 1 - a

def tt_nand(a, b):
    """NAND gate truth table"""
    return 1 - (a & b)

def tt_nor(a, b):
    """NOR gate truth table"""
    return 1 - (a | b)

def tt_xnor(a, b):
    """XNOR gate truth table"""
    return 1 - (a ^ b)

def tt_sum(a, b, cin):
    """Full adder SUM output (parity)"""
    return a ^ b ^ cin

def tt_carry(a, b, cin):
    """Full adder CARRY output (majority)"""
    return (a & b) | (cin & (a ^ b))

def tt_mux(sel, a, b):
    """2:1 Multiplexer"""
    return b if sel else a

def tt_compare_lt(a, b):
    """Less-than comparison (single bit)"""
    return 1 if (a < b) else 0

def tt_compare_eq(a, b):
    """Equality comparison (single bit)"""
    return 1 if (a == b) else 0


# =============================================================================
# Atom Library
# =============================================================================

class AtomLibrary:
    """
    Library of pre-verified atoms.
    
    Provides:
    - Standard logic gates (AND, OR, XOR, NOT, etc.)
    - Arithmetic primitives (SUM, CARRY)
    - Control primitives (MUX, COMPARE)
    
    All atoms are verified to 100% accuracy on their bounded domains.
    """
    
    # Standard atom definitions
    STANDARD_ATOMS = {
        "AND": (AtomSignature("AND", 2, 1, "Logical AND"), tt_and),
        "OR": (AtomSignature("OR", 2, 1, "Logical OR"), tt_or),
        "XOR": (AtomSignature("XOR", 2, 1, "Logical XOR"), tt_xor),
        "NOT": (AtomSignature("NOT", 1, 1, "Logical NOT"), tt_not),
        "NAND": (AtomSignature("NAND", 2, 1, "Logical NAND"), tt_nand),
        "NOR": (AtomSignature("NOR", 2, 1, "Logical NOR"), tt_nor),
        "XNOR": (AtomSignature("XNOR", 2, 1, "Logical XNOR"), tt_xnor),
        "SUM": (AtomSignature("SUM", 3, 1, "Full adder sum (parity)"), tt_sum),
        "CARRY": (AtomSignature("CARRY", 3, 1, "Full adder carry (majority)"), tt_carry),
        "MUX": (AtomSignature("MUX", 3, 1, "2:1 Multiplexer"), tt_mux),
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._atoms: Dict[str, Atom] = {}
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load atoms from cache"""
        if self._loaded:
            return
            
        if self.cache_dir and self.cache_dir.exists():
            for name in self.STANDARD_ATOMS:
                atom_path = self.cache_dir / name
                if atom_path.exists():
                    sig, tt = self.STANDARD_ATOMS[name]
                    self._atoms[name] = Atom.load(atom_path, tt)
        
        self._loaded = True
    
    def get(self, name: str, train_if_missing: bool = True) -> Atom:
        """Get an atom by name, training if necessary"""
        self._ensure_loaded()
        
        if name in self._atoms:
            return self._atoms[name]
        
        if name not in self.STANDARD_ATOMS:
            raise ValueError(f"Unknown atom: {name}")
        
        sig, tt = self.STANDARD_ATOMS[name]
        atom = Atom(sig, tt)
        
        if train_if_missing:
            print(f"Training atom: {name}...")
            success = atom.train()
            if not success:
                raise RuntimeError(f"Failed to train atom {name} to exactness")
            print(f"  -> {atom.verification}")
            
            if self.cache_dir:
                atom.save(self.cache_dir / name)
        
        self._atoms[name] = atom
        return atom
    
    def register(self, name: str, signature: AtomSignature, 
                 truth_table: Callable) -> Atom:
        """Register a custom atom"""
        atom = Atom(signature, truth_table)
        self._atoms[name] = atom
        return atom
    
    def train_all(self) -> Dict[str, VerificationResult]:
        """Train and verify all standard atoms"""
        results = {}
        for name in self.STANDARD_ATOMS:
            atom = self.get(name, train_if_missing=True)
            results[name] = atom.verification
        return results
    
    def list_atoms(self) -> List[str]:
        """List available atom names"""
        return list(self.STANDARD_ATOMS.keys())
    
    def __getitem__(self, name: str) -> Atom:
        return self.get(name)
