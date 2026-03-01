"""
Verification Engine

Ensures all atoms achieve 100% exactness on their bounded domains.
This is where we earn the right to claim "verified computation."
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .atoms import AtomLibrary, Atom, VerificationResult, AtomStatus
from .decompose import DecompositionResult


@dataclass
class AtomVerificationReport:
    """Report for a single atom's verification"""
    name: str
    atom_type: str
    status: AtomStatus
    accuracy: float
    training_time: float
    verification_time: float
    total_cases: int
    
    @property
    def passed(self) -> bool:
        return self.status == AtomStatus.VERIFIED


@dataclass
class CircuitVerificationReport:
    """Complete verification report for a circuit"""
    circuit_name: str
    total_atoms: int
    verified_atoms: int
    failed_atoms: int
    atom_reports: List[AtomVerificationReport]
    total_time: float
    
    @property
    def all_verified(self) -> bool:
        return self.failed_atoms == 0
    
    def summary(self) -> str:
        status = "VERIFIED" if self.all_verified else "FAILED"
        lines = [
            f"Circuit Verification: {self.circuit_name} [{status}]",
            f"  Atoms: {self.verified_atoms}/{self.total_atoms} verified",
            f"  Time: {self.total_time:.2f}s",
        ]
        
        if not self.all_verified:
            lines.append("  Failed atoms:")
            for r in self.atom_reports:
                if not r.passed:
                    lines.append(f"    - {r.name} ({r.atom_type}): {r.accuracy*100:.1f}%")
        
        return "\n".join(lines)


class Verifier:
    """
    Verifies that all atoms in a circuit achieve exactness.
    
    The verification process:
    1. For each unique atom type in the circuit
    2. Generate exhaustive test cases (all input combinations)
    3. Train the atom network
    4. Verify 100% accuracy
    5. Report results
    
    Only circuits where ALL atoms are verified can be compiled.
    """
    
    def __init__(self, library: AtomLibrary, 
                 training_epochs: int = 5000,
                 parallel: bool = True,
                 max_workers: int = 4):
        self.library = library
        self.training_epochs = training_epochs
        self.parallel = parallel
        self.max_workers = max_workers
    
    def verify_circuit(self, decomposition: DecompositionResult) -> CircuitVerificationReport:
        """Verify all atoms needed for a circuit"""
        start_time = time.time()
        
        # Get unique atom types needed
        atom_types = decomposition.atom_types_needed
        
        # Verify each atom type
        if self.parallel and len(atom_types) > 1:
            reports = self._verify_parallel(atom_types)
        else:
            reports = self._verify_sequential(atom_types)
        
        # Count results
        verified = sum(1 for r in reports if r.passed)
        failed = len(reports) - verified
        
        total_time = time.time() - start_time
        
        return CircuitVerificationReport(
            circuit_name=decomposition.spec.name,
            total_atoms=len(atom_types),
            verified_atoms=verified,
            failed_atoms=failed,
            atom_reports=reports,
            total_time=total_time,
        )
    
    def _verify_sequential(self, atom_types: List[str]) -> List[AtomVerificationReport]:
        """Verify atoms one at a time"""
        reports = []
        for atom_type in atom_types:
            report = self._verify_atom(atom_type)
            reports.append(report)
        return reports
    
    def _verify_parallel(self, atom_types: List[str]) -> List[AtomVerificationReport]:
        """Verify atoms in parallel"""
        reports = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._verify_atom, at): at 
                for at in atom_types
            }
            for future in as_completed(futures):
                reports.append(future.result())
        return reports
    
    def _verify_atom(self, atom_type: str) -> AtomVerificationReport:
        """Train and verify a single atom type"""
        train_start = time.time()
        
        # Get or train the atom
        atom = self.library.get(atom_type, train_if_missing=True)
        
        train_time = time.time() - train_start
        
        # Verify
        verify_start = time.time()
        if atom.verification is None:
            atom.verify()
        verify_time = time.time() - verify_start
        
        return AtomVerificationReport(
            name=atom_type,
            atom_type=atom_type,
            status=atom.status,
            accuracy=atom.verification.accuracy if atom.verification else 0.0,
            training_time=train_time,
            verification_time=verify_time,
            total_cases=atom.verification.total_cases if atom.verification else 0,
        )
    
    def verify_composition(self, decomposition: DecompositionResult,
                          test_cases: List[Tuple[Dict, Dict]]) -> Tuple[bool, List[str]]:
        """
        Verify the composed circuit against test cases.
        
        Args:
            decomposition: The decomposed circuit
            test_cases: List of (inputs, expected_outputs) dicts
        
        Returns:
            (all_passed, list of failure descriptions)
        """
        failures = []
        
        for i, (inputs, expected) in enumerate(test_cases):
            # Simulate circuit execution
            actual = self._simulate_circuit(decomposition, inputs)
            
            # Compare
            for wire, exp_val in expected.items():
                act_val = actual.get(wire)
                if act_val != exp_val:
                    failures.append(
                        f"Case {i}: {wire} expected {exp_val}, got {act_val}"
                    )
        
        return len(failures) == 0, failures
    
    def _simulate_circuit(self, decomposition: DecompositionResult,
                         inputs: Dict[str, int]) -> Dict[str, int]:
        """
        Simulate circuit execution using verified atoms.
        
        Follows the topological order and computes each wire value.
        """
        wire_values = dict(inputs)
        
        # Execute atoms in order
        for atom_name in decomposition.dependency_order:
            # Find the atom instance
            atom_inst = None
            for a in decomposition.spec.atoms:
                if a.name == atom_name:
                    atom_inst = a
                    break
            
            if atom_inst is None:
                continue
            
            # Get the atom from library
            atom = self.library.get(atom_inst.atom_type)
            
            # Gather inputs
            input_bits = []
            for inp_wire in atom_inst.inputs:
                if inp_wire in wire_values:
                    input_bits.append(float(wire_values[inp_wire]))
                else:
                    input_bits.append(0.0)  # Default
            
            # Execute atom
            input_tensor = torch.tensor([input_bits])
            output = atom(input_tensor)
            
            # Store outputs
            for j, out_wire in enumerate(atom_inst.outputs):
                wire_values[out_wire] = int(output[0, j].item())
        
        return wire_values


class ExhaustiveVerifier:
    """
    Exhaustively verifies a complete circuit.
    
    For small circuits, tests ALL possible input combinations.
    For larger circuits, uses statistical sampling with high confidence.
    """
    
    def __init__(self, library: AtomLibrary, max_exhaustive_bits: int = 16):
        self.library = library
        self.verifier = Verifier(library)
        self.max_exhaustive_bits = max_exhaustive_bits
    
    def verify_exhaustive(self, decomposition: DecompositionResult,
                         oracle: callable) -> Tuple[bool, float, List[str]]:
        """
        Exhaustively verify circuit against an oracle function.
        
        Args:
            decomposition: The circuit to verify
            oracle: Function that computes correct output given input dict
        
        Returns:
            (passed, accuracy, failure_descriptions)
        """
        spec = decomposition.spec
        
        # Count input bits
        total_input_bits = sum(w.width for w in spec.inputs)
        
        if total_input_bits > self.max_exhaustive_bits:
            return self._verify_statistical(decomposition, oracle, total_input_bits)
        
        # Generate all input combinations
        n_cases = 2 ** total_input_bits
        failures = []
        
        for i in range(n_cases):
            # Build input dict
            inputs = {}
            bit_idx = 0
            for wire in spec.inputs:
                if wire.width == 1:
                    inputs[wire.name] = (i >> bit_idx) & 1
                    bit_idx += 1
                else:
                    for j in range(wire.width):
                        inputs[wire.bit(j)] = (i >> bit_idx) & 1
                        bit_idx += 1
            
            # Get expected from oracle
            expected = oracle(inputs)
            
            # Simulate circuit
            actual = self.verifier._simulate_circuit(decomposition, inputs)
            
            # Compare outputs
            for wire in spec.outputs:
                if wire.width == 1:
                    exp = expected.get(wire.name, 0)
                    act = actual.get(wire.name, -1)
                    if exp != act:
                        failures.append(f"Input {inputs}: {wire.name} expected {exp}, got {act}")
                else:
                    for j in range(wire.width):
                        wname = wire.bit(j)
                        exp = expected.get(wname, 0)
                        act = actual.get(wname, -1)
                        if exp != act:
                            failures.append(f"Input {inputs}: {wname} expected {exp}, got {act}")
        
        accuracy = (n_cases - len(failures)) / n_cases
        return len(failures) == 0, accuracy, failures[:10]  # First 10 failures
    
    def _verify_statistical(self, decomposition: DecompositionResult,
                           oracle: callable, 
                           total_bits: int,
                           n_samples: int = 10000) -> Tuple[bool, float, List[str]]:
        """Statistical verification for large circuits"""
        import random
        
        spec = decomposition.spec
        failures = []
        
        for _ in range(n_samples):
            # Random input
            rand_val = random.randint(0, 2**total_bits - 1)
            
            inputs = {}
            bit_idx = 0
            for wire in spec.inputs:
                if wire.width == 1:
                    inputs[wire.name] = (rand_val >> bit_idx) & 1
                    bit_idx += 1
                else:
                    for j in range(wire.width):
                        inputs[wire.bit(j)] = (rand_val >> bit_idx) & 1
                        bit_idx += 1
            
            expected = oracle(inputs)
            actual = self.verifier._simulate_circuit(decomposition, inputs)
            
            # Compare
            for wire in spec.outputs:
                if wire.width == 1:
                    if expected.get(wire.name) != actual.get(wire.name):
                        failures.append(f"Sample: {wire.name} mismatch")
                else:
                    for j in range(wire.width):
                        wname = wire.bit(j)
                        if expected.get(wname) != actual.get(wname):
                            failures.append(f"Sample: {wname} mismatch")
        
        accuracy = (n_samples - len(failures)) / n_samples
        passed = len(failures) == 0
        
        return passed, accuracy, failures[:10]
