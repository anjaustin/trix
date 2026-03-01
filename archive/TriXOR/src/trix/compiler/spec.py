"""
Circuit Specification Language

Defines the structure of circuits to be compiled.
Specs can be created programmatically or loaded from files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import json
from pathlib import Path


class WireType(Enum):
    """Type of wire connection"""
    INPUT = "input"      # External input to circuit
    OUTPUT = "output"    # External output from circuit
    INTERNAL = "internal"  # Internal connection between atoms


@dataclass
class Wire:
    """A named wire in the circuit"""
    name: str
    wire_type: WireType
    width: int = 1  # Number of bits
    
    def __hash__(self):
        return hash(self.name)
    
    def bit(self, index: int) -> str:
        """Get name of specific bit"""
        if self.width == 1:
            return self.name
        return f"{self.name}[{index}]"


@dataclass
class AtomInstance:
    """An instance of an atom in the circuit"""
    name: str           # Instance name (unique in circuit)
    atom_type: str      # Type of atom (e.g., "SUM", "CARRY")
    inputs: List[str]   # Wire names for inputs
    outputs: List[str]  # Wire names for outputs
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class AtomSpec:
    """Specification for a custom atom"""
    name: str
    input_bits: int
    output_bits: int
    truth_table: Dict[Tuple, Tuple]  # {input_tuple: output_tuple}
    description: str = ""


@dataclass
class CircuitSpec:
    """
    Complete specification of a circuit.
    
    A circuit consists of:
    - Input wires (external inputs)
    - Output wires (external outputs)
    - Internal wires (connections between atoms)
    - Atom instances (the computational elements)
    
    The topology is implicit in the wire connections.
    """
    name: str
    description: str = ""
    
    inputs: List[Wire] = field(default_factory=list)
    outputs: List[Wire] = field(default_factory=list)
    internals: List[Wire] = field(default_factory=list)
    atoms: List[AtomInstance] = field(default_factory=list)
    
    # Optional: custom atom definitions
    custom_atoms: List[AtomSpec] = field(default_factory=list)
    
    def add_input(self, name: str, width: int = 1) -> Wire:
        """Add an input wire"""
        wire = Wire(name, WireType.INPUT, width)
        self.inputs.append(wire)
        return wire
    
    def add_output(self, name: str, width: int = 1) -> Wire:
        """Add an output wire"""
        wire = Wire(name, WireType.OUTPUT, width)
        self.outputs.append(wire)
        return wire
    
    def add_internal(self, name: str, width: int = 1) -> Wire:
        """Add an internal wire"""
        wire = Wire(name, WireType.INTERNAL, width)
        self.internals.append(wire)
        return wire
    
    def add_atom(self, name: str, atom_type: str, 
                 inputs: List[str], outputs: List[str]) -> AtomInstance:
        """Add an atom instance"""
        instance = AtomInstance(name, atom_type, inputs, outputs)
        self.atoms.append(instance)
        return instance
    
    def all_wires(self) -> Dict[str, Wire]:
        """Get all wires by name"""
        wires = {}
        for w in self.inputs + self.outputs + self.internals:
            if w.width == 1:
                wires[w.name] = w
            else:
                for i in range(w.width):
                    wires[w.bit(i)] = Wire(w.bit(i), w.wire_type, 1)
        return wires
    
    def validate(self) -> List[str]:
        """Validate the circuit specification"""
        errors = []
        wires = self.all_wires()
        
        # Check that all atom inputs/outputs reference valid wires
        for atom in self.atoms:
            for inp in atom.inputs:
                if inp not in wires:
                    errors.append(f"Atom {atom.name}: unknown input wire '{inp}'")
            for out in atom.outputs:
                if out not in wires:
                    errors.append(f"Atom {atom.name}: unknown output wire '{out}'")
        
        # Check for duplicate atom names
        names = [a.name for a in self.atoms]
        if len(names) != len(set(names)):
            errors.append("Duplicate atom instance names")
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "inputs": [{"name": w.name, "width": w.width} for w in self.inputs],
            "outputs": [{"name": w.name, "width": w.width} for w in self.outputs],
            "internals": [{"name": w.name, "width": w.width} for w in self.internals],
            "atoms": [
                {
                    "name": a.name,
                    "type": a.atom_type,
                    "inputs": a.inputs,
                    "outputs": a.outputs,
                }
                for a in self.atoms
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitSpec":
        """Load from dictionary"""
        spec = cls(data["name"], data.get("description", ""))
        
        for w in data.get("inputs", []):
            spec.add_input(w["name"], w.get("width", 1))
        for w in data.get("outputs", []):
            spec.add_output(w["name"], w.get("width", 1))
        for w in data.get("internals", []):
            spec.add_internal(w["name"], w.get("width", 1))
        
        for a in data.get("atoms", []):
            spec.add_atom(a["name"], a["type"], a["inputs"], a["outputs"])
        
        return spec
    
    def save(self, path: Path):
        """Save specification to file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "CircuitSpec":
        """Load specification from file"""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Circuit Templates
# =============================================================================

def full_adder_spec() -> CircuitSpec:
    """Specification for a 1-bit full adder"""
    spec = CircuitSpec("full_adder", "1-bit full adder")
    
    # Inputs
    spec.add_input("A")
    spec.add_input("B")
    spec.add_input("Cin")
    
    # Outputs
    spec.add_output("Sum")
    spec.add_output("Cout")
    
    # Atoms
    spec.add_atom("sum_atom", "SUM", ["A", "B", "Cin"], ["Sum"])
    spec.add_atom("carry_atom", "CARRY", ["A", "B", "Cin"], ["Cout"])
    
    return spec


def ripple_adder_spec(bits: int = 8) -> CircuitSpec:
    """Specification for an N-bit ripple carry adder"""
    spec = CircuitSpec(f"adder_{bits}bit", f"{bits}-bit ripple carry adder")
    
    # Inputs
    spec.add_input("A", bits)
    spec.add_input("B", bits)
    spec.add_input("Cin")
    
    # Outputs
    spec.add_output("Sum", bits)
    spec.add_output("Cout")
    
    # Internal carry chain
    if bits > 1:
        spec.add_internal("C", bits - 1)
    
    # Create adder chain
    for i in range(bits):
        a_wire = f"A[{i}]"
        b_wire = f"B[{i}]"
        sum_wire = f"Sum[{i}]"
        
        # Determine carry input
        if i == 0:
            cin_wire = "Cin"
        else:
            cin_wire = f"C[{i-1}]"
        
        # Determine carry output
        if i == bits - 1:
            cout_wire = "Cout"
        else:
            cout_wire = f"C[{i}]"
        
        # Add atoms
        spec.add_atom(f"sum_{i}", "SUM", [a_wire, b_wire, cin_wire], [sum_wire])
        spec.add_atom(f"carry_{i}", "CARRY", [a_wire, b_wire, cin_wire], [cout_wire])
    
    return spec


def alu_spec() -> CircuitSpec:
    """Specification for a simple 1-bit ALU"""
    spec = CircuitSpec("alu_1bit", "1-bit ALU with ADD, AND, OR, XOR")
    
    # Inputs
    spec.add_input("A")
    spec.add_input("B") 
    spec.add_input("Cin")
    spec.add_input("Op", 2)  # 2-bit opcode: 00=ADD, 01=AND, 10=OR, 11=XOR
    
    # Outputs
    spec.add_output("Result")
    spec.add_output("Cout")
    
    # Internal results
    spec.add_internal("add_result")
    spec.add_internal("and_result")
    spec.add_internal("or_result")
    spec.add_internal("xor_result")
    spec.add_internal("mux_01")
    spec.add_internal("mux_23")
    
    # Compute all operations
    spec.add_atom("adder_sum", "SUM", ["A", "B", "Cin"], ["add_result"])
    spec.add_atom("adder_carry", "CARRY", ["A", "B", "Cin"], ["Cout"])
    spec.add_atom("and_op", "AND", ["A", "B"], ["and_result"])
    spec.add_atom("or_op", "OR", ["A", "B"], ["or_result"])
    spec.add_atom("xor_op", "XOR", ["A", "B"], ["xor_result"])
    
    # Mux tree to select result based on opcode
    spec.add_atom("mux_0", "MUX", ["Op[0]", "add_result", "and_result"], ["mux_01"])
    spec.add_atom("mux_1", "MUX", ["Op[0]", "or_result", "xor_result"], ["mux_23"])
    spec.add_atom("mux_final", "MUX", ["Op[1]", "mux_01", "mux_23"], ["Result"])
    
    return spec


# Registry of built-in circuit templates
CIRCUIT_TEMPLATES = {
    "full_adder": full_adder_spec,
    "adder_8bit": lambda: ripple_adder_spec(8),
    "adder_16bit": lambda: ripple_adder_spec(16),
    "adder_32bit": lambda: ripple_adder_spec(32),
    "alu_1bit": alu_spec,
}


def get_template(name: str) -> CircuitSpec:
    """Get a circuit template by name"""
    if name not in CIRCUIT_TEMPLATES:
        available = ", ".join(CIRCUIT_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    return CIRCUIT_TEMPLATES[name]()
