"""
Decomposition Engine

Transforms high-level circuit specifications into atomic operations.
This is where "intent" becomes "structure."
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from .spec import CircuitSpec, AtomInstance, Wire, WireType
from .atoms import AtomLibrary, Atom


@dataclass
class DecompositionResult:
    """Result of decomposing a circuit spec"""
    spec: CircuitSpec
    atom_types_needed: List[str]
    dependency_order: List[str]  # Topological order of atom instances
    wire_dependencies: Dict[str, List[str]]  # wire -> atoms that produce it
    
    def summary(self) -> str:
        lines = [
            f"Decomposition: {self.spec.name}",
            f"  Atoms needed: {', '.join(self.atom_types_needed)}",
            f"  Total instances: {len(self.spec.atoms)}",
            f"  Execution order: {len(self.dependency_order)} stages",
        ]
        return "\n".join(lines)


class Decomposer:
    """
    Decomposes circuit specifications into atomic operations.
    
    The decomposer:
    1. Analyzes the circuit topology
    2. Determines which atoms are needed
    3. Computes execution order (topological sort)
    4. Validates that all atoms are available
    """
    
    def __init__(self, library: AtomLibrary):
        self.library = library
    
    def decompose(self, spec: CircuitSpec) -> DecompositionResult:
        """Decompose a circuit specification"""
        
        # Validate spec
        errors = spec.validate()
        if errors:
            raise ValueError(f"Invalid spec: {errors}")
        
        # Collect unique atom types
        atom_types = list(set(a.atom_type for a in spec.atoms))
        
        # Build wire dependency graph
        wire_producers = self._build_wire_producers(spec)
        
        # Topological sort for execution order
        exec_order = self._topological_sort(spec, wire_producers)
        
        return DecompositionResult(
            spec=spec,
            atom_types_needed=atom_types,
            dependency_order=exec_order,
            wire_dependencies=wire_producers,
        )
    
    def _build_wire_producers(self, spec: CircuitSpec) -> Dict[str, List[str]]:
        """Map each wire to the atoms that produce it"""
        producers = {}
        
        # Input wires are produced by the environment
        for w in spec.inputs:
            if w.width == 1:
                producers[w.name] = ["__INPUT__"]
            else:
                for i in range(w.width):
                    producers[w.bit(i)] = ["__INPUT__"]
        
        # Internal and output wires are produced by atoms
        for atom in spec.atoms:
            for out_wire in atom.outputs:
                if out_wire not in producers:
                    producers[out_wire] = []
                producers[out_wire].append(atom.name)
        
        return producers
    
    def _topological_sort(self, spec: CircuitSpec, 
                          wire_producers: Dict[str, List[str]]) -> List[str]:
        """
        Compute execution order via topological sort.
        
        An atom can execute when all its input wires are available.
        """
        # Build atom dependency graph
        atom_deps = {}  # atom_name -> set of atom names it depends on
        
        for atom in spec.atoms:
            deps = set()
            for inp_wire in atom.inputs:
                if inp_wire in wire_producers:
                    for producer in wire_producers[inp_wire]:
                        if producer != "__INPUT__" and producer != atom.name:
                            deps.add(producer)
            atom_deps[atom.name] = deps
        
        # Kahn's algorithm for topological sort
        in_degree = {name: len(deps) for name, deps in atom_deps.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            # Process atoms with no remaining dependencies
            current = queue.pop(0)
            result.append(current)
            
            # Update dependencies
            for name, deps in atom_deps.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        if len(result) != len(spec.atoms):
            raise ValueError("Circular dependency detected in circuit")
        
        return result
    
    def validate_atoms_available(self, result: DecompositionResult) -> List[str]:
        """Check that all required atoms are in the library"""
        missing = []
        available = self.library.list_atoms()
        
        for atom_type in result.atom_types_needed:
            if atom_type not in available:
                missing.append(atom_type)
        
        return missing


# =============================================================================
# High-Level Decomposition Patterns
# =============================================================================

class PatternDecomposer:
    """
    Decomposes high-level patterns into circuit specs.
    
    This is the "macro expansion" layer - turning concepts like
    "8-bit adder" into explicit atom instances and wiring.
    """
    
    def __init__(self, library: AtomLibrary):
        self.library = library
        self.decomposer = Decomposer(library)
    
    def decompose_nbit_adder(self, n: int) -> Tuple[CircuitSpec, DecompositionResult]:
        """Decompose an N-bit ripple carry adder"""
        from .spec import ripple_adder_spec
        
        spec = ripple_adder_spec(n)
        result = self.decomposer.decompose(spec)
        
        return spec, result
    
    def decompose_nbit_comparator(self, n: int) -> Tuple[CircuitSpec, DecompositionResult]:
        """Decompose an N-bit comparator (A < B, A == B, A > B)"""
        spec = CircuitSpec(f"compare_{n}bit", f"{n}-bit comparator")
        
        # Inputs
        spec.add_input("A", n)
        spec.add_input("B", n)
        
        # Outputs
        spec.add_output("LT")  # A < B
        spec.add_output("EQ")  # A == B
        spec.add_output("GT")  # A > B
        
        # For each bit, compute XOR (difference) and propagate
        spec.add_internal("diff", n)
        spec.add_internal("eq_chain", n)
        
        # Bit-wise difference
        for i in range(n):
            spec.add_atom(
                f"xor_{i}", "XOR",
                [f"A[{i}]", f"B[{i}]"],
                [f"diff[{i}]"]
            )
        
        # This is simplified - full comparator needs more logic
        # For now, just show the decomposition pattern
        
        result = self.decomposer.decompose(spec)
        return spec, result
    
    def decompose_multiplexer(self, n_inputs: int) -> Tuple[CircuitSpec, DecompositionResult]:
        """Decompose an N:1 multiplexer"""
        import math
        sel_bits = math.ceil(math.log2(n_inputs))
        
        spec = CircuitSpec(f"mux_{n_inputs}to1", f"{n_inputs}:1 multiplexer")
        
        # Inputs
        for i in range(n_inputs):
            spec.add_input(f"D{i}")
        spec.add_input("Sel", sel_bits)
        
        # Output
        spec.add_output("Y")
        
        # Build mux tree
        # For simplicity, assume n_inputs is power of 2
        current_level = [f"D{i}" for i in range(n_inputs)]
        level = 0
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                out_wire = f"mux_L{level}_{i//2}"
                if len(current_level) == 2:
                    out_wire = "Y"
                else:
                    spec.add_internal(out_wire)
                
                spec.add_atom(
                    f"mux_L{level}_{i//2}", "MUX",
                    [f"Sel[{level}]", current_level[i], current_level[i+1]],
                    [out_wire]
                )
                next_level.append(out_wire)
            
            current_level = next_level
            level += 1
        
        result = self.decomposer.decompose(spec)
        return spec, result
