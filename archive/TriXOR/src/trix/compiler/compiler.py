"""
TriX Compiler

The main compiler that orchestrates the full pipeline:
    Spec -> Decompose -> Verify -> Compose -> Emit

Usage:
    compiler = TriXCompiler()
    result = compiler.compile("adder_8bit")
    result.circuit.execute({"A[0]": 1, "B[0]": 0, ...})
"""

from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
from pathlib import Path

from .atoms import AtomLibrary
from .atoms_fp4 import FP4AtomLibrary
from .spec import CircuitSpec, get_template, CIRCUIT_TEMPLATES
from .decompose import Decomposer, DecompositionResult, PatternDecomposer
from .verify import Verifier, CircuitVerificationReport, ExhaustiveVerifier
from .compose import Composer, Topology, CircuitExecutor
from .emit import Emitter, TrixConfig, TrixLoader, CompiledCircuit, FP4Emitter, FP4Loader


@dataclass
class CompilationResult:
    """Result of compiling a circuit"""
    
    # Pipeline stages
    spec: CircuitSpec
    decomposition: DecompositionResult
    verification: CircuitVerificationReport
    topology: Topology
    config: Optional[TrixConfig] = None
    
    # Execution
    executor: Optional[CircuitExecutor] = None
    
    @property
    def success(self) -> bool:
        return self.verification.all_verified
    
    @property
    def circuit(self) -> CircuitExecutor:
        """Get the executable circuit"""
        if self.executor is None:
            raise RuntimeError("Circuit not available - compilation may have failed")
        return self.executor
    
    def summary(self) -> str:
        """Summary of compilation result"""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Compilation: {self.spec.name} [{status}]",
            "",
            self.decomposition.summary(),
            "",
            self.verification.summary(),
            "",
            self.topology.summary(),
        ]
        return "\n".join(lines)
    
    def execute(self, inputs: Dict[str, int]) -> Dict[str, int]:
        """Execute the compiled circuit"""
        return self.circuit.execute(inputs)


class TriXCompiler:
    """
    The TriX Compiler.
    
    Transforms high-level circuit specifications into verified,
    executable neural circuits.
    
    Example:
        compiler = TriXCompiler()
        
        # Compile a template
        result = compiler.compile("adder_8bit")
        
        # Or compile a custom spec
        spec = CircuitSpec("my_circuit")
        spec.add_input("A")
        spec.add_output("Y")
        spec.add_atom("inv", "NOT", ["A"], ["Y"])
        result = compiler.compile(spec)
        
        # Execute
        output = result.execute({"A[0]": 1, "B[0]": 0, ...})
    """
    
    def __init__(self, 
                 library: Optional[AtomLibrary] = None,
                 cache_dir: Optional[Path] = None,
                 verbose: bool = True,
                 use_fp4: bool = False):
        """
        Initialize the compiler.
        
        Args:
            library: Atom library to use (created if not provided)
            cache_dir: Directory to cache trained atoms
            verbose: Whether to print progress
            use_fp4: Use FP4 threshold circuit atoms (exact by construction)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".trix_cache")
        self.verbose = verbose
        self.use_fp4 = use_fp4
        
        # Initialize atom library
        if use_fp4:
            self.library = FP4AtomLibrary()
            self.fp4_library = self.library  # Keep reference for FP4-specific ops
        else:
            self.library = library or AtomLibrary(self.cache_dir / "atoms")
            self.fp4_library = None
        
        # Initialize pipeline components
        self.decomposer = Decomposer(self.library)
        self.pattern_decomposer = PatternDecomposer(self.library)
        self.verifier = Verifier(self.library)
        self.composer = Composer(self.library)
        
        # Emitter depends on FP4 mode
        if use_fp4:
            self.emitter = FP4Emitter(self.library)
            self.loader = FP4Loader()
        else:
            self.emitter = Emitter(self.library)
            self.loader = TrixLoader(self.library)
    
    def compile(self, 
                spec_or_name: Union[CircuitSpec, str],
                output_dir: Optional[Path] = None,
                verify_exhaustive: bool = False,
                oracle: Optional[Callable] = None) -> CompilationResult:
        """
        Compile a circuit specification.
        
        Args:
            spec_or_name: CircuitSpec object or name of template
            output_dir: Directory to emit compiled files (optional)
            verify_exhaustive: Whether to exhaustively verify the circuit
            oracle: Oracle function for exhaustive verification
        
        Returns:
            CompilationResult with all pipeline outputs
        """
        # Get spec
        if isinstance(spec_or_name, str):
            if spec_or_name in CIRCUIT_TEMPLATES:
                spec = get_template(spec_or_name)
                self._log(f"Using template: {spec_or_name}")
            else:
                raise ValueError(f"Unknown template: {spec_or_name}")
        else:
            spec = spec_or_name
        
        self._log(f"Compiling: {spec.name}")
        
        # Stage 1: Decompose
        self._log("Stage 1: Decomposition")
        decomposition = self.decomposer.decompose(spec)
        self._log(f"  -> {len(spec.atoms)} atom instances")
        self._log(f"  -> {len(decomposition.atom_types_needed)} unique atom types")
        
        # Stage 2: Verify atoms
        self._log("Stage 2: Verification")
        verification = self.verifier.verify_circuit(decomposition)
        self._log(f"  -> {verification.verified_atoms}/{verification.total_atoms} atoms verified")
        
        if not verification.all_verified:
            self._log("  -> VERIFICATION FAILED")
            return CompilationResult(
                spec=spec,
                decomposition=decomposition,
                verification=verification,
                topology=None,
            )
        
        # Stage 3: Compose
        self._log("Stage 3: Composition")
        topology = self.composer.compose(decomposition)
        self._log(f"  -> {len(topology.tiles)} tiles allocated")
        self._log(f"  -> {len(topology.routes)} routes created")
        
        # Create executor
        executor = CircuitExecutor(topology, self.library)
        
        # Optional: Exhaustive verification
        if verify_exhaustive and oracle:
            self._log("Stage 3b: Exhaustive Verification")
            ev = ExhaustiveVerifier(self.library)
            passed, accuracy, failures = ev.verify_exhaustive(decomposition, oracle)
            self._log(f"  -> {accuracy*100:.2f}% accuracy")
            if not passed:
                self._log(f"  -> FAILURES: {failures[:3]}")
        
        # Stage 4: Emit (optional)
        config = None
        if output_dir:
            self._log("Stage 4: Emission")
            output_dir = Path(output_dir)
            config = self.emitter.emit(
                topology, 
                output_dir,
                verified=verification.all_verified,
                accuracy=verification.verified_atoms / verification.total_atoms,
            )
            self._log(f"  -> Emitted to {output_dir}")
        
        self._log("Compilation complete!")
        
        return CompilationResult(
            spec=spec,
            decomposition=decomposition,
            verification=verification,
            topology=topology,
            config=config,
            executor=executor,
        )
    
    def compile_and_test(self,
                        spec_or_name: Union[CircuitSpec, str],
                        test_cases: list,
                        output_dir: Optional[Path] = None) -> CompilationResult:
        """
        Compile and test against provided test cases.
        
        Args:
            spec_or_name: Circuit specification
            test_cases: List of (inputs_dict, expected_outputs_dict)
            output_dir: Optional output directory
        
        Returns:
            CompilationResult
        """
        result = self.compile(spec_or_name, output_dir)
        
        if not result.success:
            return result
        
        self._log("Running test cases...")
        passed = 0
        failed = 0
        
        for inputs, expected in test_cases:
            actual = result.execute(inputs)
            
            # Compare
            match = True
            for key, exp_val in expected.items():
                if actual.get(key) != exp_val:
                    match = False
                    break
            
            if match:
                passed += 1
            else:
                failed += 1
                if failed <= 3:  # Show first 3 failures
                    self._log(f"  FAIL: input={inputs}, expected={expected}, got={actual}")
        
        self._log(f"Tests: {passed}/{passed+failed} passed")
        
        return result
    
    def load(self, config_path: Path) -> CompiledCircuit:
        """Load a previously compiled circuit"""
        return self.loader.load(config_path)
    
    def list_templates(self) -> list:
        """List available circuit templates"""
        return list(CIRCUIT_TEMPLATES.keys())
    
    def list_atoms(self) -> list:
        """List available atom types"""
        return self.library.list_atoms()
    
    def _log(self, msg: str):
        """Print log message if verbose"""
        if self.verbose:
            print(msg)


# =============================================================================
# Convenience Functions
# =============================================================================

def compile_circuit(spec_or_name: Union[CircuitSpec, str],
                   output_dir: Optional[Path] = None,
                   verbose: bool = True) -> CompilationResult:
    """
    Convenience function to compile a circuit.
    
    Example:
        result = compile_circuit("adder_8bit")
        output = result.execute({"A[0]": 1, "B[0]": 0, ...})
    """
    compiler = TriXCompiler(verbose=verbose)
    return compiler.compile(spec_or_name, output_dir)


def quick_test():
    """Quick test of the compiler"""
    print("=" * 60)
    print("TriX Compiler Quick Test")
    print("=" * 60)
    
    compiler = TriXCompiler()
    
    # Compile a full adder
    result = compiler.compile("full_adder")
    print()
    print(result.summary())
    
    # Test it
    print()
    print("Testing Full Adder:")
    test_cases = [
        ({"A": 0, "B": 0, "Cin": 0}, {"Sum": 0, "Cout": 0}),
        ({"A": 0, "B": 1, "Cin": 0}, {"Sum": 1, "Cout": 0}),
        ({"A": 1, "B": 0, "Cin": 0}, {"Sum": 1, "Cout": 0}),
        ({"A": 1, "B": 1, "Cin": 0}, {"Sum": 0, "Cout": 1}),
        ({"A": 0, "B": 0, "Cin": 1}, {"Sum": 1, "Cout": 0}),
        ({"A": 1, "B": 1, "Cin": 1}, {"Sum": 1, "Cout": 1}),
    ]
    
    all_pass = True
    for inputs, expected in test_cases:
        actual = result.execute(inputs)
        status = "OK" if actual == expected else "FAIL"
        if actual != expected:
            all_pass = False
        print(f"  {inputs} -> {actual} (expected {expected}) [{status}]")
    
    print()
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    
    return result


if __name__ == "__main__":
    quick_test()
