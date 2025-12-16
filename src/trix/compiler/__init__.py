"""
TriX Compiler

Transforms high-level specifications into verified atomic circuits.

Pipeline:
    Spec -> Decompose -> Verify -> Compose -> Emit

Usage:
    from trix.compiler import TriXCompiler
    
    compiler = TriXCompiler()
    circuit = compiler.compile("adder_8bit")
    result.execute(inputs)

FP4 Atoms:
    from trix.compiler import FP4AtomLibrary
    
    library = FP4AtomLibrary()
    library.verify_all()  # 100% exact
"""

from .atoms import AtomLibrary, Atom
from .atoms_fp4 import (
    FP4AtomLibrary,
    ThresholdCircuit,
    ThresholdLayer,
    FP4AtomModule,
    truth_table_to_circuit,
    step,
)
from .spec import CircuitSpec, AtomSpec
from .decompose import Decomposer
from .verify import Verifier
from .compose import Composer
from .emit import (
    Emitter, TrixConfig, TrixLoader, CompiledCircuit,
    FP4Emitter, FP4Loader, FP4CompiledCircuit,
)
from .fp4_pack import (
    pack_circuit, unpack_circuit,
    save_circuit as save_fp4_circuit,
    load_circuit as load_fp4_circuit,
    measure_sizes as measure_fp4_sizes,
    verify_roundtrip as verify_fp4_roundtrip,
)
from .compiler import TriXCompiler

__all__ = [
    # Main
    "TriXCompiler",
    
    # Atoms (float, trained)
    "AtomLibrary",
    "Atom",
    
    # Atoms (FP4, constructed)
    "FP4AtomLibrary",
    "ThresholdCircuit",
    "ThresholdLayer",
    "FP4AtomModule",
    "truth_table_to_circuit",
    "step",
    
    # Pipeline stages
    "CircuitSpec",
    "AtomSpec",
    "Decomposer",
    "Verifier", 
    "Composer",
    "Emitter",
    
    # Emission/Loading
    "TrixConfig",
    "TrixLoader",
    "CompiledCircuit",
    "FP4Emitter",
    "FP4Loader",
    "FP4CompiledCircuit",
    
    # FP4 Packing
    "pack_circuit",
    "unpack_circuit",
    "save_fp4_circuit",
    "load_fp4_circuit",
    "measure_fp4_sizes",
    "verify_fp4_roundtrip",
]
