"""
TriX Compiler

Transforms high-level specifications into verified atomic circuits.

Pipeline:
    Spec -> Decompose -> Verify -> Compose -> Emit

Usage:
    from trix.compiler import TriXCompiler
    
    compiler = TriXCompiler()
    circuit = compiler.compile("8bit_adder")
    circuit.verify()
    circuit.emit("adder.trix")
"""

from .atoms import AtomLibrary, Atom
from .spec import CircuitSpec, AtomSpec
from .decompose import Decomposer
from .verify import Verifier
from .compose import Composer
from .emit import Emitter
from .compiler import TriXCompiler

__all__ = [
    "TriXCompiler",
    "AtomLibrary",
    "Atom",
    "CircuitSpec",
    "AtomSpec",
    "Decomposer",
    "Verifier", 
    "Composer",
    "Emitter",
]
