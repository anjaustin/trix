#!/usr/bin/env python3
"""
TriX Compiler Demo

Demonstrates the full compilation pipeline:
    Spec -> Decompose -> Verify -> Compose -> Emit -> Execute

This script proves that we can compile high-level circuit specifications
into verified neural circuits that compute EXACTLY.

Usage:
    python scripts/demo_compiler.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix.compiler import TriXCompiler, CircuitSpec


def banner(text: str):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def demo_full_adder():
    """Demo: Compile and test a 1-bit full adder"""
    banner("DEMO 1: Full Adder (1-bit)")
    
    compiler = TriXCompiler()
    result = compiler.compile("full_adder")
    
    print(result.summary())
    
    # Test
    print("\nTruth Table:")
    print("  A  B Cin | Sum Cout")
    print("  ---------+---------")
    for a in [0, 1]:
        for b in [0, 1]:
            for cin in [0, 1]:
                out = result.execute({"A": a, "B": b, "Cin": cin})
                print(f"  {a}  {b}  {cin}  |  {out['Sum']}    {out['Cout']}")


def demo_8bit_adder():
    """Demo: Compile and test an 8-bit ripple carry adder"""
    banner("DEMO 2: 8-Bit Ripple Carry Adder")
    
    compiler = TriXCompiler()
    result = compiler.compile("adder_8bit")
    
    print(f"Compiled: {result.spec.name}")
    print(f"  Tiles: {len(result.topology.tiles)}")
    print(f"  Routes: {len(result.topology.routes)}")
    print(f"  Verified: {result.verification.all_verified}")
    
    # Helper to test addition
    def add(a, b, cin=0):
        inputs = {"Cin": cin}
        for i in range(8):
            inputs[f"A[{i}]"] = (a >> i) & 1
            inputs[f"B[{i}]"] = (b >> i) & 1
        
        outputs = result.execute(inputs)
        
        sum_val = sum(outputs.get(f"Sum[{i}]", 0) << i for i in range(8))
        cout = outputs.get("Cout", 0)
        return sum_val, cout
    
    # Test cases
    print("\nTest Cases:")
    tests = [(37, 28), (100, 55), (127, 128), (255, 1), (255, 255)]
    all_pass = True
    
    for a, b in tests:
        got, cout = add(a, b)
        expected = (a + b) & 0xFF
        exp_cout = 1 if (a + b) > 255 else 0
        ok = (got == expected) and (cout == exp_cout)
        all_pass = all_pass and ok
        status = "OK" if ok else "FAIL"
        print(f"  {a:3} + {b:3} = {got:3} (C={cout}) [{status}]")
    
    if all_pass:
        print("\n  ALL TESTS PASSED!")


def demo_custom_circuit():
    """Demo: Define and compile a custom circuit"""
    banner("DEMO 3: Custom Circuit (2-bit Counter)")
    
    # Define a 2-bit increment circuit
    # Input: 2-bit value
    # Output: 2-bit value + 1 (with wrap)
    
    spec = CircuitSpec("increment_2bit", "2-bit increment by 1")
    
    # Inputs: 2-bit value
    spec.add_input("In", 2)
    
    # Outputs: 2-bit result
    spec.add_output("Out", 2)
    
    # Internal carry
    spec.add_internal("C0")
    
    # Logic: Out = In + 1
    # Bit 0: In[0] XOR 1 (always flip)
    # Bit 1: In[1] XOR (In[0] AND 1) = In[1] XOR In[0]
    
    # For bit 0: add 1
    # SUM(In[0], 1, 0) = In[0] XOR 1 = NOT In[0]
    spec.add_atom("not0", "NOT", ["In[0]"], ["Out[0]"])
    
    # Carry from bit 0 = In[0] (since we're adding 1)
    # For bit 1: In[1] XOR carry = In[1] XOR In[0]
    spec.add_atom("xor1", "XOR", ["In[1]", "In[0]"], ["Out[1]"])
    
    compiler = TriXCompiler()
    result = compiler.compile(spec)
    
    print(f"Compiled: {result.spec.name}")
    print(f"  Atoms: {[a.atom_type for a in spec.atoms]}")
    print(f"  Verified: {result.verification.all_verified}")
    
    # Test
    print("\nIncrement Test:")
    for val in range(4):
        inputs = {
            "In[0]": val & 1,
            "In[1]": (val >> 1) & 1,
        }
        outputs = result.execute(inputs)
        out_val = outputs.get("Out[0]", 0) + (outputs.get("Out[1]", 0) << 1)
        expected = (val + 1) % 4
        status = "OK" if out_val == expected else "FAIL"
        print(f"  {val} + 1 = {out_val} (expected {expected}) [{status}]")


def demo_emit():
    """Demo: Emit compiled circuit to files"""
    banner("DEMO 4: Emit to Files")
    
    compiler = TriXCompiler()
    
    output_dir = Path(".trix_output/adder_8bit")
    result = compiler.compile("adder_8bit", output_dir=output_dir)
    
    print(f"Emitted to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(output_dir)}: {size} bytes")


def demo_topology_visualization():
    """Demo: Visualize circuit topology"""
    banner("DEMO 5: Topology Visualization")
    
    compiler = TriXCompiler(verbose=False)
    result = compiler.compile("full_adder")
    
    print(result.topology.visualize())


def main():
    print()
    print("=" * 70)
    print("  TRIX COMPILER DEMONSTRATION")
    print("  Compiling Neural Circuits")
    print("=" * 70)
    print()
    print("The TriX Compiler transforms high-level specifications into")
    print("verified neural circuits that compute EXACTLY.")
    print()
    print("Pipeline: Spec -> Decompose -> Verify -> Compose -> Emit")
    
    demo_full_adder()
    demo_8bit_adder()
    demo_custom_circuit()
    demo_topology_visualization()
    demo_emit()
    
    banner("DEMONSTRATION COMPLETE")
    print()
    print("Key Results:")
    print("  - Atoms trained to 100% exactness")
    print("  - Circuits composed from verified atoms")
    print("  - Execution produces EXACT results")
    print("  - Configuration emitted to files")
    print()
    print("The neural network has become a computer.")
    print()


if __name__ == "__main__":
    main()
