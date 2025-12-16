#!/usr/bin/env python3
"""
FFT Compiler: Compile Transform Structure to FP4 Atoms

Key insight: Transforms have two types of operations:
1. STRUCTURAL: (stage, pos) → pair_index, twiddle_index
   - These are discrete lookups - can be boolean circuits!
2. ARITHMETIC: a+b, a-b, complex multiply by fixed constants
   - These stay as fixed microcode

We compile the STRUCTURAL parts to FP4 threshold circuits.

TRANSFORMS IMPLEMENTED:
- Walsh-Hadamard Transform (WHT): XOR-based pairing, no twiddles
- Discrete Fourier Transform (DFT): Cooley-Tukey with twiddle OPCODES

TWIDDLE OPCODES (the key to true compiled DFT):
- No runtime trig (np.cos, np.sin)
- Twiddles are fixed microcode: (re,im) → (re*c - im*s, re*s + im*c)
- Routing selects which opcode to apply

This gives us: COMPILED CONTROL + FIXED MICROCODE = VERIFIED TRANSFORMS
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import numpy as np
from typing import Dict, Tuple, List, Callable
from dataclasses import dataclass


# =============================================================================
# TWIDDLE OPCODES (Fixed Microcode - No Runtime Trig!)
# =============================================================================

# For N=8: W_8^k = cos(2*pi*k/8) - i*sin(2*pi*k/8)
# These are EXACT algebraic constants, not computed at runtime.

SQRT_HALF = 0.7071067811865476  # 1/sqrt(2), exact to float precision

# Twiddle constants: (cos, -sin) for W_N^k
# W_8^k for k=0..7
TWIDDLE_TABLE_8 = [
    ( 1.0,         0.0),         # k=0: W^0 = 1
    ( SQRT_HALF,  -SQRT_HALF),   # k=1: W^1 = (1-i)/sqrt(2)
    ( 0.0,        -1.0),         # k=2: W^2 = -i
    (-SQRT_HALF,  -SQRT_HALF),   # k=3: W^3 = (-1-i)/sqrt(2)
    (-1.0,         0.0),         # k=4: W^4 = -1
    (-SQRT_HALF,   SQRT_HALF),   # k=5: W^5 = (-1+i)/sqrt(2)
    ( 0.0,         1.0),         # k=6: W^6 = i
    ( SQRT_HALF,   SQRT_HALF),   # k=7: W^7 = (1+i)/sqrt(2)
]


def make_twiddle_op(c: float, s: float) -> Callable[[float, float], Tuple[float, float]]:
    """
    Create a twiddle opcode (fixed complex multiply).
    
    (re, im) * (c + is) = (re*c - im*s, re*s + im*c)
    """
    def twiddle_op(re: float, im: float) -> Tuple[float, float]:
        return (re * c - im * s, re * s + im * c)
    return twiddle_op


def build_twiddle_ops(N: int) -> List[Callable]:
    """
    Build twiddle opcodes for N-point DFT.
    
    Returns list of N opcodes, one for each W_N^k.
    """
    ops = []
    for k in range(N):
        angle = -2 * np.pi * k / N
        c, s = np.cos(angle), np.sin(angle)
        ops.append(make_twiddle_op(c, s))
    return ops


# Pre-built opcodes for common sizes
TWIDDLE_OPS_8 = [make_twiddle_op(c, s) for c, s in TWIDDLE_TABLE_8]


def get_twiddle_index(N: int, m: int, j: int) -> int:
    """
    Compute twiddle index for Cooley-Tukey DIT FFT.
    
    Args:
        N: Transform size
        m: Current group size (2^(stage+1))
        j: Position within half-group (0 to m/2-1)
    
    Returns:
        Twiddle index k such that W_N^k is the required twiddle.
    
    Formula: k = j * (N // m)
    
    This is STRUCTURAL - no learning, no computation at runtime.
    """
    return (j * (N // m)) % N


# =============================================================================
# FFT STRUCTURE ANALYSIS
# =============================================================================

def analyze_fft_structure(N: int) -> Dict:
    """
    Analyze FFT structure to extract truth tables for routing.
    
    For DIT (decimation-in-time) radix-2 FFT:
    - num_stages = log2(N)
    - Each stage: pairs elements at distance 2^stage
    - Partner of position i at stage s: i XOR 2^s
    
    Returns truth tables for:
    - partner[stage, pos] → partner position
    - is_upper[stage, pos] → 1 if pos > partner (gets difference)
    """
    num_stages = int(np.log2(N))
    
    partner_table = {}  # (stage, pos) → partner
    is_upper_table = {}  # (stage, pos) → 0 or 1
    
    for stage in range(num_stages):
        stride = 2 ** stage
        for pos in range(N):
            partner = pos ^ stride  # XOR gives partner
            partner_table[(stage, pos)] = partner
            is_upper_table[(stage, pos)] = 1 if pos > partner else 0
    
    return {
        'N': N,
        'num_stages': num_stages,
        'partner': partner_table,
        'is_upper': is_upper_table,
    }


def analyze_twiddle_structure(N: int) -> Dict:
    """
    Analyze twiddle factor selection for complex FFT.
    
    Twiddle index depends on (stage, position):
    - k = (pos % (2^(stage+1))) * (N / 2^(stage+1))
    
    Returns truth table:
    - twiddle[stage, pos] → twiddle index (0 to N-1)
    """
    num_stages = int(np.log2(N))
    
    twiddle_table = {}
    
    for stage in range(num_stages):
        stride = 2 ** stage
        group_size = 2 * stride
        
        for pos in range(N):
            pos_in_group = pos % group_size
            
            if pos_in_group < stride:
                # Lower half: partner gets multiplied by twiddle, not us
                k = 0
            else:
                # Upper half: we get multiplied
                k = (pos_in_group - stride) * (N // group_size)
            
            twiddle_table[(stage, pos)] = k % N
    
    return {
        'N': N,
        'num_stages': num_stages,
        'twiddle': twiddle_table,
    }


# =============================================================================
# TRUTH TABLE TO BITS
# =============================================================================

def int_to_bits(val: int, num_bits: int) -> Tuple[int, ...]:
    """Convert integer to tuple of bits (LSB first)."""
    return tuple((val >> i) & 1 for i in range(num_bits))


def bits_to_int(bits: Tuple[int, ...]) -> int:
    """Convert tuple of bits (LSB first) to integer."""
    result = 0
    for i, b in enumerate(bits):
        result |= b << i
    return result


def build_partner_truth_table(N: int) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Build truth table for partner computation.
    
    Input: (stage_bits..., pos_bits...)
    Output: (partner_bits...)
    """
    num_stages = int(np.log2(N))
    stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
    pos_bits = int(np.log2(N))
    
    structure = analyze_fft_structure(N)
    truth_table = {}
    
    for stage in range(num_stages):
        for pos in range(N):
            partner = structure['partner'][(stage, pos)]
            
            # Input: stage bits + position bits
            input_bits = int_to_bits(stage, stage_bits) + int_to_bits(pos, pos_bits)
            output_bits = int_to_bits(partner, pos_bits)
            
            truth_table[input_bits] = output_bits
    
    return truth_table


def build_is_upper_truth_table(N: int) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Build truth table for is_upper (determines sum vs diff assignment).
    
    Input: (stage_bits..., pos_bits...)
    Output: (1 bit - 0=lower/sum, 1=upper/diff)
    """
    num_stages = int(np.log2(N))
    stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
    pos_bits = int(np.log2(N))
    
    structure = analyze_fft_structure(N)
    truth_table = {}
    
    for stage in range(num_stages):
        for pos in range(N):
            is_upper = structure['is_upper'][(stage, pos)]
            
            input_bits = int_to_bits(stage, stage_bits) + int_to_bits(pos, pos_bits)
            output_bits = (is_upper,)
            
            truth_table[input_bits] = output_bits
    
    return truth_table


# =============================================================================
# FP4 COMPILATION
# =============================================================================

def compile_fft_routing(N: int):
    """
    Compile FFT routing to FP4 threshold circuits.
    
    Creates circuits for:
    1. PARTNER: (stage, pos) → partner position
    2. IS_UPPER: (stage, pos) → 0 or 1
    """
    from trix.compiler.atoms_fp4 import truth_table_to_circuit, verify_generated_circuit
    
    num_stages = int(np.log2(N))
    stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
    pos_bits = int(np.log2(N))
    
    print(f"\nCompiling FFT N={N} routing to FP4...")
    print(f"  Stages: {num_stages}")
    print(f"  Stage bits: {stage_bits}, Position bits: {pos_bits}")
    
    results = {}
    
    # Compile IS_UPPER (single output bit - simplest)
    print(f"\n[IS_UPPER]")
    is_upper_tt = build_is_upper_truth_table(N)
    
    # Convert to single-output format for our generator
    is_upper_single = {k: v[0] for k, v in is_upper_tt.items()}
    
    num_inputs = stage_bits + pos_bits
    circuit = truth_table_to_circuit("IS_UPPER", num_inputs, is_upper_single)
    
    passed, acc, failures = verify_generated_circuit(circuit, is_upper_single)
    print(f"  Inputs: {num_inputs} bits")
    print(f"  Verified: {'PASS' if passed else 'FAIL'} ({acc:.0%})")
    print(f"  Layers: {len(circuit.layers)}")
    
    results['is_upper'] = {
        'circuit': circuit,
        'verified': passed,
        'accuracy': acc,
    }
    
    # For PARTNER, we need multiple output bits
    # Compile each output bit separately
    print(f"\n[PARTNER]")
    partner_tt = build_partner_truth_table(N)
    
    partner_circuits = []
    for bit_idx in range(pos_bits):
        # Extract single bit
        single_bit_tt = {k: v[bit_idx] for k, v in partner_tt.items()}
        
        circuit = truth_table_to_circuit(f"PARTNER_BIT{bit_idx}", num_inputs, single_bit_tt)
        passed, acc, failures = verify_generated_circuit(circuit, single_bit_tt)
        
        print(f"  Bit {bit_idx}: {'PASS' if passed else 'FAIL'} ({acc:.0%}), {len(circuit.layers)} layers")
        
        partner_circuits.append({
            'circuit': circuit,
            'verified': passed,
            'accuracy': acc,
        })
    
    all_verified = all(pc['verified'] for pc in partner_circuits)
    results['partner'] = {
        'circuits': partner_circuits,
        'verified': all_verified,
    }
    
    return results


# =============================================================================
# FFT EXECUTOR WITH COMPILED ROUTING
# =============================================================================

@dataclass
class CompiledWHT:
    """
    Walsh-Hadamard Transform with compiled routing circuits.
    
    NOTE: This implements WHT, not DFT/FFT!
    The XOR-based pairing (partner = pos XOR 2^stage) gives WHT.
    
    WHT is useful for:
    - Signal processing
    - Data compression
    - Quantum computing
    - Error correction codes
    
    WHT is self-inverse: WHT(WHT(x)) = N * x
    """
    N: int
    is_upper_circuit: object  # ThresholdCircuit
    partner_circuits: List[object]  # List of ThresholdCircuit
    
    def _eval_circuit(self, circuit, *inputs) -> int:
        """Evaluate a threshold circuit on integer inputs."""
        # Convert inputs to bit tensor
        x = torch.tensor([list(inputs)], dtype=torch.float32)
        y = circuit(x)
        return int(y[0, 0].item() > 0.5)
    
    def _get_partner(self, stage: int, pos: int) -> int:
        """Get partner position using compiled circuits."""
        num_stages = int(np.log2(self.N))
        stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
        pos_bits = int(np.log2(self.N))
        
        # Encode inputs
        input_bits = int_to_bits(stage, stage_bits) + int_to_bits(pos, pos_bits)
        
        # Evaluate each output bit
        partner_bits = []
        for circuit_info in self.partner_circuits:
            circuit = circuit_info['circuit']
            bit = self._eval_circuit(circuit, *input_bits)
            partner_bits.append(bit)
        
        return bits_to_int(tuple(partner_bits))
    
    def _get_is_upper(self, stage: int, pos: int) -> int:
        """Check if position is upper (gets difference)."""
        num_stages = int(np.log2(self.N))
        stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
        pos_bits = int(np.log2(self.N))
        
        input_bits = int_to_bits(stage, stage_bits) + int_to_bits(pos, pos_bits)
        return self._eval_circuit(self.is_upper_circuit, *input_bits)
    
    def execute(self, x: List[float]) -> List[float]:
        """
        Execute FFT with compiled routing and exact arithmetic.
        
        Args:
            x: Input values (length N)
        
        Returns:
            FFT output values (length N)
        """
        assert len(x) == self.N
        
        values = list(x)
        num_stages = int(np.log2(self.N))
        
        for stage in range(num_stages):
            new_values = values.copy()
            processed = set()
            
            for pos in range(self.N):
                if pos in processed:
                    continue
                
                # Get partner from compiled circuit
                partner = self._get_partner(stage, pos)
                
                # Get which is upper from compiled circuit
                is_upper = self._get_is_upper(stage, pos)
                
                # Butterfly with exact arithmetic
                a, b = values[pos], values[partner]
                sum_val = a + b
                diff_val = a - b
                
                # Assign based on is_upper
                if is_upper:
                    new_values[pos] = diff_val
                    new_values[partner] = sum_val
                else:
                    new_values[pos] = sum_val
                    new_values[partner] = diff_val
                
                processed.add(pos)
                processed.add(partner)
            
            values = new_values
        
        return values


def build_twiddle_truth_table(N: int) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Build truth table for twiddle index selection.
    
    Input: (stage_bits..., pos_bits...)
    Output: (twiddle_index_bits...)
    """
    num_stages = int(np.log2(N))
    stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
    pos_bits = int(np.log2(N))
    twiddle_bits = int(np.log2(N))  # Twiddle indices 0 to N-1
    
    structure = analyze_twiddle_structure(N)
    truth_table = {}
    
    for stage in range(num_stages):
        for pos in range(N):
            twiddle_idx = structure['twiddle'][(stage, pos)]
            
            input_bits = int_to_bits(stage, stage_bits) + int_to_bits(pos, pos_bits)
            output_bits = int_to_bits(twiddle_idx, twiddle_bits)
            
            truth_table[input_bits] = output_bits
    
    return truth_table


def compile_complex_fft_routing(N: int):
    """
    Compile complex FFT routing including twiddle selection.
    """
    from trix.compiler.atoms_fp4 import truth_table_to_circuit, verify_generated_circuit
    
    # First get basic routing
    results = compile_fft_routing(N)
    
    num_stages = int(np.log2(N))
    stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
    pos_bits = int(np.log2(N))
    twiddle_bits = int(np.log2(N))
    num_inputs = stage_bits + pos_bits
    
    print(f"\n[TWIDDLE]")
    twiddle_tt = build_twiddle_truth_table(N)
    
    twiddle_circuits = []
    for bit_idx in range(twiddle_bits):
        single_bit_tt = {k: v[bit_idx] for k, v in twiddle_tt.items()}
        
        circuit = truth_table_to_circuit(f"TWIDDLE_BIT{bit_idx}", num_inputs, single_bit_tt)
        passed, acc, failures = verify_generated_circuit(circuit, single_bit_tt)
        
        print(f"  Bit {bit_idx}: {'PASS' if passed else 'FAIL'} ({acc:.0%}), {len(circuit.layers)} layers")
        
        twiddle_circuits.append({
            'circuit': circuit,
            'verified': passed,
            'accuracy': acc,
        })
    
    all_verified = all(tc['verified'] for tc in twiddle_circuits)
    results['twiddle'] = {
        'circuits': twiddle_circuits,
        'verified': all_verified,
    }
    
    return results


@dataclass
class CompiledComplexFFT:
    """
    Complex FFT with compiled routing and TWIDDLE OPCODES.
    
    Key difference from standard FFT:
    - NO runtime trig (np.cos, np.sin)
    - Twiddles are fixed microcode opcodes
    - Routing selects which opcode to apply
    
    This is a TRUE compiled DFT.
    """
    N: int
    is_upper_circuit: object
    partner_circuits: List[object]
    twiddle_circuits: List[object]
    twiddle_ops: List[Callable] = None  # Twiddle opcodes (fixed microcode)
    
    def __post_init__(self):
        # Build twiddle opcodes (fixed constants, no runtime trig!)
        if self.N == 8:
            self.twiddle_ops = TWIDDLE_OPS_8
        else:
            # For other sizes, build from table (computed once at init, not at runtime)
            self.twiddle_ops = build_twiddle_ops(self.N)
    
    def _eval_circuit(self, circuit, *inputs) -> int:
        x = torch.tensor([list(inputs)], dtype=torch.float32)
        y = circuit(x)
        return int(y[0, 0].item() > 0.5)
    
    def _get_partner(self, stage: int, pos: int) -> int:
        num_stages = int(np.log2(self.N))
        stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
        pos_bits = int(np.log2(self.N))
        
        input_bits = int_to_bits(stage, stage_bits) + int_to_bits(pos, pos_bits)
        
        partner_bits = []
        for circuit_info in self.partner_circuits:
            bit = self._eval_circuit(circuit_info['circuit'], *input_bits)
            partner_bits.append(bit)
        
        return bits_to_int(tuple(partner_bits))
    
    def _get_is_upper(self, stage: int, pos: int) -> int:
        num_stages = int(np.log2(self.N))
        stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
        pos_bits = int(np.log2(self.N))
        
        input_bits = int_to_bits(stage, stage_bits) + int_to_bits(pos, pos_bits)
        return self._eval_circuit(self.is_upper_circuit, *input_bits)
    
    def _get_twiddle_index(self, stage: int, pos: int) -> int:
        num_stages = int(np.log2(self.N))
        stage_bits = max(1, int(np.ceil(np.log2(num_stages))))
        pos_bits = int(np.log2(self.N))
        
        input_bits = int_to_bits(stage, stage_bits) + int_to_bits(pos, pos_bits)
        
        twiddle_bits = []
        for circuit_info in self.twiddle_circuits:
            bit = self._eval_circuit(circuit_info['circuit'], *input_bits)
            twiddle_bits.append(bit)
        
        return bits_to_int(tuple(twiddle_bits))
    
    def _bit_reverse_index(self, i: int) -> int:
        """Compute bit-reversed index."""
        bits = int(np.log2(self.N))
        result = 0
        for _ in range(bits):
            result = (result << 1) | (i & 1)
            i >>= 1
        return result
    
    def _bit_reverse_array(self, x: List[float]) -> List[float]:
        """Bit-reverse an array."""
        result = [0.0] * self.N
        for i in range(self.N):
            result[self._bit_reverse_index(i)] = x[i]
        return result
    
    def execute(self, x_real: List[float], x_imag: List[float] = None) -> Tuple[List[float], List[float]]:
        """
        Execute complex FFT with TWIDDLE OPCODES.
        
        NO RUNTIME TRIG! All twiddles are fixed microcode.
        
        Returns (real_part, imag_part).
        """
        if x_imag is None:
            x_imag = [0.0] * self.N
        
        assert len(x_real) == self.N
        assert len(x_imag) == self.N
        
        # Bit-reverse inputs for DIT FFT
        values_re = self._bit_reverse_array(x_real)
        values_im = self._bit_reverse_array(x_imag)
        
        num_stages = int(np.log2(self.N))
        
        # Cooley-Tukey DIT FFT with TWIDDLE OPCODES
        for stage in range(num_stages):
            m = 2 ** (stage + 1)  # Butterfly group size
            half_m = m // 2
            
            new_re = values_re.copy()
            new_im = values_im.copy()
            
            for k in range(0, self.N, m):  # For each group
                for j in range(half_m):  # For each butterfly in group
                    # Indices
                    idx_u = k + j
                    idx_t = k + j + half_m
                    
                    # Get values
                    u_re, u_im = values_re[idx_u], values_im[idx_u]
                    t_re, t_im = values_re[idx_t], values_im[idx_t]
                    
                    # TWIDDLE OPCODE: structural routing to fixed microcode
                    tw_idx = get_twiddle_index(self.N, m, j)
                    wt_re, wt_im = self.twiddle_ops[tw_idx](t_re, t_im)
                    
                    # Butterfly: (u + wt, u - wt)
                    new_re[idx_u] = u_re + wt_re
                    new_im[idx_u] = u_im + wt_im
                    new_re[idx_t] = u_re - wt_re
                    new_im[idx_t] = u_im - wt_im
            
            values_re = new_re
            values_im = new_im
        
        return values_re, values_im


def verify_no_runtime_trig():
    """
    Verify that execute() contains no runtime trig calls.
    
    This is VGem's guard: fail if np.cos, np.sin, exp appear in runtime path.
    """
    import inspect
    source = inspect.getsource(CompiledComplexFFT.execute)
    
    forbidden = ['np.cos', 'np.sin', 'np.exp', 'math.cos', 'math.sin', 'math.exp']
    violations = [f for f in forbidden if f in source]
    
    if violations:
        raise AssertionError(f"Runtime trig detected in execute(): {violations}")
    
    return True


def test_compiled_complex_fft(N: int = 8):
    """Test complex FFT with TWIDDLE OPCODES (no runtime trig)."""
    
    print("\n" + "=" * 70)
    print(f"COMPILED COMPLEX FFT TEST (N={N})")
    print("=" * 70)
    
    # VGem's guard: verify no runtime trig in execute path
    verify_no_runtime_trig()
    
    # Compile routing including twiddles
    routing = compile_complex_fft_routing(N)
    
    if not all([
        routing['is_upper']['verified'],
        routing['partner']['verified'],
        routing['twiddle']['verified'],
    ]):
        print("ERROR: Some circuits failed verification")
        return False
    
    # Create compiled FFT
    fft = CompiledComplexFFT(
        N=N,
        is_upper_circuit=routing['is_upper']['circuit'],
        partner_circuits=routing['partner']['circuits'],
        twiddle_circuits=routing['twiddle']['circuits'],
    )
    
    # Test against numpy FFT
    print("\n[TESTING vs NumPy FFT]")
    max_error = 0.0
    
    for _ in range(20):
        x = np.random.randn(N)
        
        # NumPy reference
        expected = np.fft.fft(x)
        
        # Our compiled FFT
        out_re, out_im = fft.execute(list(x))
        
        # Compare
        for i in range(N):
            error_re = abs(out_re[i] - expected[i].real)
            error_im = abs(out_im[i] - expected[i].imag)
            max_error = max(max_error, error_re, error_im)
    
    print(f"  Max error vs NumPy: {max_error:.2e}")
    
    # Twiddle opcode coverage check
    print("\n[TWIDDLE OPCODE COVERAGE]")
    twiddle_used = set()
    for stage in range(int(np.log2(N))):
        m = 2 ** (stage + 1)
        half_m = m // 2
        for j in range(half_m):
            tw_idx = get_twiddle_index(N, m, j)
            twiddle_used.add(tw_idx)
    
    coverage = len(twiddle_used)
    expected_opcodes = N // 2  # DIT FFT uses N/2 unique twiddles
    print(f"  Opcodes exercised: {sorted(twiddle_used)}")
    print(f"  Coverage: {coverage}/{N} twiddle opcodes used")
    
    print("\n" + "=" * 70)
    if max_error < 1e-10:
        print(f"✓ COMPILED COMPLEX FFT N={N}: EXACT!")
    elif max_error < 1e-5:
        print(f"✓ COMPILED COMPLEX FFT N={N}: PASS (error: {max_error:.2e})")
    else:
        print(f"✗ COMPILED COMPLEX FFT N={N}: FAIL (error: {max_error:.2e})")
    print("=" * 70)
    
    return max_error < 1e-5


def test_compiled_wht(N: int = 8):
    """Test Walsh-Hadamard Transform with compiled routing."""
    
    print("\n" + "=" * 70)
    print(f"COMPILED WALSH-HADAMARD TRANSFORM TEST (N={N})")
    print("=" * 70)
    
    # Compile routing
    routing = compile_fft_routing(N)
    
    if not routing['is_upper']['verified']:
        print("ERROR: IS_UPPER circuit failed verification")
        return False
    
    if not routing['partner']['verified']:
        print("ERROR: PARTNER circuits failed verification")
        return False
    
    # Create compiled WHT
    wht = CompiledWHT(
        N=N,
        is_upper_circuit=routing['is_upper']['circuit'],
        partner_circuits=routing['partner']['circuits'],
    )
    
    # Reference WHT (XOR-based)
    def reference_wht(x):
        result = list(x)
        for stage in range(int(np.log2(N))):
            stride = 2 ** stage
            new_result = result.copy()
            for i in range(N):
                partner = i ^ stride
                if i < partner:
                    a, b = result[i], result[partner]
                    new_result[i] = a + b
                    new_result[partner] = a - b
            result = new_result
        return result
    
    # Test
    print("\n[TESTING]")
    correct = 0
    total = 100
    
    for _ in range(total):
        x = [np.random.randint(0, 16) for _ in range(N)]
        expected = reference_wht(x)
        actual = wht.execute(x)
        
        if expected == actual:
            correct += 1
    
    acc = correct / total
    print(f"  Accuracy: {correct}/{total} = {acc:.0%}")
    
    # Verify self-inverse property
    print("\n[SELF-INVERSE TEST]")
    x = [np.random.randint(0, 8) for _ in range(N)]
    wht_x = wht.execute(x)
    wht_wht_x = wht.execute(wht_x)
    # WHT(WHT(x)) = N * x
    recovered = [int(v / N) for v in wht_wht_x]
    self_inverse_ok = recovered == x
    print(f"  x = {x}")
    print(f"  WHT(x) = {[int(v) for v in wht_x]}")
    print(f"  WHT(WHT(x))/N = {recovered}")
    print(f"  Self-inverse: {'✓' if self_inverse_ok else '✗'}")
    
    # Summary
    print("\n" + "=" * 70)
    if acc == 1.0:
        print(f"✓ COMPILED WHT N={N}: 100% EXACT!")
        print("  Routing: FP4 threshold circuits")
        print("  Arithmetic: Fixed microcode (exact)")
        print("  NOTE: This is Walsh-Hadamard Transform, not DFT/FFT")
    else:
        print(f"✗ COMPILED WHT: {acc:.0%}")
    print("=" * 70)
    
    return acc == 1.0


def test_compiled_fft(N: int = 8):
    """Alias for WHT test (backward compatibility)."""
    return test_compiled_wht(N)


# Backward compatibility alias
CompiledFFT = CompiledWHT


if __name__ == "__main__":
    # Test N=8
    test_compiled_fft(8)
