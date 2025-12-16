#!/usr/bin/env python3
"""
Butterfly MatMul: Structured Matrix Multiplication via Routing + Blocks

Key insight: MatMul is FFT with different blocks.
- FFT: Route → Twiddle → Route → Twiddle → ...
- MatMul: Route → Block → Route → Block → ...

Both follow: Permutation × Block-Diagonal × Permutation × Block-Diagonal × ...

This is the Monarch matrix structure. We already built the engine for FFT.
Now we load a different cartridge.

References:
- Monarch Matrices (Dao et al.)
- Butterfly Matrices
- Our FFT implementation (fft_compiler.py)
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from itertools import product
import math


# =============================================================================
# TERNARY BLOCK OPCODES (Like Twiddle Opcodes, but for MatMul)
# =============================================================================

# All 2x2 ternary matrices: 3^4 = 81 total
# We enumerate the useful ones as "block opcodes"

def enumerate_ternary_2x2():
    """Enumerate all 2x2 matrices with entries in {-1, 0, 1}."""
    matrices = []
    for a, b, c, d in product([-1, 0, 1], repeat=4):
        matrices.append(np.array([[a, b], [c, d]]))
    return matrices


def is_invertible(M: np.ndarray) -> bool:
    """Check if matrix is invertible."""
    return abs(np.linalg.det(M)) > 0.5


def is_orthogonal(M: np.ndarray) -> bool:
    """Check if matrix is orthogonal (up to scaling)."""
    MtM = M.T @ M
    # Check if diagonal
    off_diag = abs(MtM[0, 1]) + abs(MtM[1, 0])
    return off_diag < 1e-10


def classify_ternary_2x2():
    """Classify 2x2 ternary matrices by properties."""
    all_matrices = enumerate_ternary_2x2()
    
    classified = {
        'identity_like': [],  # Acts like identity (diagonal, positive)
        'swap_like': [],      # Swaps elements
        'hadamard_like': [],  # Mixes with ±1 (like FFT butterfly)
        'projection': [],     # Rank 1 (projects)
        'zero': [],           # Zero matrix
        'invertible': [],     # Invertible but not above
        'other': [],          # Everything else
    }
    
    for M in all_matrices:
        det = np.linalg.det(M)
        trace = np.trace(M)
        
        if np.allclose(M, 0):
            classified['zero'].append(M)
        elif np.allclose(M, np.eye(2)) or np.allclose(M, -np.eye(2)):
            classified['identity_like'].append(M)
        elif np.allclose(M, [[0, 1], [1, 0]]) or np.allclose(M, [[0, -1], [-1, 0]]):
            classified['swap_like'].append(M)
        elif abs(det) > 0.5 and is_orthogonal(M):
            classified['hadamard_like'].append(M)
        elif abs(det) < 0.5 and not np.allclose(M, 0):
            classified['projection'].append(M)
        elif abs(det) > 0.5:
            classified['invertible'].append(M)
        else:
            classified['other'].append(M)
    
    return classified


# Named block opcodes (the most useful ones)
BLOCK_OPCODES = {
    # Identity family
    'I': np.array([[1, 0], [0, 1]]),
    '-I': np.array([[-1, 0], [0, -1]]),
    
    # Swap family
    'SWAP': np.array([[0, 1], [1, 0]]),
    '-SWAP': np.array([[0, -1], [-1, 0]]),
    
    # Hadamard family (mixing)
    'H+': np.array([[1, 1], [1, -1]]),    # Standard Hadamard (unnormalized)
    'H-': np.array([[1, -1], [1, 1]]),    # Reversed signs
    'H++': np.array([[1, 1], [-1, 1]]),   # Different arrangement
    'H--': np.array([[-1, 1], [1, 1]]),
    
    # Diagonal family
    'D+': np.array([[1, 0], [0, -1]]),    # Flip sign of second
    'D-': np.array([[-1, 0], [0, 1]]),    # Flip sign of first
    
    # Projection family (rank 1)
    'P1': np.array([[1, 0], [0, 0]]),     # Project to first
    'P2': np.array([[0, 0], [0, 1]]),     # Project to second
    'P+': np.array([[1, 1], [0, 0]]),     # Sum to first
    'P-': np.array([[1, -1], [0, 0]]),    # Diff to first
}


def make_block_op(M: np.ndarray) -> Callable[[float, float], Tuple[float, float]]:
    """
    Create a block opcode (like twiddle opcode, but 2x2 matrix).
    
    (a, b) → (M[0,0]*a + M[0,1]*b, M[1,0]*a + M[1,1]*b)
    """
    m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    
    def block_op(a: float, b: float) -> Tuple[float, float]:
        return (m00 * a + m01 * b, m10 * a + m11 * b)
    
    return block_op


# Pre-built block opcodes
BLOCK_OPS = {name: make_block_op(M) for name, M in BLOCK_OPCODES.items()}


# =============================================================================
# BUTTERFLY LAYER
# =============================================================================

@dataclass
class ButterflyLayer:
    """
    One stage of butterfly computation.
    
    Structure:
    - N inputs, N outputs
    - Pairs elements at distance `stride`
    - Applies 2x2 block to each pair
    
    This is identical to one FFT stage, but with arbitrary blocks
    instead of twiddle-scaled butterflies.
    """
    N: int
    stride: int
    blocks: List[np.ndarray]  # N/2 blocks of 2x2
    
    def __post_init__(self):
        assert self.N % 2 == 0
        assert len(self.blocks) == self.N // 2
        
        # Pre-compile block ops
        self.block_ops = [make_block_op(B) for B in self.blocks]
    
    def get_pairs(self) -> List[Tuple[int, int]]:
        """Get index pairs for this butterfly stage."""
        pairs = []
        for i in range(self.N):
            partner = i ^ self.stride  # XOR pairing, like FFT
            if i < partner:
                pairs.append((i, partner))
        return pairs
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply butterfly layer."""
        assert len(x) == self.N
        y = np.zeros_like(x)
        
        pairs = self.get_pairs()
        for pair_idx, (i, j) in enumerate(pairs):
            a, b = x[i], x[j]
            block_op = self.block_ops[pair_idx]
            y[i], y[j] = block_op(a, b)
        
        return y
    
    def as_matrix(self) -> np.ndarray:
        """Return the full N×N matrix representation."""
        M = np.zeros((self.N, self.N))
        
        pairs = self.get_pairs()
        for pair_idx, (i, j) in enumerate(pairs):
            block = self.blocks[pair_idx]
            M[i, i] = block[0, 0]
            M[i, j] = block[0, 1]
            M[j, i] = block[1, 0]
            M[j, j] = block[1, 1]
        
        return M


# =============================================================================
# BUTTERFLY NETWORK (Multiple Stages)
# =============================================================================

@dataclass  
class ButterflyNetwork:
    """
    Full butterfly network: multiple stages of butterfly layers.
    
    For N inputs with log2(N) stages, this can represent any
    matrix with O(N log N) parameters instead of O(N²).
    """
    N: int
    layers: List[ButterflyLayer]
    
    def __post_init__(self):
        self.num_stages = len(self.layers)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply all butterfly stages."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def as_matrix(self) -> np.ndarray:
        """Return the full N×N matrix representation."""
        M = np.eye(self.N)
        for layer in self.layers:
            M = layer.as_matrix() @ M
        return M
    
    @classmethod
    def from_blocks(cls, N: int, all_blocks: List[List[np.ndarray]]) -> 'ButterflyNetwork':
        """Create butterfly network from list of block lists per stage."""
        num_stages = int(np.log2(N))
        assert len(all_blocks) == num_stages
        
        layers = []
        for stage in range(num_stages):
            stride = 2 ** stage
            layer = ButterflyLayer(N, stride, all_blocks[stage])
            layers.append(layer)
        
        return cls(N, layers)


# =============================================================================
# CONSTRUCTORS FOR KNOWN MATRICES
# =============================================================================

def identity_butterfly(N: int) -> ButterflyNetwork:
    """
    Construct butterfly network for identity matrix.
    
    All blocks are identity: [[1,0],[0,1]]
    """
    num_stages = int(np.log2(N))
    all_blocks = []
    
    for stage in range(num_stages):
        blocks = [BLOCK_OPCODES['I'].copy() for _ in range(N // 2)]
        all_blocks.append(blocks)
    
    return ButterflyNetwork.from_blocks(N, all_blocks)


def hadamard_butterfly(N: int) -> ButterflyNetwork:
    """
    Construct butterfly network for Hadamard matrix.
    
    All blocks are Hadamard: [[1,1],[1,-1]]
    This should match our WHT implementation!
    """
    num_stages = int(np.log2(N))
    all_blocks = []
    
    for stage in range(num_stages):
        blocks = [BLOCK_OPCODES['H+'].copy() for _ in range(N // 2)]
        all_blocks.append(blocks)
    
    return ButterflyNetwork.from_blocks(N, all_blocks)


def dft_butterfly(N: int) -> ButterflyNetwork:
    """
    Construct butterfly network for DFT matrix.
    
    This uses twiddle factors, exactly like our FFT!
    Blocks are: [[1, W^k], [1, -W^k]] where W = e^{-2πi/N}
    
    Note: This produces complex results, so we'd need complex blocks.
    For now, we return the real-only structure.
    """
    num_stages = int(np.log2(N))
    all_blocks = []
    
    for stage in range(num_stages):
        m = 2 ** (stage + 1)
        blocks = []
        
        for j in range(N // 2):
            # Which twiddle for this butterfly?
            # In FFT, twiddle depends on position within group
            group_idx = j // (m // 2)
            pos_in_group = j % (m // 2)
            
            k = pos_in_group * (N // m)
            
            # Twiddle factor (complex)
            W_re = np.cos(-2 * np.pi * k / N)
            W_im = np.sin(-2 * np.pi * k / N)
            
            # For real-only, use just the real part structure
            # Full complex would need 4x4 blocks for 2 complex numbers
            block = np.array([[1, W_re], [1, -W_re]])
            blocks.append(block)
        
        all_blocks.append(blocks)
    
    return ButterflyNetwork.from_blocks(N, all_blocks)


def permutation_butterfly(N: int, perm: List[int]) -> ButterflyNetwork:
    """
    Construct butterfly network that approximates a permutation.
    
    Note: Not all permutations can be exactly represented!
    Only those decomposable into butterfly stages.
    """
    # This is complex - for now, return identity as placeholder
    return identity_butterfly(N)


# =============================================================================
# MONARCH LAYER (Generalization)
# =============================================================================

@dataclass
class MonarchLayer:
    """
    Monarch matrix layer: M = (B₁ ⊗ I_q) × P × (I_p ⊗ B₂)
    
    For N = p × q:
    - Reshape input to (q, p)
    - Apply B₁ (p×p) to each of q groups
    - Transpose (the permutation P)
    - Apply B₂ (q×q) to each of p groups
    - Reshape back to N
    
    Complexity: O(N × (p + q)) instead of O(N²)
    """
    N: int
    p: int  # Block size for first stage
    q: int  # Block size for second stage
    B1: List[np.ndarray]  # q blocks of p×p
    B2: List[np.ndarray]  # p blocks of q×q
    
    def __post_init__(self):
        assert self.N == self.p * self.q
        assert len(self.B1) == self.q
        assert len(self.B2) == self.p
        for B in self.B1:
            assert B.shape == (self.p, self.p)
        for B in self.B2:
            assert B.shape == (self.q, self.q)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Monarch layer."""
        assert len(x) == self.N
        
        # Reshape to (q, p)
        x = x.reshape(self.q, self.p)
        
        # Apply B1 to each row (p-dimensional)
        for i in range(self.q):
            x[i, :] = self.B1[i] @ x[i, :]
        
        # Transpose (the permutation!)
        x = x.T  # Now (p, q)
        
        # Apply B2 to each row (q-dimensional)
        for i in range(self.p):
            x[i, :] = self.B2[i] @ x[i, :]
        
        # Reshape back
        return x.reshape(self.N)
    
    def as_matrix(self) -> np.ndarray:
        """Return the full N×N matrix representation."""
        M = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            e_i = np.zeros(self.N)
            e_i[i] = 1.0
            M[:, i] = self.forward(e_i)
        
        return M


def identity_monarch(N: int, p: int = None) -> MonarchLayer:
    """
    Construct Monarch layer for identity matrix.
    
    For identity: M = I = P^T × P where P is the transpose permutation.
    So we need B1 and B2 to undo each other after transpose.
    
    Actually, for Monarch identity:
    - B1 = I (identity blocks)
    - B2 = I (identity blocks)
    - But transpose in the middle permutes!
    
    To get true identity, we need to account for the transpose.
    The Monarch structure inherently permutes, so identity requires
    B1 and B2 to "undo" the transpose.
    
    For simplicity: Monarch(I) requires special handling.
    Let's construct it properly.
    """
    if p is None:
        p = int(np.sqrt(N))
    q = N // p
    assert N == p * q
    
    # For identity, we need to cancel the transpose permutation
    # After reshape (q, p) -> transpose -> (p, q) -> reshape
    # Element at position i goes to position (i % p) * q + (i // p)
    
    # With identity blocks, we get a permutation matrix, not identity
    # To get true identity, B2 needs to "unscramble" what transpose did
    
    # Actually, let's just verify what the basic structure gives us
    B1 = [np.eye(p) for _ in range(q)]
    B2 = [np.eye(q) for _ in range(p)]
    
    return MonarchLayer(N, p, q, B1, B2)


def test_monarch_structure(N: int = 16):
    """Test what Monarch with identity blocks actually computes."""
    print(f"\n[MONARCH STRUCTURE N={N}]")
    
    p = int(np.sqrt(N))
    q = N // p
    print(f"  Block sizes: p={p}, q={q}")
    
    layer = identity_monarch(N, p)
    M = layer.as_matrix()
    
    # What did we get?
    print(f"  Result is permutation: {np.allclose(M @ M.T, np.eye(N))}")
    
    # What permutation?
    perm = np.argmax(M, axis=1)
    print(f"  Permutation: {perm.tolist()}")
    
    # This is the transpose/reshape permutation!
    expected_perm = [(i % p) * q + (i // p) for i in range(N)]
    print(f"  Expected:    {expected_perm}")
    print(f"  Match: {list(perm) == expected_perm}")
    
    return list(perm) == expected_perm


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_butterfly_network(net: ButterflyNetwork, target: np.ndarray, 
                             name: str = "Network") -> Tuple[bool, float]:
    """Verify butterfly network against target matrix."""
    M = net.as_matrix()
    error = np.max(np.abs(M - target))
    passed = error < 1e-10
    return passed, error


def test_identity(N: int = 8):
    """Test identity matrix construction."""
    print(f"\n[IDENTITY N={N}]")
    
    net = identity_butterfly(N)
    target = np.eye(N)
    passed, error = verify_butterfly_network(net, target, "Identity")
    
    print(f"  Max error: {error:.2e}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return passed


def test_hadamard(N: int = 8):
    """Test Hadamard matrix construction."""
    print(f"\n[HADAMARD N={N}]")
    
    from scipy.linalg import hadamard
    
    net = hadamard_butterfly(N)
    target = hadamard(N).astype(float)
    passed, error = verify_butterfly_network(net, target, "Hadamard")
    
    print(f"  Max error: {error:.2e}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    # Also test forward pass
    x = np.random.randn(N)
    y_net = net.forward(x)
    y_target = target @ x
    forward_error = np.max(np.abs(y_net - y_target))
    print(f"  Forward error: {forward_error:.2e}")
    
    return passed


def test_monarch_identity(N: int = 16):
    """Test Monarch identity construction."""
    print(f"\n[MONARCH IDENTITY N={N}]")
    
    p = int(np.sqrt(N))
    q = N // p
    print(f"  Block sizes: p={p}, q={q}")
    
    layer = identity_monarch(N, p)
    target = np.eye(N)
    
    M = layer.as_matrix()
    error = np.max(np.abs(M - target))
    passed = error < 1e-10
    
    print(f"  Max error: {error:.2e}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return passed


def test_random_approximation(N: int = 8, num_stages: int = None):
    """Test how well butterfly network approximates random matrix."""
    print(f"\n[RANDOM MATRIX APPROXIMATION N={N}]")
    
    if num_stages is None:
        num_stages = int(np.log2(N))
    
    # Random target matrix
    np.random.seed(42)
    target = np.random.randn(N, N)
    
    # Random butterfly network
    all_blocks = []
    for stage in range(num_stages):
        blocks = [np.random.randn(2, 2) for _ in range(N // 2)]
        all_blocks.append(blocks)
    
    net = ButterflyNetwork.from_blocks(N, all_blocks)
    M = net.as_matrix()
    
    # Frobenius norm error (relative)
    error = np.linalg.norm(M - target, 'fro') / np.linalg.norm(target, 'fro')
    
    print(f"  Relative Frobenius error: {error:.4f}")
    print(f"  (Random butterfly cannot match random dense matrix)")
    
    return error


def test_ternary_expressiveness():
    """Test what matrices can be exactly represented with ternary blocks."""
    print(f"\n[TERNARY EXPRESSIVENESS]")
    
    classified = classify_ternary_2x2()
    
    for category, matrices in classified.items():
        print(f"  {category}: {len(matrices)} matrices")
    
    # Show Hadamard-like (most useful for computation)
    print(f"\n  Hadamard-like matrices (orthogonal, ternary):")
    for M in classified['hadamard_like'][:5]:
        print(f"    {M.tolist()}")


# =============================================================================
# TRIX BUTTERFLY MLP
# =============================================================================

class TriXButterflyMLP:
    """
    Butterfly-structured MLP to replace dense FFN in transformers.
    
    Instead of: Linear(d, 4d) → GELU → Linear(4d, d)
    We use: ButterflyUp → GELU → ButterflyDown
    
    Complexity: O(d log d) instead of O(d²)
    """
    
    def __init__(self, d_model: int, expansion: int = 4, 
                 use_ternary: bool = False):
        self.d_model = d_model
        self.expansion = expansion
        self.d_hidden = d_model * expansion
        
        # For now, use Hadamard-like blocks (good mixing)
        if use_ternary:
            block_template = BLOCK_OPCODES['H+']
        else:
            # Use scaled Hadamard for better conditioning
            block_template = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        
        # Up projection: d → 4d
        # This requires expanding, not just mixing
        # For simplicity, we'll use multiple butterfly passes
        self.up_stages = self._build_expansion_stages(d_model, self.d_hidden, block_template)
        
        # Down projection: 4d → d
        self.down_stages = self._build_contraction_stages(self.d_hidden, d_model, block_template)
    
    def _build_expansion_stages(self, d_in, d_out, block):
        """Build stages for expansion (d → 4d)."""
        # Simplified: just use d_out with butterfly mixing
        # Real implementation would need proper expansion logic
        num_stages = int(np.log2(d_out))
        
        stages = []
        for stage in range(num_stages):
            stride = 2 ** stage
            blocks = [block.copy() for _ in range(d_out // 2)]
            layer = ButterflyLayer(d_out, stride, blocks)
            stages.append(layer)
        
        return stages
    
    def _build_contraction_stages(self, d_in, d_out, block):
        """Build stages for contraction (4d → d)."""
        # Simplified: just use d_in with butterfly mixing
        num_stages = int(np.log2(d_in))
        
        stages = []
        for stage in range(num_stages):
            stride = 2 ** stage
            blocks = [block.copy() for _ in range(d_in // 2)]
            layer = ButterflyLayer(d_in, stride, blocks)
            stages.append(layer)
        
        return stages
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Note: This is simplified. Real implementation needs:
        - Proper expansion from d to 4d
        - GELU activation
        - Proper contraction from 4d to d
        """
        # Pad input to d_hidden
        x_padded = np.zeros(self.d_hidden)
        x_padded[:self.d_model] = x
        
        # Up projection
        for layer in self.up_stages:
            x_padded = layer.forward(x_padded)
        
        # Activation (GELU approximation)
        x_padded = x_padded * (1 / (1 + np.exp(-1.702 * x_padded)))
        
        # Down projection
        for layer in self.down_stages:
            x_padded = layer.forward(x_padded)
        
        # Return first d_model elements
        return x_padded[:self.d_model]


def test_butterfly_mlp():
    """Test TriX Butterfly MLP."""
    print(f"\n[TRIX BUTTERFLY MLP]")
    
    d_model = 8
    mlp = TriXButterflyMLP(d_model, expansion=4, use_ternary=True)
    
    x = np.random.randn(d_model)
    y = mlp.forward(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Input: {x[:4]}...")
    print(f"  Output: {y[:4]}...")
    
    # Compare parameter count
    dense_params = d_model * (4 * d_model) + (4 * d_model) * d_model
    butterfly_params = 2 * int(np.log2(4 * d_model)) * (4 * d_model // 2) * 4
    
    print(f"  Dense params: {dense_params}")
    print(f"  Butterfly params: {butterfly_params}")
    print(f"  Reduction: {dense_params / butterfly_params:.1f}x")


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("BUTTERFLY MATMUL TESTS")
    print("=" * 60)
    
    results = {}
    
    # Core butterfly tests
    results['butterfly_identity_8'] = test_identity(8)
    results['butterfly_identity_16'] = test_identity(16)
    results['butterfly_hadamard_8'] = test_hadamard(8)
    results['butterfly_hadamard_16'] = test_hadamard(16)
    
    # Monarch structure test
    results['monarch_permutation'] = test_monarch_structure(16)
    
    # Expressiveness
    test_ternary_expressiveness()
    
    # Random (expected to fail - just informational)
    test_random_approximation(8)
    
    # MLP prototype
    test_butterfly_mlp()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("ALL CORE TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
