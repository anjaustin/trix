"""
TriX - 2-Bit Sparse Ternary Neural Networks

Content-addressable memory with learned functions.
"Qdrant with a brain at every address."

Features:
- True 2-bit ternary weights {-1, 0, +1} (4 weights per byte)
- Zero-parameter emergent routing via weight signatures
- Hierarchical content-addressable memory for O(sqrt(n)) routing
- 16x memory compression vs FP32
- 4x inference speedup at 75% sparsity

Quick Start - Drop-in FFN Replacement:
    from trix import HierarchicalTriXFFN
    
    # Replace your FFN with TriX (includes residual + normalization)
    ffn = HierarchicalTriXFFN(
        d_model=512,
        num_tiles=16,          # More tiles = more specialists
        tiles_per_cluster=4,   # Tiles per routing cluster
    )
    
    # Use like any PyTorch module
    output, routing_info, aux_losses = ffn(x)
    
    # Training: add aux_losses to your loss
    loss = task_loss + aux_losses['total_aux']

Simple 4-Tile Version:
    from trix import SparseTriXFFN
    
    ffn = SparseTriXFFN(d_model=512, num_tiles=4)
    output, gate, aux = ffn(x)

Full Transformer Block:
    from trix import HierarchicalTriXBlock
    
    block = HierarchicalTriXBlock(
        d_model=512,
        n_heads=8,
        num_tiles=16,
    )
    output, routing, aux = block(x)

Core Principle:
    Don't learn what you can read.
    Ternary weights encode preferences.
    Preferences enable routing.
    Routing enables sparsity.
    Sparsity enables speed.
"""

__version__ = "0.4.0"

# =============================================================================
# RECOMMENDED: Hierarchical Content-Addressable Memory (The Big Leap)
# =============================================================================

from .nn import (
    # The main components - use these
    HierarchicalTriXFFN,    # FFN with 2-level hierarchical routing
    HierarchicalTriXBlock,  # Full transformer block
    TriXTile,               # Individual 2-bit specialist
)

# =============================================================================
# NEW: Sparse Lookup (Routing IS The Computation)
# =============================================================================

from .nn import (
    SparseLookupFFN,        # Routing selects direction, spline selects magnitude
    SparseLookupBlock,      # Full transformer block with SparseLookup
    TernarySpline2D,        # 2D spline with ternary coefficients
)

# =============================================================================
# SIMPLE: Sparse Training (Option B) - 4 tiles, proven
# =============================================================================

from .nn import (
    SparseTriXFFN,    # Simple sparse FFN
    SparseTriXBlock,  # Simple transformer block
)

# =============================================================================
# CLASSIC: Original emergent routing (reference implementation)
# =============================================================================

from .nn import (
    TriXFFN,
    TriXBlock,
    TriXStack,
)

# =============================================================================
# CORE: Low-level kernel components
# =============================================================================

from .kernel import (
    TriXLinear,       # Base ternary linear layer
    STESign,          # Straight-through estimator for sign()
    pack_weights,     # Pack to 2-bit
    unpack_weights,   # Unpack from 2-bit
    trix_forward,     # NEON-accelerated forward
    is_neon_available,
)

# =============================================================================
# TRAINING: Quantization-aware training utilities
# =============================================================================

from .qat import (
    TernaryQuantizer,
    SoftTernaryQuantizer,
    TriXLinearQAT,
    progressive_quantization_schedule,
    QATTrainer,
)

# =============================================================================
# LEGACY: Learned routing (alternative approach)
# =============================================================================

from .nn import (
    Top1Gate,
    GatedFFN,
    TriXTransformerBlock,
)

__all__ = [
    # Version
    "__version__",
    
    # Recommended - Hierarchical (The Big Leap)
    "HierarchicalTriXFFN",
    "HierarchicalTriXBlock", 
    "TriXTile",
    
    # New - Sparse Lookup (Routing IS Computation)
    "SparseLookupFFN",
    "SparseLookupBlock",
    "TernarySpline2D",
    
    # Simple - Sparse Training
    "SparseTriXFFN",
    "SparseTriXBlock",
    
    # Classic - Emergent Routing
    "TriXFFN",
    "TriXBlock",
    "TriXStack",
    
    # Core - Kernel
    "TriXLinear",
    "STESign",
    "pack_weights",
    "unpack_weights",
    "trix_forward",
    "is_neon_available",
    
    # Training - QAT
    "TernaryQuantizer",
    "SoftTernaryQuantizer",
    "TriXLinearQAT",
    "progressive_quantization_schedule",
    "QATTrainer",
    
    # Legacy - Learned Routing
    "Top1Gate",
    "GatedFFN",
    "TriXTransformerBlock",
]
