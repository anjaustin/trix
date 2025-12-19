"""
TriX Neural Network Layers

High-level neural network components built on TriX sparse layers.

Recommended (emergent routing):
- TriXFFN: Feed-forward network with signature-based routing
- TriXBlock: Transformer block with TriXFFN
- TriXStack: Stack of transformer blocks

Alternative (learned routing):
- GatedFFN: Feed-forward network with learned gate network
"""

from .layers import (
    Top1Gate,
    GatedFFN,
    TriXTransformerBlock,
)

from .emergent import (
    EmergentGatedFFN,
    EmergentTransformerBlock,
)

from .trix import (
    TriXFFN,
    TriXBlock,
    TriXStack,
)

from .sparse import (
    SparseTriXFFN,
    SparseTriXBlock,
)

from .hierarchical import (
    TriXTile,
    HierarchicalTriXFFN,
    HierarchicalTriXBlock,
)

from .sparse_lookup import (
    SparseLookupFFN,
    SparseLookupBlock,
    TernarySpline2D,
    FloatSpline2D,
)

from .sparse_lookup_v2 import (
    SparseLookupFFNv2,
    SparseLookupBlockV2,
    ScoreCalibrationSpline,
)

from .temporal_tiles import (
    TemporalTileLayer,
    TemporalTileStack,
)
from .compiled_dispatch import (
    CompiledDispatch,
    CompiledEntry,
    ProfileStats,
)

__all__ = [
    # Recommended - emergent routing (zero parameters)
    "TriXFFN",
    "TriXBlock",
    "TriXStack",
    # Sparse training (Option B)
    "SparseTriXFFN",
    "SparseTriXBlock",
    # Hierarchical (The Big Leap - 64+ tiles)
    "TriXTile",
    "HierarchicalTriXFFN",
    "HierarchicalTriXBlock",
    # SparseLookup (Routing IS Computation)
    "SparseLookupFFN",
    "SparseLookupBlock",
    "TernarySpline2D",
    "FloatSpline2D",
    # SparseLookup v2 (with Surgery + Regularization)
    "SparseLookupFFNv2",
    "SparseLookupBlockV2",
    "ScoreCalibrationSpline",
    # Compiled Dispatch (Path Compilation)
    "CompiledDispatch",
    "CompiledEntry",
    "ProfileStats",
    # Temporal Tiles (Mesa 4)
    "TemporalTileLayer",
    "TemporalTileStack",
    # Alternative - learned routing
    "Top1Gate",
    "GatedFFN",
    "TriXTransformerBlock",
    # Experimental
    "EmergentGatedFFN",
    "EmergentTransformerBlock",
]
