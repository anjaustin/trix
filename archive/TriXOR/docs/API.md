# TriX API Reference

Complete API documentation for the TriX library.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Neural Network Layers](#neural-network-layers)
3. [Kernel Operations](#kernel-operations)
4. [Quantization-Aware Training](#quantization-aware-training)
5. [Compiler](#compiler)

---

## Core Modules

### Top-Level Imports

```python
from trix import (
    # Recommended - Hierarchical Architecture
    HierarchicalTriXFFN,
    HierarchicalTriXBlock,
    TriXTile,
    
    # Sparse Lookup Architecture
    SparseLookupFFN,
    SparseLookupBlock,
    TernarySpline2D,
    
    # Simple Sparse Architecture
    SparseTriXFFN,
    SparseTriXBlock,
    
    # Classic Emergent Routing
    TriXFFN,
    TriXBlock,
    TriXStack,
    
    # Low-Level Kernel
    TriXLinear,
    STESign,
    pack_weights,
    unpack_weights,
    
    # QAT Utilities
    TernaryQuantizer,
    QATTrainer,
)
```

---

## Neural Network Layers

### HierarchicalTriXFFN

Two-level hierarchical routing FFN for large-scale deployments.

```python
class HierarchicalTriXFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        d_hidden: int = None,  # Default: 4 * d_model // num_tiles
        dropout: float = 0.1,
        aux_weight: float = 0.01,
    ):
        """
        Args:
            d_model: Input/output dimension
            num_tiles: Total number of specialist tiles
            tiles_per_cluster: Tiles per routing cluster
            d_hidden: Hidden dimension per tile (auto-computed if None)
            dropout: Dropout probability
            aux_weight: Weight for auxiliary losses
        """
```

**Forward Pass:**

```python
def forward(
    self,
    x: Tensor,                    # (batch, seq, d_model)
    labels: Optional[Tensor] = None,  # For claim tracking
) -> Tuple[Tensor, Dict, Dict]:
    """
    Returns:
        output: (batch, seq, d_model)
        routing_info: {
            'cluster_indices': (batch, seq),
            'tile_indices': (batch, seq),
            'global_indices': (batch, seq),
            'cluster_scores': (batch, seq, num_clusters),
            'tile_scores': (batch, seq, tiles_per_cluster),
        }
        aux_losses: {
            'load_balance': Tensor,
            'entropy': Tensor,
            'total_aux': Tensor,
        }
    """
```

**Methods:**

```python
def get_signatures(self) -> Tensor:
    """Return all tile signatures. Shape: (num_tiles, d_model)"""

def get_routing_stats(self) -> Dict[str, Tensor]:
    """Return routing statistics from last forward pass."""

def pack_weights(self) -> Dict[str, Tensor]:
    """Pack all weights to 2-bit representation."""

def unpack_weights(self, packed: Dict[str, Tensor]):
    """Restore weights from packed representation."""
```

---

### HierarchicalTriXBlock

Full transformer block with hierarchical FFN.

```python
class HierarchicalTriXBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            num_tiles: Number of FFN tiles
            tiles_per_cluster: Tiles per cluster
            dropout: Dropout probability
            causal: Use causal attention mask
        """
```

---

### SparseLookupFFN

"Routing IS the computation" architecture.

```python
class SparseLookupFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        spline_knots: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Input/output dimension
            num_tiles: Number of direction tiles
            spline_knots: Knot points for magnitude spline
            dropout: Dropout probability
        """
```

**Concept:**
- Routing selects a **direction** (tile signature)
- Spline modulates **magnitude** based on input norm
- No hidden layer computation in the hot path

---

### SparseLookupFFNv2

Enhanced with surgery API and regularization.

```python
class SparseLookupFFNv2(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        spline_knots: int = 8,
        dropout: float = 0.1,
        island_weight: float = 0.01,  # Signature diversity loss
        ternary_weight: float = 0.01, # Ternarization loss
    ):
        """
        Additional Args:
            island_weight: Weight for island regularization
            ternary_weight: Weight for ternary convergence
        """
```

**Surgery API:**

```python
def surgery_replace_tile(self, tile_idx: int, new_signature: Tensor):
    """Replace a tile's signature."""

def surgery_merge_tiles(self, tile_a: int, tile_b: int):
    """Merge two tiles into one."""

def surgery_split_tile(self, tile_idx: int):
    """Split a tile into two."""
```

---

### TemporalTileLayer

State-aware routing for sequential tasks.

```python
class TemporalTileLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        num_tiles: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Input/output dimension
            d_state: State vector dimension
            num_tiles: Number of tiles
            dropout: Dropout probability
        """
```

**Forward Pass:**

```python
def forward(
    self,
    x: Tensor,                      # (batch, d_model)
    state: Optional[Tensor] = None, # (batch, d_state)
) -> Tuple[Tensor, Tensor, Dict]:
    """
    Single-step forward.
    
    Returns:
        output: (batch, d_model)
        new_state: (batch, d_state)
        routing_info: {...}
    """

def forward_sequence(
    self,
    x: Tensor,  # (batch, seq, d_model)
) -> Tuple[Tensor, Tensor, List[Dict]]:
    """
    Process full sequence.
    
    Returns:
        output: (batch, seq, d_model)
        final_state: (batch, d_state)
        routing_infos: List of per-step routing info
    """

def init_state(self, batch_size: int) -> Tensor:
    """Initialize state for new sequence."""
```

---

### CompiledDispatch

O(1) inference via precomputed routes.

```python
class CompiledDispatch(nn.Module):
    def __init__(self, base_ffn: SparseLookupFFNv2):
        """
        Args:
            base_ffn: The FFN to compile
        """
```

**Methods:**

```python
def profile_class(
    self,
    class_id: int,
    samples: Tensor,
) -> ProfileStats:
    """Profile routing for a class."""

def compile_stable(
    self,
    threshold: float = 0.9,
) -> int:
    """Compile all classes with stability > threshold."""

def forward(
    self,
    x: Tensor,
    class_hint: Optional[int] = None,
    confidence: float = 0.9,
) -> Tuple[Tensor, Dict, Dict]:
    """
    Forward with optional compiled dispatch.
    
    Args:
        x: Input tensor
        class_hint: Known class ID (enables O(1) dispatch)
        confidence: Required confidence for compiled path
    """
```

---

## Kernel Operations

### TriXLinear

Base ternary linear layer.

```python
class TriXLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Include bias term (default: False)
        """
```

**Methods:**

```python
def get_ternary_weights(self) -> Tensor:
    """Return quantized ternary weights."""

def get_signature(self) -> Tensor:
    """Return weight signature."""
```

---

### Weight Packing

```python
def pack_weights(weights: Tensor) -> Tensor:
    """
    Pack ternary weights to 2-bit representation.
    
    Args:
        weights: Ternary tensor with values in {-1, 0, +1}
        
    Returns:
        Packed tensor (4x smaller)
    """

def unpack_weights(packed: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    Unpack 2-bit weights to ternary.
    
    Args:
        packed: Packed weight tensor
        shape: Original weight shape
        
    Returns:
        Ternary tensor with values in {-1, 0, +1}
    """
```

---

### NEON Acceleration

```python
def trix_forward(
    input: Tensor,
    packed_weights: Tensor,
    output_features: int,
) -> Tensor:
    """
    NEON-accelerated ternary matrix multiply.
    
    Only available on ARM platforms with NEON support.
    Falls back to PyTorch on other platforms.
    """

def is_neon_available() -> bool:
    """Check if NEON acceleration is available."""
```

---

## Quantization-Aware Training

### TernaryQuantizer

```python
class TernaryQuantizer(nn.Module):
    def __init__(
        self,
        threshold: float = 0.5,
    ):
        """
        Args:
            threshold: Values with |w| < threshold become 0
        """
    
    def forward(self, weights: Tensor) -> Tensor:
        """Quantize weights to ternary."""
```

### QATTrainer

```python
class QATTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        quantizer: TernaryQuantizer,
    ):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            quantizer: Ternary quantizer
        """
    
    def train_step(
        self,
        batch: Tensor,
        labels: Tensor,
    ) -> Dict[str, float]:
        """Perform one training step with QAT."""
```

### Progressive Schedule

```python
def progressive_quantization_schedule(
    epoch: int,
    total_epochs: int,
    start_threshold: float = 0.1,
    end_threshold: float = 0.5,
) -> float:
    """
    Compute quantization threshold for progressive training.
    
    Starts with soft quantization, progressively hardens.
    """
```

---

## Compiler

### TriXCompiler

```python
from trix.compiler import TriXCompiler

compiler = TriXCompiler(
    use_fp4: bool = False,  # Use FP4 atoms (exact by construction)
    cache_dir: str = None,  # Cache for trained atoms
    verbose: bool = True,   # Print progress
)
```

**Methods:**

```python
def compile(
    self,
    spec_or_name: Union[CircuitSpec, str],
    output_dir: Optional[str] = None,
) -> CompilationResult:
    """
    Compile a circuit specification.
    
    Args:
        spec_or_name: CircuitSpec or template name 
                     ('full_adder', 'adder_8bit', etc.)
        output_dir: Directory for emitted files
        
    Returns:
        CompilationResult with topology, verification, and executor
    """
```

### CircuitSpec

```python
from trix.compiler import CircuitSpec

spec = CircuitSpec(name="my_circuit", description="...")

spec.add_input("A", width=8)
spec.add_output("Y", width=8)
spec.add_atom("gate1", "AND", inputs=["A[0]", "A[1]"], outputs=["Y[0]"])
```

---

## Type Definitions

```python
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor

RoutingInfo = Dict[str, Tensor]
AuxLosses = Dict[str, Tensor]
ForwardOutput = Tuple[Tensor, RoutingInfo, AuxLosses]
```

---

## Error Handling

All TriX components raise standard PyTorch exceptions:

- `ValueError`: Invalid arguments
- `RuntimeError`: Computation errors
- `TypeError`: Type mismatches

```python
# Example: Invalid tile count
try:
    ffn = HierarchicalTriXFFN(d_model=512, num_tiles=7)  # Not divisible
except ValueError as e:
    print(f"Configuration error: {e}")
```
