"""
TriX Python Bindings

Clean interface to the C++ NEON kernel.
2-bit ternary weights with Straight-Through Estimator for training.
"""

import os
import ctypes
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class STESign(torch.autograd.Function):
    """
    Straight-Through Estimator for ternary quantization.
    
    Forward: quantize to {-1, 0, +1} with threshold 0.5
             (matches the C++ kernel: w > 0.5 -> +1, w < -0.5 -> -1, else 0)
    Backward: pass gradient through unchanged
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        out = torch.zeros_like(x)
        out[x > 0.5] = 1.0
        out[x < -0.5] = -1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1.5] = 0
        return grad_input


# Library management
_LIB_PATH = None
_LIB = None


def _find_library() -> Optional[Path]:
    """Search for libtrix.so in common locations."""
    search_paths = [
        Path(__file__).parent / "build" / "libtrix.so",
        Path(__file__).parent / "libtrix.so",
        Path(__file__).parent.parent.parent.parent.parent / "build" / "libtrix.so",
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    return None


def _load_library():
    """Load the TriX C library."""
    global _LIB_PATH, _LIB
    
    if _LIB is not None:
        return _LIB
    
    _LIB_PATH = _find_library()
    if _LIB_PATH is None:
        raise RuntimeError(
            "libtrix.so not found. Build with:\n"
            "  cd src/trix/kernel && mkdir build && cd build && cmake .. && make"
        )
    
    _LIB = ctypes.CDLL(str(_LIB_PATH))
    
    # Set up function signatures
    _LIB.pack_weights.argtypes = [
        ctypes.c_void_p,  # weights
        ctypes.c_void_p,  # packed output
        ctypes.c_int,     # rows
        ctypes.c_int,     # cols
    ]
    
    _LIB.unpack_weights.argtypes = [
        ctypes.c_void_p,  # packed
        ctypes.c_void_p,  # output
        ctypes.c_int,     # rows
        ctypes.c_int,     # cols
    ]
    
    _LIB.trix_forward.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # packed weights
        ctypes.c_void_p,  # scales
        ctypes.c_void_p,  # gate mask
        ctypes.c_void_p,  # output
        ctypes.c_int,     # batch
        ctypes.c_int,     # in_features
        ctypes.c_int,     # out_features
        ctypes.c_int,     # num_tiles
    ]
    
    return _LIB


def is_neon_available() -> bool:
    """Check if NEON acceleration is available."""
    try:
        _load_library()
        return True
    except RuntimeError:
        return False


def pack_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Pack float32 ternary weights to 2-bit representation.
    
    Args:
        weights: Float tensor with values in {-1, 0, +1} [out_features, in_features]
        
    Returns:
        Packed uint8 tensor (4x smaller in element count)
    """
    lib = _load_library()
    
    rows, cols = weights.shape
    # Must be on CPU for C library
    weights_f32 = weights.cpu().to(torch.float32).contiguous()
    packed_cols = (cols + 3) // 4
    packed = torch.zeros((rows, packed_cols), dtype=torch.uint8)
    
    lib.pack_weights(
        weights_f32.data_ptr(),
        packed.data_ptr(),
        rows,
        cols
    )
    
    return packed


def unpack_weights(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Unpack 2-bit weights back to float32.
    
    Args:
        packed: Packed uint8 tensor
        rows: Original row count
        cols: Original column count
        
    Returns:
        Float32 tensor with ternary values
    """
    lib = _load_library()
    
    output = torch.zeros((rows, cols), dtype=torch.float32)
    
    lib.unpack_weights(
        packed.data_ptr(),
        output.data_ptr(),
        rows,
        cols
    )
    
    return output


def trix_forward(
    x: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    gate: torch.Tensor,
    out_features: int,
    num_tiles: int
) -> torch.Tensor:
    """
    Sparse forward pass using packed 2-bit weights.
    
    Args:
        x: Input tensor [batch, in_features]
        packed_weights: Packed weight tensor
        scales: Per-output scaling factors [out_features]
        gate: Tile activation mask [batch, num_tiles]
        out_features: Output dimension
        num_tiles: Number of routing tiles
        
    Returns:
        Output tensor [batch, out_features]
    """
    lib = _load_library()
    
    batch, in_features = x.shape
    x = x.contiguous()
    output = torch.zeros((batch, out_features), dtype=torch.float32)
    
    gate_mask = gate.to(torch.int8).contiguous()
    
    lib.trix_forward(
        x.data_ptr(),
        packed_weights.data_ptr(),
        scales.data_ptr(),
        gate_mask.data_ptr(),
        output.data_ptr(),
        batch,
        in_features,
        out_features,
        num_tiles
    )
    
    return output


class TriXLinear(nn.Module):
    """
    Linear layer with TriX sparse 2-bit weights.
    
    During training, uses full-precision PyTorch operations with STE.
    During inference, uses packed 2-bit NEON kernel (4x faster at 75% sparsity).
    
    Supports signature-based emergent routing via get_signature() and
    get_tile_signatures() methods.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_tiles: Number of routing tiles (default: 4)
        bias: Include bias term (default: False)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tiles: int = 4,
        bias: bool = False
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_tiles = num_tiles
        
        # Ternary weights stored as float for training
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.scales = nn.Parameter(torch.ones(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Packed weights and cached signatures for inference
        self.register_buffer('packed_weight', None)
        self.register_buffer('cached_signatures', None)
        self._packed = False
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights around ternary threshold values."""
        nn.init.uniform_(self.weight, -0.8, 0.8)
    
    def get_signature(self) -> torch.Tensor:
        """
        Get the overall signature of this layer.
        
        The signature is a ternary vector summarizing what input patterns
        this layer responds to overall (sum of weight preferences).
        
        Returns:
            Ternary tensor [in_features] with values in {-1, 0, +1}
        """
        return torch.sign(self.weight.sum(dim=0))
    
    def get_tile_signatures(self) -> torch.Tensor:
        """
        Get signatures for each tile.
        
        Each tile's signature summarizes what input patterns that tile
        prefers. Used for emergent routing decisions.
        
        Returns:
            Ternary tensor [num_tiles, in_features] with values in {-1, 0, +1}
        """
        # Use cached signatures if available (inference mode)
        if self._packed and self.cached_signatures is not None:
            return self.cached_signatures
        
        return self._compute_tile_signatures()
    
    def _compute_tile_signatures(self) -> torch.Tensor:
        """Compute tile signatures from current weights."""
        tile_size = self.out_features // self.num_tiles
        signatures = []
        
        for t in range(self.num_tiles):
            start = t * tile_size
            end = start + tile_size
            tile_weight = self.weight[start:end, :]
            signature = torch.sign(tile_weight.sum(dim=0))
            signatures.append(signature)
        
        return torch.stack(signatures)
    
    def pack(self):
        """Pack weights and cache signatures for fast inference."""
        if not self._packed:
            self.packed_weight = pack_weights(self.weight.data)
            self.cached_signatures = self._compute_tile_signatures()
            self._packed = True
    
    def unpack(self):
        """Unpack weights for training, clear caches."""
        self._packed = False
        self.packed_weight = None
        self.cached_signatures = None
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor, use_gate: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, in_features]
            gate: Tile activation [batch, num_tiles]
            use_gate: Force using gate even without packing (for inference validation)
            
        Returns:
            Output [batch, out_features]
        """
        if self._packed and not self.training:
            # Use NEON kernel (CPU only)
            return trix_forward(
                x, self.packed_weight, self.scales,
                gate, self.out_features, self.num_tiles
            )
        elif use_gate or (not self.training and gate is not None):
            # PyTorch gated inference (works on any device)
            w = STESign.apply(self.weight)
            
            # Apply tile gating
            tile_size = self.out_features // self.num_tiles
            out = torch.zeros(x.shape[0], self.out_features, device=x.device)
            
            for t in range(self.num_tiles):
                tile_mask = gate[:, t:t+1]  # [batch, 1]
                if tile_mask.sum() > 0:  # Only compute if tile is active
                    start = t * tile_size
                    end = start + tile_size
                    tile_w = w[start:end, :]
                    tile_out = torch.mm(x, tile_w.t()) * self.scales[start:end]
                    out[:, start:end] = tile_out * tile_mask
            
            if self.bias is not None:
                out = out + self.bias
            
            return out
        else:
            # PyTorch training with STE (dense, no gating)
            w = STESign.apply(self.weight)
            out = torch.mm(x, w.t()) * self.scales
            
            if self.bias is not None:
                out = out + self.bias
            
            return out
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_tiles={self.num_tiles}, "
            f"bias={self.bias is not None}"
        )
