"""
FP4 Packing Utilities

Custom 4-bit encoding for atom weights and biases using lookup tables.
Provides ~7x compression with zero quantization error.

Weight values: {-1.0, 0.0, 1.0} -> indices {0, 1, 2}
Bias values: {-2.5, -1.5, -0.5, 0.5, 1.5} -> indices {0, 1, 2, 3, 4}
"""

import torch
import struct
from typing import Tuple, List
from pathlib import Path
from dataclasses import dataclass

from .atoms_fp4 import ThresholdCircuit, ThresholdLayer


# =============================================================================
# VALUE TABLES
# =============================================================================

WEIGHT_VALUES = [-1.0, 0.0, 1.0]
BIAS_VALUES = [-2.5, -1.5, -0.5, 0.5, 1.5]

WEIGHT_TABLE = torch.tensor(WEIGHT_VALUES, dtype=torch.float32)
BIAS_TABLE = torch.tensor(BIAS_VALUES, dtype=torch.float32)

# Reverse lookup (value -> index)
WEIGHT_TO_INDEX = {v: i for i, v in enumerate(WEIGHT_VALUES)}
BIAS_TO_INDEX = {v: i for i, v in enumerate(BIAS_VALUES)}

# File format constants
FP4_MAGIC = b'TFP4'
FP4_VERSION = 1


# =============================================================================
# INDEX CONVERSION
# =============================================================================

def weight_to_index(w: float) -> int:
    """Convert weight value to 4-bit index."""
    # Round to handle floating point imprecision
    w_rounded = round(w * 2) / 2  # Round to nearest 0.5
    if w_rounded not in WEIGHT_TO_INDEX:
        raise ValueError(f"Weight {w} not in table {WEIGHT_VALUES}")
    return WEIGHT_TO_INDEX[w_rounded]


def bias_to_index(b: float) -> int:
    """Convert bias value to 4-bit index."""
    b_rounded = round(b * 2) / 2
    if b_rounded not in BIAS_TO_INDEX:
        raise ValueError(f"Bias {b} not in table {BIAS_VALUES}")
    return BIAS_TO_INDEX[b_rounded]


def index_to_weight(i: int) -> float:
    """Convert index to weight value."""
    return WEIGHT_VALUES[i]


def index_to_bias(i: int) -> float:
    """Convert index to bias value."""
    return BIAS_VALUES[i]


# =============================================================================
# NIBBLE PACKING
# =============================================================================

def pack_nibbles(values: List[int]) -> bytes:
    """
    Pack list of 4-bit values into bytes.
    Two values per byte: high nibble first.
    """
    result = bytearray()
    for i in range(0, len(values), 2):
        high = values[i] & 0x0F
        low = values[i + 1] & 0x0F if i + 1 < len(values) else 0
        result.append((high << 4) | low)
    return bytes(result)


def unpack_nibbles(data: bytes, count: int) -> List[int]:
    """
    Unpack bytes into list of 4-bit values.
    """
    result = []
    for byte in data:
        high = (byte >> 4) & 0x0F
        low = byte & 0x0F
        result.append(high)
        result.append(low)
    return result[:count]


# =============================================================================
# TENSOR PACKING
# =============================================================================

def pack_weights(tensor: torch.Tensor) -> bytes:
    """Pack weight tensor to bytes."""
    flat = tensor.flatten().tolist()
    indices = [weight_to_index(w) for w in flat]
    return pack_nibbles(indices)


def pack_biases(tensor: torch.Tensor) -> bytes:
    """Pack bias tensor to bytes."""
    flat = tensor.flatten().tolist()
    indices = [bias_to_index(b) for b in flat]
    return pack_nibbles(indices)


def unpack_weights(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Unpack bytes to weight tensor."""
    count = 1
    for dim in shape:
        count *= dim
    indices = unpack_nibbles(data, count)
    values = [index_to_weight(i) for i in indices]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def unpack_biases(data: bytes, size: int) -> torch.Tensor:
    """Unpack bytes to bias tensor."""
    indices = unpack_nibbles(data, size)
    values = [index_to_bias(i) for i in indices]
    return torch.tensor(values, dtype=torch.float32)


# =============================================================================
# LAYER PACKING
# =============================================================================

@dataclass
class PackedLayer:
    """Packed representation of a threshold layer."""
    weight_shape: Tuple[int, int]
    bias_size: int
    weight_data: bytes
    bias_data: bytes


def pack_layer(layer: ThresholdLayer) -> PackedLayer:
    """Pack a threshold layer."""
    weight_shape = tuple(layer.weights.shape)
    bias_size = layer.bias.shape[0]
    
    return PackedLayer(
        weight_shape=weight_shape,
        bias_size=bias_size,
        weight_data=pack_weights(layer.weights),
        bias_data=pack_biases(layer.bias)
    )


def unpack_layer(packed: PackedLayer) -> ThresholdLayer:
    """Unpack a threshold layer."""
    weights = unpack_weights(packed.weight_data, packed.weight_shape)
    bias = unpack_biases(packed.bias_data, packed.bias_size)
    return ThresholdLayer(weights=weights, bias=bias)


# =============================================================================
# CIRCUIT PACKING
# =============================================================================

def pack_circuit(circuit: ThresholdCircuit) -> bytes:
    """
    Pack entire circuit to bytes.
    
    Format:
        Magic (4 bytes): "TFP4"
        Version (1 byte): 1
        Num layers (1 byte)
        Name length (1 byte)
        Num inputs (1 byte)
        Num outputs (1 byte)
        Name (N bytes)
        
        Per layer:
            Weight rows (2 bytes, little-endian)
            Weight cols (2 bytes, little-endian)
            Bias size (2 bytes, little-endian)
            Weight data (ceil(rows*cols/2) bytes)
            Bias data (ceil(bias_size/2) bytes)
    """
    result = bytearray()
    
    # Header
    result.extend(FP4_MAGIC)
    result.append(FP4_VERSION)
    result.append(len(circuit.layers))
    result.append(len(circuit.name))
    result.append(circuit.num_inputs)
    result.append(circuit.num_outputs)
    result.extend(circuit.name.encode('utf-8'))
    
    # Layers
    for layer in circuit.layers:
        packed = pack_layer(layer)
        
        # Shape info
        result.extend(struct.pack('<H', packed.weight_shape[0]))
        result.extend(struct.pack('<H', packed.weight_shape[1]))
        result.extend(struct.pack('<H', packed.bias_size))
        
        # Data
        result.extend(packed.weight_data)
        result.extend(packed.bias_data)
    
    return bytes(result)


def unpack_circuit(data: bytes) -> ThresholdCircuit:
    """Unpack circuit from bytes."""
    offset = 0
    
    # Header
    magic = data[offset:offset+4]
    if magic != FP4_MAGIC:
        raise ValueError(f"Invalid magic: {magic}")
    offset += 4
    
    version = data[offset]
    if version != FP4_VERSION:
        raise ValueError(f"Unsupported version: {version}")
    offset += 1
    
    num_layers = data[offset]
    offset += 1
    
    name_len = data[offset]
    offset += 1
    
    num_inputs = data[offset]
    offset += 1
    
    num_outputs = data[offset]
    offset += 1
    
    name = data[offset:offset+name_len].decode('utf-8')
    offset += name_len
    
    # Layers
    layers = []
    for _ in range(num_layers):
        rows = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2
        
        cols = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2
        
        bias_size = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2
        
        # Weight data
        weight_count = rows * cols
        weight_bytes = (weight_count + 1) // 2
        weight_data = data[offset:offset+weight_bytes]
        offset += weight_bytes
        
        # Bias data
        bias_bytes = (bias_size + 1) // 2
        bias_data = data[offset:offset+bias_bytes]
        offset += bias_bytes
        
        packed = PackedLayer(
            weight_shape=(rows, cols),
            bias_size=bias_size,
            weight_data=weight_data,
            bias_data=bias_data
        )
        layers.append(unpack_layer(packed))
    
    return ThresholdCircuit(
        layers=layers,
        name=name,
        num_inputs=num_inputs,
        num_outputs=num_outputs
    )


# =============================================================================
# FILE I/O
# =============================================================================

def save_circuit(circuit: ThresholdCircuit, path: Path):
    """Save circuit to .fp4 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = pack_circuit(circuit)
    path.write_bytes(data)
    
    return len(data)


def load_circuit(path: Path) -> ThresholdCircuit:
    """Load circuit from .fp4 file."""
    data = Path(path).read_bytes()
    return unpack_circuit(data)


# =============================================================================
# UTILITIES
# =============================================================================

def measure_sizes(circuit: ThresholdCircuit) -> dict:
    """
    Measure storage sizes for a circuit.
    
    Returns dict with float32_bytes, fp4_bytes, compression_ratio.
    """
    # Float32 size
    float32_bytes = 0
    for layer in circuit.layers:
        float32_bytes += layer.weights.numel() * 4  # 4 bytes per float32
        float32_bytes += layer.bias.numel() * 4
    
    # FP4 size
    fp4_data = pack_circuit(circuit)
    fp4_bytes = len(fp4_data)
    
    return {
        'float32_bytes': float32_bytes,
        'fp4_bytes': fp4_bytes,
        'compression_ratio': float32_bytes / fp4_bytes if fp4_bytes > 0 else 0
    }


def verify_roundtrip(circuit: ThresholdCircuit) -> bool:
    """Verify that pack/unpack round-trip is exact."""
    packed = pack_circuit(circuit)
    unpacked = unpack_circuit(packed)
    
    # Compare layers
    if len(circuit.layers) != len(unpacked.layers):
        return False
    
    for orig, loaded in zip(circuit.layers, unpacked.layers):
        if not torch.allclose(orig.weights, loaded.weights):
            return False
        if not torch.allclose(orig.bias, loaded.bias):
            return False
    
    return True
