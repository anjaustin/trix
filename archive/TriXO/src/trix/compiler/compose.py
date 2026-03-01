"""
Composition Engine

Wires verified atoms together to create circuits.
This generates the "Hollywood Squares" topology.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .atoms import AtomLibrary, Atom
from .decompose import DecompositionResult
from .spec import CircuitSpec, AtomInstance


class MessageType(Enum):
    """Types of messages in the Hollywood Squares bus"""
    DATA = "data"           # Normal data transfer
    SYNC = "sync"           # Synchronization signal
    CONTROL = "control"     # Control flow


@dataclass
class Message:
    """A message on the Hollywood Squares bus"""
    source: str             # Source tile/wire
    destination: str        # Destination tile/wire
    msg_type: MessageType
    payload: Any
    timestamp: int = 0


@dataclass
class TileConfig:
    """Configuration for a single tile"""
    tile_id: str
    atom_type: str
    atom_instance: str      # Name in the circuit spec
    input_wires: List[str]
    output_wires: List[str]
    signature: torch.Tensor  # Content-addressable signature
    
    def __hash__(self):
        return hash(self.tile_id)


@dataclass
class Route:
    """A route in the topology"""
    source_tile: str
    source_port: int
    dest_tile: str
    dest_port: int
    wire_name: str


@dataclass
class Topology:
    """
    Complete topology for a compiled circuit.
    
    This is the "bitstream" - the configuration that turns
    the TriX substrate into the desired circuit.
    """
    name: str
    tiles: Dict[str, TileConfig]
    routes: List[Route]
    input_map: Dict[str, str]   # external input -> wire
    output_map: Dict[str, str]  # wire -> external output
    execution_order: List[str]  # Tile execution order
    
    def summary(self) -> str:
        lines = [
            f"Topology: {self.name}",
            f"  Tiles: {len(self.tiles)}",
            f"  Routes: {len(self.routes)}",
            f"  Inputs: {list(self.input_map.keys())}",
            f"  Outputs: {list(self.output_map.keys())}",
        ]
        return "\n".join(lines)
    
    def visualize(self) -> str:
        """ASCII visualization of the topology"""
        lines = [
            "=" * 60,
            f"TOPOLOGY: {self.name}",
            "=" * 60,
            "",
            "TILES:",
        ]
        
        for tid, config in self.tiles.items():
            lines.append(f"  [{tid}]")
            lines.append(f"    Type: {config.atom_type}")
            lines.append(f"    In:  {config.input_wires}")
            lines.append(f"    Out: {config.output_wires}")
        
        lines.append("")
        lines.append("ROUTES:")
        for route in self.routes:
            lines.append(f"  {route.source_tile}[{route.source_port}] "
                        f"--({route.wire_name})--> "
                        f"{route.dest_tile}[{route.dest_port}]")
        
        lines.append("")
        lines.append("EXECUTION ORDER:")
        for i, tid in enumerate(self.execution_order):
            lines.append(f"  {i+1}. {tid}")
        
        return "\n".join(lines)


class Composer:
    """
    Composes verified atoms into a complete circuit topology.
    
    The composer:
    1. Allocates tiles for each atom instance
    2. Generates signatures for content-addressable routing
    3. Creates the routing table (Hollywood Squares bus)
    4. Determines execution order
    """
    
    def __init__(self, library: AtomLibrary):
        self.library = library
        self._tile_counter = 0
    
    def compose(self, decomposition: DecompositionResult) -> Topology:
        """Compose a circuit from its decomposition"""
        spec = decomposition.spec
        
        # Allocate tiles
        tiles = self._allocate_tiles(spec)
        
        # Generate routes
        routes = self._generate_routes(spec, tiles)
        
        # Build input/output maps
        input_map = {w.name: w.name for w in spec.inputs}
        output_map = {w.name: w.name for w in spec.outputs}
        
        # Map execution order to tile IDs
        exec_order = []
        for atom_name in decomposition.dependency_order:
            for tid, config in tiles.items():
                if config.atom_instance == atom_name:
                    exec_order.append(tid)
                    break
        
        return Topology(
            name=spec.name,
            tiles=tiles,
            routes=routes,
            input_map=input_map,
            output_map=output_map,
            execution_order=exec_order,
        )
    
    def _allocate_tiles(self, spec: CircuitSpec) -> Dict[str, TileConfig]:
        """Allocate a tile for each atom instance"""
        tiles = {}
        
        for atom_inst in spec.atoms:
            tile_id = self._generate_tile_id(atom_inst.atom_type)
            
            # Generate signature from atom type and connections
            signature = self._generate_signature(atom_inst)
            
            config = TileConfig(
                tile_id=tile_id,
                atom_type=atom_inst.atom_type,
                atom_instance=atom_inst.name,
                input_wires=atom_inst.inputs,
                output_wires=atom_inst.outputs,
                signature=signature,
            )
            
            tiles[tile_id] = config
        
        return tiles
    
    def _generate_tile_id(self, atom_type: str) -> str:
        """Generate unique tile ID"""
        tid = f"TILE_{self._tile_counter:04d}_{atom_type}"
        self._tile_counter += 1
        return tid
    
    def _generate_signature(self, atom_inst: AtomInstance) -> torch.Tensor:
        """
        Generate content-addressable signature for a tile.
        
        The signature is derived from:
        - Atom type (what operation)
        - Input wire names (what data it expects)
        
        This enables the "content-addressable function" behavior:
        data with matching signatures routes to this tile.
        """
        # Simple hash-based signature
        sig_str = f"{atom_inst.atom_type}:{','.join(atom_inst.inputs)}"
        sig_hash = hash(sig_str)
        
        # Convert to ternary signature vector
        sig_len = 32  # Signature length
        sig = torch.zeros(sig_len)
        
        for i in range(sig_len):
            bit = (sig_hash >> i) & 3  # 2 bits
            if bit == 0:
                sig[i] = 0
            elif bit == 1:
                sig[i] = 1
            elif bit == 2:
                sig[i] = -1
            else:
                sig[i] = 0
        
        return sig
    
    def _generate_routes(self, spec: CircuitSpec, 
                        tiles: Dict[str, TileConfig]) -> List[Route]:
        """Generate routing table connecting tiles"""
        routes = []
        
        # Map wire names to producing tiles/ports
        wire_producers: Dict[str, Tuple[str, int]] = {}
        
        # Input wires are "produced" by the environment
        for w in spec.inputs:
            if w.width == 1:
                wire_producers[w.name] = ("__INPUT__", 0)
            else:
                for i in range(w.width):
                    wire_producers[w.bit(i)] = ("__INPUT__", i)
        
        # Find wire producers from tiles
        for tid, config in tiles.items():
            for port, wire in enumerate(config.output_wires):
                wire_producers[wire] = (tid, port)
        
        # Create routes for each tile's inputs
        for tid, config in tiles.items():
            for port, wire in enumerate(config.input_wires):
                if wire in wire_producers:
                    src_tile, src_port = wire_producers[wire]
                    routes.append(Route(
                        source_tile=src_tile,
                        source_port=src_port,
                        dest_tile=tid,
                        dest_port=port,
                        wire_name=wire,
                    ))
        
        # Create routes for outputs
        for w in spec.outputs:
            if w.width == 1:
                if w.name in wire_producers:
                    src_tile, src_port = wire_producers[w.name]
                    routes.append(Route(
                        source_tile=src_tile,
                        source_port=src_port,
                        dest_tile="__OUTPUT__",
                        dest_port=0,
                        wire_name=w.name,
                    ))
            else:
                for i in range(w.width):
                    wire = w.bit(i)
                    if wire in wire_producers:
                        src_tile, src_port = wire_producers[wire]
                        routes.append(Route(
                            source_tile=src_tile,
                            source_port=src_port,
                            dest_tile="__OUTPUT__",
                            dest_port=i,
                            wire_name=wire,
                        ))
        
        return routes


class CircuitExecutor:
    """
    Executes a composed circuit using the Hollywood Squares model.
    
    Execution follows the message-passing paradigm:
    1. Input values are placed on input wires
    2. Messages propagate through the routing network
    3. Each tile executes when all inputs are available
    4. Output values are collected from output wires
    """
    
    def __init__(self, topology: Topology, library: AtomLibrary):
        self.topology = topology
        self.library = library
        self._atoms: Dict[str, Atom] = {}
        
        # Load atoms for each tile type
        for tid, config in topology.tiles.items():
            if config.atom_type not in self._atoms:
                self._atoms[config.atom_type] = library.get(config.atom_type)
    
    def execute(self, inputs: Dict[str, int]) -> Dict[str, int]:
        """
        Execute the circuit on given inputs.
        
        Args:
            inputs: Dictionary mapping input names to bit values
        
        Returns:
            Dictionary mapping output names to bit values
        """
        # Initialize wire values
        wire_values: Dict[str, float] = {}
        
        # Set input values
        for name, value in inputs.items():
            wire_values[name] = float(value)
        
        # Execute tiles in order
        for tile_id in self.topology.execution_order:
            config = self.topology.tiles[tile_id]
            
            # Gather input values
            input_bits = []
            for wire in config.input_wires:
                input_bits.append(wire_values.get(wire, 0.0))
            
            # Execute atom
            atom = self._atoms[config.atom_type]
            input_tensor = torch.tensor([input_bits])
            output = atom(input_tensor)
            
            # Store output values
            for i, wire in enumerate(config.output_wires):
                wire_values[wire] = output[0, i].item()
        
        # Collect outputs - return all wire values that match output patterns
        outputs = {}
        for wire, value in wire_values.items():
            # Check if this wire is an output
            for out_wire in self.topology.output_map.keys():
                if wire == out_wire or wire.startswith(out_wire + "["):
                    outputs[wire] = int(value)
                    break
        
        return outputs
    
    def execute_traced(self, inputs: Dict[str, int]) -> Tuple[Dict[str, int], List[Message]]:
        """
        Execute with full message trace (for debugging/visualization).
        """
        wire_values: Dict[str, float] = {}
        messages: List[Message] = []
        timestamp = 0
        
        # Input messages
        for name, value in inputs.items():
            wire_values[name] = float(value)
            messages.append(Message(
                source="__INPUT__",
                destination=name,
                msg_type=MessageType.DATA,
                payload=value,
                timestamp=timestamp,
            ))
        timestamp += 1
        
        # Execute tiles
        for tile_id in self.topology.execution_order:
            config = self.topology.tiles[tile_id]
            
            # Input messages to tile
            input_bits = []
            for wire in config.input_wires:
                val = wire_values.get(wire, 0.0)
                input_bits.append(val)
                messages.append(Message(
                    source=wire,
                    destination=tile_id,
                    msg_type=MessageType.DATA,
                    payload=val,
                    timestamp=timestamp,
                ))
            timestamp += 1
            
            # Execute
            atom = self._atoms[config.atom_type]
            output = atom(torch.tensor([input_bits]))
            
            # Output messages from tile
            for i, wire in enumerate(config.output_wires):
                val = output[0, i].item()
                wire_values[wire] = val
                messages.append(Message(
                    source=tile_id,
                    destination=wire,
                    msg_type=MessageType.DATA,
                    payload=val,
                    timestamp=timestamp,
                ))
            timestamp += 1
        
        # Collect outputs
        outputs = {}
        for wire, ext_name in self.topology.output_map.items():
            if wire in wire_values:
                outputs[ext_name] = int(wire_values[wire])
        
        return outputs, messages
