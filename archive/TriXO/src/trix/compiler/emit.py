"""
Code Emitter

Generates TriX configuration files from composed topologies.
The output is the "bitstream" that configures the neural substrate.

Supports both float32 and FP4 weight formats.
"""

import torch
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
from datetime import datetime

from .atoms import AtomLibrary, Atom
from .atoms_fp4 import FP4AtomLibrary, ThresholdCircuit
from .compose import Topology, TileConfig, Route


@dataclass
class TrixConfig:
    """Complete TriX configuration for a compiled circuit"""
    
    # Metadata
    name: str
    version: str
    compiled_at: str
    compiler_version: str = "0.1.0"
    
    # Verification status
    verified: bool = False
    verification_accuracy: float = 0.0
    
    # Tile configurations
    num_tiles: int = 0
    tile_configs: List[dict] = None
    
    # Routing table
    num_routes: int = 0
    routes: List[dict] = None
    
    # I/O specification
    inputs: List[dict] = None
    outputs: List[dict] = None
    
    # Execution
    execution_order: List[str] = None
    
    def save(self, path: Path):
        """Save configuration to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "TrixConfig":
        """Load configuration from file"""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class Emitter:
    """
    Emits TriX configuration files from composed topologies.
    
    The emitter generates:
    1. Circuit configuration (.trix.json) - topology and routing
    2. Weight files (.weights/) - trained atom networks
    3. Manifest (.manifest.json) - metadata and checksums
    """
    
    def __init__(self, library: AtomLibrary):
        self.library = library
    
    def emit(self, topology: Topology, output_dir: Path,
             verified: bool = False, 
             accuracy: float = 0.0) -> TrixConfig:
        """
        Emit complete TriX configuration.
        
        Args:
            topology: The composed circuit topology
            output_dir: Directory to write files
            verified: Whether the circuit is verified
            accuracy: Verification accuracy
        
        Returns:
            The configuration object
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build configuration
        config = TrixConfig(
            name=topology.name,
            version="1.0.0",
            compiled_at=datetime.now().isoformat(),
            verified=verified,
            verification_accuracy=accuracy,
            num_tiles=len(topology.tiles),
            num_routes=len(topology.routes),
            execution_order=topology.execution_order,
        )
        
        # Emit tile configs
        config.tile_configs = self._emit_tiles(topology, output_dir)
        
        # Emit routes
        config.routes = self._emit_routes(topology)
        
        # Emit I/O spec
        config.inputs = [{"name": k, "wire": v} for k, v in topology.input_map.items()]
        config.outputs = [{"name": k, "wire": v} for k, v in topology.output_map.items()]
        
        # Save main config
        config.save(output_dir / f"{topology.name}.trix.json")
        
        # Generate manifest
        self._emit_manifest(topology, output_dir, config)
        
        return config
    
    def _emit_tiles(self, topology: Topology, output_dir: Path) -> List[dict]:
        """Emit tile configurations and weights"""
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        tile_configs = []
        emitted_types = set()
        
        for tid, config in topology.tiles.items():
            # Save weights for each unique atom type
            if config.atom_type not in emitted_types:
                atom = self.library.get(config.atom_type)
                weight_file = weights_dir / f"{config.atom_type}.pt"
                torch.save(atom.network.state_dict(), weight_file)
                emitted_types.add(config.atom_type)
            
            tile_configs.append({
                "tile_id": tid,
                "atom_type": config.atom_type,
                "atom_instance": config.atom_instance,
                "input_wires": config.input_wires,
                "output_wires": config.output_wires,
                "signature": config.signature.tolist(),
                "weight_file": f"weights/{config.atom_type}.pt",
            })
        
        return tile_configs
    
    def _emit_routes(self, topology: Topology) -> List[dict]:
        """Emit routing table"""
        return [
            {
                "source_tile": r.source_tile,
                "source_port": r.source_port,
                "dest_tile": r.dest_tile,
                "dest_port": r.dest_port,
                "wire": r.wire_name,
            }
            for r in topology.routes
        ]
    
    def _emit_manifest(self, topology: Topology, output_dir: Path,
                       config: TrixConfig):
        """Emit manifest with checksums"""
        manifest = {
            "name": topology.name,
            "version": config.version,
            "compiled_at": config.compiled_at,
            "compiler_version": config.compiler_version,
            "verified": config.verified,
            "files": {},
        }
        
        # Compute checksums for all generated files
        for path in output_dir.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                rel_path = path.relative_to(output_dir)
                manifest["files"][str(rel_path)] = self._file_checksum(path)
        
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def _file_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()


class FP4Emitter:
    """
    Emits FP4-packed TriX configuration files.
    
    Uses custom 4-bit lookup table encoding for weights/biases.
    """
    
    def __init__(self, library: FP4AtomLibrary):
        self.library = library
    
    def emit(self, topology: Topology, output_dir: Path,
             verified: bool = True,  # FP4 atoms are verified by construction
             accuracy: float = 1.0) -> TrixConfig:
        """
        Emit FP4-packed TriX configuration.
        """
        from .fp4_pack import save_circuit, measure_sizes
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build configuration
        config = TrixConfig(
            name=topology.name,
            version="1.0.0",
            compiled_at=datetime.now().isoformat(),
            compiler_version="0.1.0-fp4",
            verified=verified,
            verification_accuracy=accuracy,
            num_tiles=len(topology.tiles),
            num_routes=len(topology.routes),
            execution_order=topology.execution_order,
        )
        
        # Emit tile configs and FP4 weights
        config.tile_configs = self._emit_tiles_fp4(topology, output_dir)
        
        # Emit routes
        config.routes = self._emit_routes(topology)
        
        # Emit I/O spec
        config.inputs = [{"name": k, "wire": v} for k, v in topology.input_map.items()]
        config.outputs = [{"name": k, "wire": v} for k, v in topology.output_map.items()]
        
        # Save main config
        config.save(output_dir / f"{topology.name}.trix.json")
        
        # Generate manifest
        self._emit_manifest(topology, output_dir, config)
        
        return config
    
    def _emit_tiles_fp4(self, topology: Topology, output_dir: Path) -> List[dict]:
        """Emit tile configurations with FP4 weights"""
        from .fp4_pack import save_circuit
        
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        tile_configs = []
        emitted_types = set()
        
        for tid, config in topology.tiles.items():
            # Save FP4 weights for each unique atom type
            if config.atom_type not in emitted_types:
                circuit = self.library.get_atom(config.atom_type)
                weight_file = weights_dir / f"{config.atom_type}.fp4"
                save_circuit(circuit, weight_file)
                emitted_types.add(config.atom_type)
            
            tile_configs.append({
                "tile_id": tid,
                "atom_type": config.atom_type,
                "atom_instance": config.atom_instance,
                "input_wires": config.input_wires,
                "output_wires": config.output_wires,
                "signature": config.signature.tolist(),
                "weight_file": f"weights/{config.atom_type}.fp4",
                "weight_format": "fp4",
            })
        
        return tile_configs
    
    def _emit_routes(self, topology: Topology) -> List[dict]:
        """Emit routing table"""
        return [
            {
                "source_tile": r.source_tile,
                "source_port": r.source_port,
                "dest_tile": r.dest_tile,
                "dest_port": r.dest_port,
                "wire": r.wire_name,
            }
            for r in topology.routes
        ]
    
    def _emit_manifest(self, topology: Topology, output_dir: Path,
                       config: TrixConfig):
        """Emit manifest with checksums"""
        manifest = {
            "name": topology.name,
            "version": config.version,
            "compiled_at": config.compiled_at,
            "compiler_version": config.compiler_version,
            "verified": config.verified,
            "weight_format": "fp4",
            "files": {},
        }
        
        for path in output_dir.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                rel_path = path.relative_to(output_dir)
                manifest["files"][str(rel_path)] = self._file_checksum(path)
        
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def _file_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()


class FP4Loader:
    """
    Loads FP4-packed TriX configurations for execution.
    """
    
    def __init__(self):
        pass
    
    def load(self, config_path: Path) -> "CompiledCircuit":
        """Load a compiled FP4 circuit from configuration"""
        from .fp4_pack import load_circuit
        
        config_path = Path(config_path)
        
        # Load main config
        config = TrixConfig.load(config_path)
        
        # Load FP4 weights
        atom_circuits = {}
        
        for tile_config in config.tile_configs:
            atom_type = tile_config["atom_type"]
            if atom_type not in atom_circuits:
                weight_file = config_path.parent / tile_config["weight_file"]
                if weight_file.exists():
                    atom_circuits[atom_type] = load_circuit(weight_file)
        
        return FP4CompiledCircuit(config, atom_circuits)


class FP4CompiledCircuit:
    """
    A loaded FP4 circuit ready for execution.
    """
    
    def __init__(self, config: TrixConfig, circuits: Dict[str, ThresholdCircuit]):
        self.config = config
        self.circuits = circuits
        self._tile_map = {t["tile_id"]: t for t in config.tile_configs}
    
    def __call__(self, **inputs) -> Dict[str, int]:
        return self.execute(inputs)
    
    def execute(self, inputs: Dict[str, int]) -> Dict[str, int]:
        """Execute the FP4 circuit."""
        wire_values: Dict[str, float] = {}
        
        # Set inputs
        for inp in self.config.inputs:
            name = inp["name"]
            wire = inp["wire"]
            if name in inputs:
                wire_values[wire] = float(inputs[name])
        
        # Execute tiles in order
        for tile_id in self.config.execution_order:
            tile = self._tile_map[tile_id]
            
            # Gather inputs
            input_vals = []
            for wire in tile["input_wires"]:
                input_vals.append(wire_values.get(wire, 0.0))
            
            # Execute threshold circuit
            circuit = self.circuits[tile["atom_type"]]
            x = torch.tensor([input_vals], dtype=torch.float32)
            output = circuit(x)
            
            # Store outputs
            for i, wire in enumerate(tile["output_wires"]):
                wire_values[wire] = output[0, i].item()
        
        # Collect outputs - match wire patterns
        results = {}
        for wire, value in wire_values.items():
            for out in self.config.outputs:
                out_wire = out["wire"]
                if wire == out_wire or wire.startswith(out_wire + "["):
                    results[wire] = int(value > 0.5)
                    break
        
        return results
    
    @property
    def is_verified(self) -> bool:
        return self.config.verified
    
    def summary(self) -> str:
        status = "VERIFIED" if self.is_verified else "UNVERIFIED"
        return (
            f"FP4CompiledCircuit: {self.config.name} [{status}]\n"
            f"  Tiles: {self.config.num_tiles}\n"
            f"  Routes: {self.config.num_routes}\n"
            f"  Format: FP4 (4-bit packed)\n"
            f"  Inputs: {[i['name'] for i in self.config.inputs]}\n"
            f"  Outputs: {[o['name'] for o in self.config.outputs]}"
        )


class TrixLoader:
    """
    Loads compiled TriX configurations for execution.
    """
    
    def __init__(self, library: AtomLibrary):
        self.library = library
    
    def load(self, config_path: Path) -> "CompiledCircuit":
        """Load a compiled circuit from configuration"""
        config_path = Path(config_path)
        
        # Load main config
        config = TrixConfig.load(config_path)
        
        # Load weights
        weights_dir = config_path.parent / "weights"
        atom_networks = {}
        
        for tile_config in config.tile_configs:
            atom_type = tile_config["atom_type"]
            if atom_type not in atom_networks:
                weight_file = config_path.parent / tile_config["weight_file"]
                
                # Create network and load weights
                atom = self.library.get(atom_type, train_if_missing=False)
                if weight_file.exists():
                    atom.network.load_state_dict(torch.load(weight_file))
                atom_networks[atom_type] = atom
        
        return CompiledCircuit(config, atom_networks)


class CompiledCircuit:
    """
    A loaded, executable compiled circuit.
    """
    
    def __init__(self, config: TrixConfig, atoms: Dict[str, Atom]):
        self.config = config
        self.atoms = atoms
        
        # Build execution structures
        self._wire_values: Dict[str, float] = {}
        self._tile_map = {t["tile_id"]: t for t in config.tile_configs}
    
    def __call__(self, **inputs) -> Dict[str, int]:
        """Execute the circuit"""
        return self.execute(inputs)
    
    def execute(self, inputs: Dict[str, int]) -> Dict[str, int]:
        """
        Execute the compiled circuit.
        
        Args:
            inputs: Dictionary mapping input names to values
        
        Returns:
            Dictionary mapping output names to values
        """
        # Reset wire values
        self._wire_values = {}
        
        # Set inputs
        for inp in self.config.inputs:
            name = inp["name"]
            wire = inp["wire"]
            if name in inputs:
                self._wire_values[wire] = float(inputs[name])
        
        # Execute tiles in order
        for tile_id in self.config.execution_order:
            tile = self._tile_map[tile_id]
            
            # Gather inputs
            input_vals = []
            for wire in tile["input_wires"]:
                input_vals.append(self._wire_values.get(wire, 0.0))
            
            # Execute atom
            atom = self.atoms[tile["atom_type"]]
            output = atom(torch.tensor([input_vals]))
            
            # Store outputs
            for i, wire in enumerate(tile["output_wires"]):
                self._wire_values[wire] = output[0, i].item()
        
        # Collect outputs
        results = {}
        for out in self.config.outputs:
            name = out["name"]
            wire = out["wire"]
            if wire in self._wire_values:
                results[name] = int(self._wire_values[wire])
        
        return results
    
    @property
    def is_verified(self) -> bool:
        return self.config.verified
    
    def summary(self) -> str:
        """Summary of the compiled circuit"""
        status = "VERIFIED" if self.is_verified else "UNVERIFIED"
        return (
            f"CompiledCircuit: {self.config.name} [{status}]\n"
            f"  Tiles: {self.config.num_tiles}\n"
            f"  Routes: {self.config.num_routes}\n"
            f"  Inputs: {[i['name'] for i in self.config.inputs]}\n"
            f"  Outputs: {[o['name'] for o in self.config.outputs]}"
        )
