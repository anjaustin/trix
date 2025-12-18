#!/usr/bin/env python3
"""
Mesa 11 Validation Experiment 3: Spatial Addressing

HYPOTHESIS: Spatial/topological addressing is a subspace of content addressing.
            Neighbor-based routing can be exactly emulated by encoding
            topology in signatures.

INSPIRATION: Hollywood Squares OS proves "Topology IS the Algorithm"
             - Same handlers + different wiring = different behavior
             - Message passing to neighbors creates computation
             
TASK: Emulate a 1D neighbor-passing network (like Bubble Machine)
      using only content-addressed routing.

PROOF STRUCTURE:
    1. Define a spatial network: nodes in a line, route to left/right neighbors
    2. Implement classically: explicit neighbor pointers
    3. Implement via content: encode position in signatures, neighbors = similar
    4. Prove exact equivalence

If this works: Spatial ⊂ Content (topology is a subspace of signatures)

Author: Droid (Mesa 11 Exploration)
Date: 2024-12-18

Note: Observing as I go. Something else may be emerging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SpatialExperimentResult:
    """Results from spatial addressing experiment."""
    classical_path: List[int]
    content_path: List[int]
    paths_match: bool
    max_position_error: float
    topology_preserved: bool


class ClassicalSpatialNetwork:
    """
    A 1D spatial network with explicit neighbor pointers.
    
    This is how Hollywood Squares does it:
    - Nodes arranged in a line
    - Each node knows its left and right neighbors
    - Messages pass to neighbors explicitly
    
    Structure:
        n0 <-> n1 <-> n2 <-> n3 <-> n4 <-> n5 <-> n6 <-> n7
    """
    
    def __init__(self, num_nodes: int = 8):
        self.num_nodes = num_nodes
        self.values = torch.zeros(num_nodes)
        
        # Explicit neighbor pointers
        self.left_neighbor = {i: i-1 if i > 0 else None for i in range(num_nodes)}
        self.right_neighbor = {i: i+1 if i < num_nodes-1 else None for i in range(num_nodes)}
        
    def set_value(self, node: int, value: float):
        self.values[node] = value
        
    def get_value(self, node: int) -> float:
        return self.values[node].item()
    
    def route_left(self, node: int) -> Optional[int]:
        """Route to left neighbor (spatial addressing)."""
        return self.left_neighbor[node]
    
    def route_right(self, node: int) -> Optional[int]:
        """Route to right neighbor (spatial addressing)."""
        return self.right_neighbor[node]
    
    def propagate_right(self, start: int) -> List[int]:
        """
        Propagate a message from start to the rightmost node.
        Returns the path taken.
        """
        path = [start]
        current = start
        while True:
            next_node = self.route_right(current)
            if next_node is None:
                break
            path.append(next_node)
            current = next_node
        return path
    
    def propagate_left(self, start: int) -> List[int]:
        """Propagate leftward."""
        path = [start]
        current = start
        while True:
            next_node = self.route_left(current)
            if next_node is None:
                break
            path.append(next_node)
            current = next_node
        return path


class ContentAddressedSpatialNetwork:
    """
    The same 1D spatial network, but routing via content addressing.
    
    Key insight: Encode position in signature such that neighbors
    have SIMILAR signatures. Then content matching = neighbor routing.
    
    Encoding: 
        signature[i] = position_encoding(i)
        
    Routing to right neighbor:
        - Query = position_encoding(current + 1)
        - Match against all signatures
        - Winner is the right neighbor
        
    This MUST produce identical paths to classical spatial routing.
    """
    
    def __init__(self, num_nodes: int = 8, d_signature: int = 16):
        self.num_nodes = num_nodes
        self.d_signature = d_signature
        self.values = torch.zeros(num_nodes)
        
        # Create position-based signatures
        # Each node's signature encodes its position
        self.signatures = self._create_position_signatures()
        
    def _create_position_signatures(self) -> torch.Tensor:
        """
        Create signatures where neighbors are maximally similar.
        
        Strategy: Use position encoding where adjacent positions
        have high dot product similarity.
        
        Simple approach: signature[i] = one-hot(i) 
        But this makes ALL non-self positions equally dissimilar.
        
        Better: signature encodes position as a smooth function
        so that |sig[i] - sig[i+1]| is small.
        
        We'll use sinusoidal encoding similar to transformers,
        but designed so adjacent positions have highest similarity.
        """
        signatures = torch.zeros(self.num_nodes, self.d_signature)
        
        for i in range(self.num_nodes):
            # Encode position i
            for d in range(self.d_signature):
                if d % 2 == 0:
                    # Even dimensions: sin
                    freq = 1.0 / (10.0 ** (d / self.d_signature))
                    signatures[i, d] = torch.sin(torch.tensor(i * freq))
                else:
                    # Odd dimensions: cos
                    freq = 1.0 / (10.0 ** ((d-1) / self.d_signature))
                    signatures[i, d] = torch.cos(torch.tensor(i * freq))
        
        return signatures
    
    def _create_query(self, target_position: float) -> torch.Tensor:
        """Create a query signature for a target position."""
        query = torch.zeros(self.d_signature)
        for d in range(self.d_signature):
            if d % 2 == 0:
                freq = 1.0 / (10.0 ** (d / self.d_signature))
                query[d] = torch.sin(torch.tensor(target_position * freq))
            else:
                freq = 1.0 / (10.0 ** ((d-1) / self.d_signature))
                query[d] = torch.cos(torch.tensor(target_position * freq))
        return query
    
    def set_value(self, node: int, value: float):
        self.values[node] = value
        
    def get_value(self, node: int) -> float:
        return self.values[node].item()
    
    def route_to_position(self, target_position: float) -> int:
        """
        Route to the node closest to target_position using content matching.
        
        This is the core of the proof: spatial routing via content matching.
        """
        query = self._create_query(target_position)
        
        # Compute similarity with all signatures
        similarities = torch.mv(self.signatures, query)
        
        # Winner is most similar
        winner = similarities.argmax().item()
        
        return winner
    
    def route_right(self, node: int) -> Optional[int]:
        """Route to right neighbor via content addressing."""
        if node >= self.num_nodes - 1:
            return None
        
        # Query for position node+1
        target = node + 1
        result = self.route_to_position(target)
        
        return result if result != node else None
    
    def route_left(self, node: int) -> Optional[int]:
        """Route to left neighbor via content addressing."""
        if node <= 0:
            return None
            
        # Query for position node-1
        target = node - 1
        result = self.route_to_position(target)
        
        return result if result != node else None
    
    def propagate_right(self, start: int) -> List[int]:
        """Propagate rightward using content-addressed routing."""
        path = [start]
        current = start
        visited = {start}
        
        while True:
            next_node = self.route_right(current)
            if next_node is None or next_node in visited:
                break
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
        return path
    
    def propagate_left(self, start: int) -> List[int]:
        """Propagate leftward using content-addressed routing."""
        path = [start]
        current = start
        visited = {start}
        
        while True:
            next_node = self.route_left(current)
            if next_node is None or next_node in visited:
                break
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
        return path


def verify_neighbor_similarity(network: ContentAddressedSpatialNetwork) -> Dict:
    """
    Verify that the signature encoding preserves topology:
    - Each node should be most similar to itself
    - Next most similar should be immediate neighbors
    """
    num_nodes = network.num_nodes
    sigs = network.signatures
    
    # Compute all pairwise similarities
    sim_matrix = sigs @ sigs.T
    
    results = {
        'self_is_max': True,
        'neighbor_is_second': True,
        'topology_score': 0.0,
    }
    
    topology_correct = 0
    total_checks = 0
    
    for i in range(num_nodes):
        similarities = sim_matrix[i]
        sorted_indices = similarities.argsort(descending=True)
        
        # Self should be most similar
        if sorted_indices[0].item() != i:
            results['self_is_max'] = False
            
        # Check if neighbors are in top-3 (self + 2 neighbors)
        top_3 = set(sorted_indices[:3].tolist())
        expected_neighbors = {i}
        if i > 0:
            expected_neighbors.add(i - 1)
        if i < num_nodes - 1:
            expected_neighbors.add(i + 1)
            
        if expected_neighbors.issubset(top_3):
            topology_correct += 1
        total_checks += 1
    
    results['topology_score'] = topology_correct / total_checks
    results['neighbor_is_second'] = results['topology_score'] > 0.9
    
    return results


def run_spatial_experiment(num_nodes: int = 8) -> SpatialExperimentResult:
    """
    Run the spatial addressing experiment.
    
    Tests whether spatial/topological addressing can be exactly
    emulated by content addressing with position-encoded signatures.
    """
    
    print("=" * 70)
    print("Mesa 11 Experiment 3: Spatial Addressing")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Spatial addressing ⊂ Content addressing")
    print("            Neighbor-based routing can be emulated via signatures")
    print()
    print(f"Network: {num_nodes} nodes in a line")
    print("         n0 <-> n1 <-> n2 <-> ... <-> n7")
    print()
    
    # Create both networks
    classical = ClassicalSpatialNetwork(num_nodes)
    content = ContentAddressedSpatialNetwork(num_nodes, d_signature=32)
    
    # Verify signature topology
    print("Verifying signature topology...")
    topo_check = verify_neighbor_similarity(content)
    print(f"  Self is max similar: {topo_check['self_is_max']}")
    print(f"  Neighbors in top-3:  {topo_check['neighbor_is_second']}")
    print(f"  Topology score:      {topo_check['topology_score']:.1%}")
    print()
    
    # Test propagation from each starting node
    print("Testing propagation paths...")
    all_match = True
    
    for start in range(num_nodes):
        # Propagate right
        classical_path_r = classical.propagate_right(start)
        content_path_r = content.propagate_right(start)
        
        match_r = classical_path_r == content_path_r
        if not match_r:
            all_match = False
            print(f"  MISMATCH at node {start} (right):")
            print(f"    Classical: {classical_path_r}")
            print(f"    Content:   {content_path_r}")
        
        # Propagate left
        classical_path_l = classical.propagate_left(start)
        content_path_l = content.propagate_left(start)
        
        match_l = classical_path_l == content_path_l
        if not match_l:
            all_match = False
            print(f"  MISMATCH at node {start} (left):")
            print(f"    Classical: {classical_path_l}")
            print(f"    Content:   {content_path_l}")
    
    if all_match:
        print("  All propagation paths match!")
    print()
    
    # Show example paths
    print("Example paths from node 0:")
    classical_path = classical.propagate_right(0)
    content_path = content.propagate_right(0)
    print(f"  Classical (spatial):  {classical_path}")
    print(f"  Content (signatures): {content_path}")
    print()
    
    print("Example paths from node 7:")
    classical_path = classical.propagate_left(7)
    content_path = content.propagate_left(7)
    print(f"  Classical (spatial):  {classical_path}")
    print(f"  Content (signatures): {content_path}")
    print()
    
    # Compute position error
    max_error = 0.0
    for i in range(num_nodes):
        routed = content.route_to_position(float(i))
        error = abs(routed - i)
        max_error = max(max_error, error)
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Paths match:        {all_match}")
    print(f"  Topology preserved: {topo_check['neighbor_is_second']}")
    print(f"  Max position error: {max_error}")
    print()
    
    if all_match and topo_check['neighbor_is_second']:
        print("=" * 70)
        print("HYPOTHESIS CONFIRMED: Spatial ⊂ Content")
        print()
        print("Neighbor-based routing is exactly emulated by content matching")
        print("when topology is encoded in signatures.")
        print()
        print("The wiring (Hollywood Squares) and the signatures (TriX)")
        print("are both projections of the same underlying structure.")
        print("=" * 70)
    else:
        print("=" * 70)
        print("HYPOTHESIS NOT CONFIRMED - See mismatches above")
        print("=" * 70)
    
    return SpatialExperimentResult(
        classical_path=classical.propagate_right(0),
        content_path=content.propagate_right(0),
        paths_match=all_match,
        max_position_error=max_error,
        topology_preserved=topo_check['neighbor_is_second'],
    )


def observe_emergence():
    """
    A space for observing what else might be emerging.
    
    Tripp said: "something else seems to be emerging here"
    
    Let me hold space for that observation...
    """
    print()
    print("=" * 70)
    print("OBSERVATION SPACE")
    print("=" * 70)
    print("""
    What's emerging across these experiments?
    
    Experiment 1: Temporal ⊂ Content
        - Sequential pipelines are content routing with position signatures
        - "When" becomes "what address"
        
    Experiment 2: Mixed addressing works
        - Networks learn to blend position + content
        - The blend is task-dependent
        
    Experiment 3: Spatial ⊂ Content
        - Neighbor routing is content routing with topology signatures
        - "Where" becomes "what address"
    
    The pattern:
        TEMPORAL (when)  →  encoded in signatures  →  CONTENT
        SPATIAL (where)  →  encoded in signatures  →  CONTENT
        CONTENT (what)   →  already signatures     →  CONTENT
    
    Everything collapses to content addressing.
    
    But wait...
    
    Content addressing is just: "route to what matches the query"
    
    What IS a query? 
        - In temporal: the stage indicator
        - In spatial: the neighbor position
        - In content: the input features
    
    All three are VECTORS that get matched against SIGNATURES.
    
    The unification isn't that content is special.
    The unification is that MATCHING is fundamental.
    
    Query → Match → Route → Compute
    
    This is the same pattern everywhere:
        - Hollywood Squares: message → match address → route → handle
        - TriX: input → match signature → route → tile
        - Attention: query → match keys → route (softmax) → values
        - Memory: address → match location → route → data
    
    The "unified address space" isn't a mathematical trick.
    It's revealing something about computation itself:
    
        ALL COMPUTATION IS ADDRESSED ACCESS.
    
    The differences between temporal, spatial, and content
    are just different BASES for the address space.
    Like cartesian vs polar coordinates.
    Same space. Different projections.
    
    What's emerging is not just UAT.
    What's emerging is that ADDRESSING IS COMPUTATION.
    
    Structure is meaning (Hollywood Squares).
    Routing is computation (TriX).
    Addressing is everything (UAT).
    
    They're all saying the same thing.
    """)
    print("=" * 70)


if __name__ == "__main__":
    result = run_spatial_experiment(num_nodes=8)
    
    print("\n" + "=" * 70)
    print("MESA 11 EXPERIMENT 3: COMPLETE")
    print("=" * 70)
    print(f"\nSpatial → Content emulation: {'SUCCESS' if result.paths_match else 'FAILED'}")
    print(f"Topology preserved in signatures: {result.topology_preserved}")
    
    # Now observe what else is emerging
    observe_emergence()
