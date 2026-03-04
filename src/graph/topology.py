"""
Railway Topology Definition.
Defines the physical and logical structure of the railway network.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from enum import Enum


class EdgeType(Enum):
    """Types of connections between railway components"""
    PHYSICAL = "physical"
    ELECTRICAL = "electrical"
    LOAD_SHARING = "load_sharing"
    SIGNAL_CONTROL = "signal_control"


class ComponentType(Enum):
    """Types of railway components"""
    TRACK = "track"
    SWITCH = "switch"
    SIGNAL = "signal"
    BRIDGE = "bridge"


@dataclass
class NodeDefinition:
    """Definition of a single node in the railway graph"""
    node_id: str
    name: str
    component_type: ComponentType
    position: Tuple[float, float]
    properties: Dict = field(default_factory=dict)


@dataclass
class EdgeDefinition:
    """Definition of an edge between two nodes"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    bidirectional: bool = True


class RailwayTopology:
    """
    Defines the complete railway network topology.
    
    Layout visualization:
    
    [SG1]--[SW1]--[T01]--[T02]--[T03]--[BR1]--[T04]--[T05]--[SW3]--[SG2]
                                  |                    |
                                [SW2]                [T06]
                                  |                    |
              [T07]--[BR2]--[SW4]--[T08]--[T09]--[BR3]--[T10]--[SG3]
    """
    
    def __init__(self):
        self.nodes = self._define_nodes()
        self.edges = self._define_edges()
        self.node_ids = [n.node_id for n in self.nodes]
        self.node_to_idx = {n.node_id: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n.node_id for i, n in enumerate(self.nodes)}
    
    def _define_nodes(self) -> List[NodeDefinition]:
        """Define all nodes in the railway network"""
        nodes = [
            # Tracks - Main Line 1
            NodeDefinition("T01", "Track_1", ComponentType.TRACK, (0, 0)),
            NodeDefinition("T02", "Track_2", ComponentType.TRACK, (1, 0)),
            NodeDefinition("T03", "Track_3", ComponentType.TRACK, (2, 0)),
            NodeDefinition("T04", "Track_4", ComponentType.TRACK, (4, 0)),
            NodeDefinition("T05", "Track_5", ComponentType.TRACK, (5, 0)),
            # Tracks - Main Line 2
            NodeDefinition("T06", "Track_6", ComponentType.TRACK, (5, -1)),
            NodeDefinition("T07", "Track_7", ComponentType.TRACK, (2, -2)),
            NodeDefinition("T08", "Track_8", ComponentType.TRACK, (4, -2)),
            NodeDefinition("T09", "Track_9", ComponentType.TRACK, (5, -2)),
            NodeDefinition("T10", "Track_10", ComponentType.TRACK, (7, -2)),
            # Switches
            NodeDefinition("SW1", "Switch_1", ComponentType.SWITCH, (-1, 0)),
            NodeDefinition("SW2", "Switch_2", ComponentType.SWITCH, (3, -1)),
            NodeDefinition("SW3", "Switch_3", ComponentType.SWITCH, (6, 0)),
            NodeDefinition("SW4", "Switch_4", ComponentType.SWITCH, (3, -2)),
            # Signals
            NodeDefinition("SG1", "Signal_1", ComponentType.SIGNAL, (-2, 0)),
            NodeDefinition("SG2", "Signal_2", ComponentType.SIGNAL, (7, 0)),
            NodeDefinition("SG3", "Signal_3", ComponentType.SIGNAL, (8, -2)),
            # Bridges
            NodeDefinition("BR1", "Bridge_1", ComponentType.BRIDGE, (3, 0)),
            NodeDefinition("BR2", "Bridge_2", ComponentType.BRIDGE, (2, -1.5)),
            NodeDefinition("BR3", "Bridge_3", ComponentType.BRIDGE, (6, -2)),
        ]
        return nodes
    
    def _define_edges(self) -> List[EdgeDefinition]:
        """Define all edges in the railway network"""
        edges = []
        
        # Physical connections
        physical = [
            ("SG1", "SW1", 0.8), ("SW1", "T01", 1.0), ("T01", "T02", 1.0),
            ("T02", "T03", 1.0), ("T03", "BR1", 0.9), ("BR1", "T04", 0.9),
            ("T04", "T05", 1.0), ("T05", "SW3", 1.0), ("SW3", "SG2", 0.8),
            ("T03", "SW2", 0.7), ("T05", "T06", 0.8), ("SW2", "BR2", 0.9),
            ("T07", "BR2", 0.9), ("BR2", "SW4", 0.9), ("SW4", "T08", 1.0),
            ("T08", "T09", 1.0), ("T09", "BR3", 0.9), ("BR3", "T10", 0.9),
            ("T10", "SG3", 0.8), ("T06", "T09", 0.7),
        ]
        for src, tgt, w in physical:
            edges.append(EdgeDefinition(src, tgt, EdgeType.PHYSICAL, w))
        
        # Electrical connections
        electrical = [
            ("SG1", "SW1", 0.6), ("SG1", "T01", 0.5), ("SG2", "SW3", 0.6),
            ("SG2", "T05", 0.5), ("SG3", "T10", 0.5), ("SG3", "BR3", 0.4),
        ]
        for src, tgt, w in electrical:
            edges.append(EdgeDefinition(src, tgt, EdgeType.ELECTRICAL, w))
        
        # Load sharing
        load_sharing = [
            ("BR1", "T03", 0.7), ("BR1", "T04", 0.7), ("BR2", "T07", 0.7),
            ("BR2", "SW4", 0.6), ("BR3", "T09", 0.7), ("BR3", "T10", 0.7),
        ]
        for src, tgt, w in load_sharing:
            edges.append(EdgeDefinition(src, tgt, EdgeType.LOAD_SHARING, w))
        
        # Signal control
        signal_ctrl = [("SG1", "SW1", 0.9), ("SG2", "SW3", 0.9), ("SW2", "SG1", 0.3)]
        for src, tgt, w in signal_ctrl:
            edges.append(EdgeDefinition(src, tgt, EdgeType.SIGNAL_CONTROL, w))
        
        return edges
    
    def get_neighbors(self, node_id: str, edge_type: EdgeType = None) -> List[str]:
        """Get neighboring nodes, optionally filtered by edge type"""
        neighbors = set()
        for edge in self.edges:
            if edge_type and edge.edge_type != edge_type:
                continue
            if edge.source == node_id:
                neighbors.add(edge.target)
            elif edge.bidirectional and edge.target == node_id:
                neighbors.add(edge.source)
        return list(neighbors)
    
    def get_edge_weight(self, source: str, target: str, edge_type: EdgeType = None) -> float:
        """Get edge weight between two nodes"""
        for edge in self.edges:
            if edge_type and edge.edge_type != edge_type:
                continue
            if (edge.source == source and edge.target == target) or \
               (edge.bidirectional and edge.source == target and edge.target == source):
                return edge.weight
        return 0.0
    
    def get_component_type(self, node_id: str) -> ComponentType:
        """Get the component type for a node"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node.component_type
        return None
    
    def get_propagation_paths(self, source_node: str, max_hops: int = 3) -> Dict[str, List[str]]:
        """Find all propagation paths from a source node."""
        paths = {source_node: [source_node]}
        visited = {source_node}
        frontier = [source_node]
        
        for hop in range(max_hops):
            new_frontier = []
            for node in frontier:
                for neighbor in self.get_neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        paths[neighbor] = paths[node] + [neighbor]
                        new_frontier.append(neighbor)
            frontier = new_frontier
        return paths


DEFAULT_TOPOLOGY = RailwayTopology()