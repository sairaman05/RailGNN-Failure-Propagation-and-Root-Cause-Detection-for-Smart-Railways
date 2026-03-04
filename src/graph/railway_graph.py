"""
Railway Graph Construction and Management.
Creates graph representations compatible with NetworkX and PyTorch Geometric.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("NetworkX not available. Some features will be limited.")

from .topology import RailwayTopology, DEFAULT_TOPOLOGY, EdgeType, ComponentType
from .node_features import NodeFeatureManager, TemporalFeatureManager, NodeFeatureConfig


@dataclass
class GraphSnapshot:
    """A snapshot of the graph state at a point in time"""
    timestamp: str
    node_features: np.ndarray  # (num_nodes, feature_dim)
    edge_index: np.ndarray     # (2, num_edges)
    edge_attr: np.ndarray      # (num_edges, edge_feature_dim)
    node_labels: np.ndarray    # (num_nodes,) risk labels
    node_ids: List[str]


class RailwayGraph:
    """
    Main class for railway graph construction and management.
    
    Provides:
    - NetworkX graph for visualization and analysis
    - PyTorch Geometric compatible tensors for GNN training
    - Dynamic feature updates from sensor data
    """
    
    def __init__(
        self,
        topology: RailwayTopology = None,
        feature_config: NodeFeatureConfig = None,
        use_temporal: bool = True,
        sequence_length: int = 12
    ):
        self.topology = topology or DEFAULT_TOPOLOGY
        
        # Initialize feature manager
        if use_temporal:
            self.feature_manager = TemporalFeatureManager(
                topology=self.topology,
                config=feature_config,
                sequence_length=sequence_length
            )
        else:
            self.feature_manager = NodeFeatureManager(
                topology=self.topology,
                config=feature_config
            )
        
        # Build edge structures
        self.edge_index, self.edge_attr, self.edge_types = self._build_edge_index()
        
        # Node mappings
        self.node_ids = self.topology.node_ids
        self.node_to_idx = self.topology.node_to_idx
        self.idx_to_node = self.topology.idx_to_node
        self.num_nodes = len(self.node_ids)
        self.num_edges = self.edge_index.shape[1]
        
        # Build NetworkX graph if available
        self.nx_graph = self._build_networkx_graph() if HAS_NETWORKX else None
        
        print(f"RailwayGraph initialized:")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Edges: {self.num_edges}")
        print(f"  Feature dim: {self.feature_manager.feature_dim}")
    
    def _build_edge_index(self) -> Tuple[np.ndarray, np.ndarray, List[EdgeType]]:
        """
        Build edge index in COO format for PyTorch Geometric.
        
        Returns:
            edge_index: (2, num_edges) source and target node indices
            edge_attr: (num_edges, 2) edge weight and type encoding
            edge_types: List of edge types
        """
        sources = []
        targets = []
        weights = []
        edge_types = []
        
        # Edge type encoding
        type_to_idx = {
            EdgeType.PHYSICAL: 0,
            EdgeType.ELECTRICAL: 1,
            EdgeType.LOAD_SHARING: 2,
            EdgeType.SIGNAL_CONTROL: 3
        }
        
        for edge in self.topology.edges:
            src_idx = self.topology.node_to_idx[edge.source]
            tgt_idx = self.topology.node_to_idx[edge.target]
            
            # Add forward edge
            sources.append(src_idx)
            targets.append(tgt_idx)
            weights.append(edge.weight)
            edge_types.append(edge.edge_type)
            
            # Add reverse edge if bidirectional
            if edge.bidirectional:
                sources.append(tgt_idx)
                targets.append(src_idx)
                weights.append(edge.weight)
                edge_types.append(edge.edge_type)
        
        edge_index = np.array([sources, targets], dtype=np.int64)
        
        # Edge attributes: [weight, type_encoding]
        edge_attr = np.zeros((len(sources), 2), dtype=np.float32)
        edge_attr[:, 0] = weights
        edge_attr[:, 1] = [type_to_idx[et] for et in edge_types]
        
        return edge_index, edge_attr, edge_types
    
    def _build_networkx_graph(self):
        """Build NetworkX graph for visualization and analysis"""
        if not HAS_NETWORKX:
            return None
        
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in self.topology.nodes:
            G.add_node(
                node.node_id,
                name=node.name,
                component_type=node.component_type.value,
                pos=node.position
            )
        
        # Add edges with attributes
        for edge in self.topology.edges:
            G.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
                edge_type=edge.edge_type.value
            )
            if edge.bidirectional:
                G.add_edge(
                    edge.target,
                    edge.source,
                    weight=edge.weight,
                    edge_type=edge.edge_type.value
                )
        
        return G
    
    def update_from_sensor_data(self, df: pd.DataFrame, timestamp: str = None):
        """
        Update node features from sensor data.
        
        Args:
            df: DataFrame with sensor readings
            timestamp: Specific timestamp to use (if None, uses all data)
        """
        self.feature_manager.bulk_update_from_dataframe(df, timestamp)
        
        # Take snapshot for temporal features
        if isinstance(self.feature_manager, TemporalFeatureManager):
            self.feature_manager.snapshot()
    
    def get_snapshot(self, df: pd.DataFrame = None, timestamp: str = None) -> GraphSnapshot:
        """
        Get current graph snapshot with all features.
        
        Args:
            df: Optional DataFrame to extract labels from
            timestamp: Timestamp to filter df
        
        Returns:
            GraphSnapshot with node features, edges, and labels
        """
        # Get node features
        if isinstance(self.feature_manager, TemporalFeatureManager):
            # For temporal, use the current (last) features
            node_features = self.feature_manager.get_all_node_features()
        else:
            node_features = self.feature_manager.get_all_node_features()
        
        # Get labels if df provided
        if df is not None:
            if timestamp:
                df_t = df[df["timestamp"] == timestamp]
            else:
                # Use latest timestamp
                df_t = df[df["timestamp"] == df["timestamp"].max()]
            
            node_labels = np.zeros(self.num_nodes, dtype=np.int64)
            for _, row in df_t.iterrows():
                if row["component_id"] in self.node_to_idx:
                    idx = self.node_to_idx[row["component_id"]]
                    node_labels[idx] = row["risk_label"]
        else:
            node_labels = np.zeros(self.num_nodes, dtype=np.int64)
        
        return GraphSnapshot(
            timestamp=timestamp or "",
            node_features=node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_labels=node_labels,
            node_ids=self.node_ids
        )
    
    def get_temporal_snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get temporal features for sequence modeling.
        
        Returns:
            temporal_features: (seq_len, num_nodes, feature_dim)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_feature_dim)
        """
        if not isinstance(self.feature_manager, TemporalFeatureManager):
            raise ValueError("Temporal features require TemporalFeatureManager")
        
        temporal_features = self.feature_manager.get_temporal_features()
        return temporal_features, self.edge_index, self.edge_attr
    
    def get_adjacency_matrix(self, weighted: bool = True) -> np.ndarray:
        """Get adjacency matrix representation"""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        for i in range(self.edge_index.shape[1]):
            src, tgt = self.edge_index[:, i]
            if weighted:
                adj[src, tgt] = self.edge_attr[i, 0]  # Use weight
            else:
                adj[src, tgt] = 1.0
        
        return adj
    
    def get_node_risk_scores(self, df: pd.DataFrame, timestamp: str = None) -> Dict[str, float]:
        """Get risk scores for all nodes at a timestamp"""
        if timestamp:
            df_t = df[df["timestamp"] == timestamp]
        else:
            df_t = df[df["timestamp"] == df["timestamp"].max()]
        
        risk_scores = {}
        for _, row in df_t.iterrows():
            risk_scores[row["component_id"]] = row["risk_label"]
        
        return risk_scores
    
    def find_high_risk_paths(self, df: pd.DataFrame, timestamp: str = None, threshold: int = 2) -> List[List[str]]:
        """Find paths connecting high-risk components"""
        risk_scores = self.get_node_risk_scores(df, timestamp)
        high_risk_nodes = [n for n, r in risk_scores.items() if r >= threshold]
        
        paths = []
        for node in high_risk_nodes:
            propagation_paths = self.topology.get_propagation_paths(node, max_hops=2)
            for target, path in propagation_paths.items():
                if target in high_risk_nodes and target != node:
                    paths.append(path)
        
        return paths
    
    def reset(self):
        """Reset all dynamic state"""
        self.feature_manager.reset()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph structure to dictionary"""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_ids": self.node_ids,
            "edge_index": self.edge_index.tolist(),
            "edge_attr": self.edge_attr.tolist(),
            "feature_dim": self.feature_manager.feature_dim,
            "feature_names": self.feature_manager.get_feature_names()
        }
    
    def save_structure(self, filepath: str):
        """Save graph structure to JSON"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Graph structure saved to {filepath}")


def create_pyg_data(snapshot: GraphSnapshot):
    """
    Convert GraphSnapshot to PyTorch Geometric Data object.
    Requires torch and torch_geometric to be installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
        
        data = Data(
            x=torch.tensor(snapshot.node_features, dtype=torch.float),
            edge_index=torch.tensor(snapshot.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(snapshot.edge_attr, dtype=torch.float),
            y=torch.tensor(snapshot.node_labels, dtype=torch.long)
        )
        return data
    except ImportError:
        print("PyTorch Geometric not available")
        return None


def demo_graph():
    """Demonstrate graph functionality"""
    print("=" * 60)
    print("RAILWAY GRAPH DEMO")
    print("=" * 60)
    
    # Create graph
    graph = RailwayGraph(use_temporal=True, sequence_length=12)
    
    # Load sensor data
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "railway_sensor_data.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"\nLoaded {len(df)} sensor records")
        
        # Get unique timestamps
        timestamps = df["timestamp"].unique()[:20]
        
        # Simulate streaming updates
        print("\nSimulating streaming updates...")
        for ts in timestamps[:10]:
            graph.update_from_sensor_data(df, timestamp=ts)
        
        # Get snapshot
        snapshot = graph.get_snapshot(df, timestamp=timestamps[9])
        print(f"\nSnapshot at {timestamps[9]}:")
        print(f"  Node features shape: {snapshot.node_features.shape}")
        print(f"  Edge index shape: {snapshot.edge_index.shape}")
        print(f"  Labels: {snapshot.node_labels}")
        
        # Get temporal features
        temporal_features, edge_index, edge_attr = graph.get_temporal_snapshot()
        print(f"\nTemporal features shape: {temporal_features.shape}")
        
        # Find high risk paths
        paths = graph.find_high_risk_paths(df, timestamp=timestamps[9])
        print(f"\nHigh risk paths: {len(paths)}")
        
    else:
        print(f"Data file not found: {data_path}")
        print("Run data generation first: python -m src.data_generation.sensor_simulator")
    
    # Print adjacency matrix
    adj = graph.get_adjacency_matrix()
    print(f"\nAdjacency matrix shape: {adj.shape}")
    print(f"Non-zero entries: {np.count_nonzero(adj)}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_graph()