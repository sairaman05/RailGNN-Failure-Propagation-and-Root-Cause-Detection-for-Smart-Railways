#!/usr/bin/env python3
"""
Test script for Phase 2: Graph Module
Run this after generating sensor data with:
    python -m src.data_generation.sensor_simulator
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Import graph modules
from src.graph.topology import RailwayTopology, EdgeType, ComponentType
from src.graph.node_features import NodeFeatureManager, TemporalFeatureManager
from src.graph.railway_graph import RailwayGraph, GraphSnapshot
from src.graph.visualize import print_graph_summary, save_graph_visualization


def main():
    print("=" * 60)
    print("PHASE 2: GRAPH MODULE TEST")
    print("=" * 60)
    
    # Test 1: Topology
    print(" [1] Testing Topology...")
    topology = RailwayTopology()
    print(f"    Nodes: {len(topology.nodes)}")
    print(f"    Edges: {len(topology.edges)}")
    
    # Show node types
    type_counts = {}
    for node in topology.nodes:
        t = node.component_type.value
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"    By type: {type_counts}")
    
    # Test propagation paths
    paths = topology.get_propagation_paths("T05", max_hops=2)
    print(f"    Propagation paths from T05: {len(paths)}")
    
    # Test 2: Node Features
    print(" [2] Testing Node Feature Manager...")
    feature_mgr = NodeFeatureManager(topology)
    print(f"    Feature dimension: {feature_mgr.feature_dim}")
    print(f"    Feature names: {feature_mgr.get_feature_names()}")
    
    # Test 3: Railway Graph
    print(" [3] Testing Railway Graph...")
    graph = RailwayGraph(use_temporal=True, sequence_length=12)
    print(f"    Nodes: {graph.num_nodes}")
    print(f"    Edges: {graph.num_edges}")
    print(f"    Edge index shape: {graph.edge_index.shape}")
    print(f"    Edge attr shape: {graph.edge_attr.shape}")
    
    # Test 4: Load data and simulate streaming
    print(" [4] Testing with sensor data...")
    data_path = os.path.join(os.path.dirname(__file__), "data", "raw", "railway_sensor_data.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"    Loaded {len(df)} records")
        
        timestamps = df["timestamp"].unique()[:20]
        print(f"    Simulating {len(timestamps)} time steps...")
        
        for ts in timestamps:
            graph.update_from_sensor_data(df, timestamp=ts)
        
        # Get snapshot
        snapshot = graph.get_snapshot(df, timestamp=timestamps[-1])
        print(f"    Snapshot node features: {snapshot.node_features.shape}")
        print(f"    Snapshot labels: {snapshot.node_labels}")
        
        # Get temporal features
        temporal_feat, edge_idx, edge_attr = graph.get_temporal_snapshot()
        print(f"    Temporal features shape: {temporal_feat.shape}")
        
        # Show sample features
        print(" Sample node features (T05):")
        t05_idx = graph.node_to_idx["T05"]
        features = snapshot.node_features[t05_idx]
        names = feature_mgr.get_feature_names()
        for name, val in zip(names[:6], features[:6]):
            print(f"      {name}: {val:.4f}")
        
        # Find high risk components
        print(" Risk levels at final timestamp:")
        risk_scores = graph.get_node_risk_scores(df, timestamp=timestamps[-1])
        high_risk = {k: v for k, v in risk_scores.items() if v >= 1}
        if high_risk:
            for node, risk in sorted(high_risk.items(), key=lambda x: -x[1])[:5]:
                print(f"      {node}: Risk Level {int(risk)}")
        else:
            print("      All components normal")
        
    else:
        print(f"    Data not found: {data_path}")
        print("    Run: python -m src.data_generation.sensor_simulator")
    
    # Test 5: Adjacency matrix
    print(" [5] Testing Adjacency Matrix...")
    adj = graph.get_adjacency_matrix(weighted=True)
    print(f"    Shape: {adj.shape}")
    print(f"    Non-zero entries: {np.count_nonzero(adj)}")
    print(f"    Density: {np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1]):.2%}")
    
    # Test 6: Save outputs
    print(" [6] Saving outputs...")
    output_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    viz_path = os.path.join(output_dir, "railway_graph.html")
    save_graph_visualization(viz_path, graph)
    
    # Save structure
    struct_path = os.path.join(output_dir, "graph_structure.json")
    graph.save_structure(struct_path)
    
    # Print summary
    print_graph_summary(graph)
    
    print(" " + "=" * 60)
    print("PHASE 2 COMPLETE!")
    print("=" * 60)
    print(" Next steps:")
    print("  1. View graph: Open data/processed/railway_graph.html in browser")
    print("  2. Continue to Phase 3: Kafka Streaming")

if __name__ == "__main__":
    main()