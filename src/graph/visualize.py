"""
Graph Visualization Utilities.
Provides functions to visualize the railway network graph.
"""

import numpy as np
from typing import Dict, List, Optional
import os

from .topology import RailwayTopology, DEFAULT_TOPOLOGY, EdgeType, ComponentType
from .railway_graph import RailwayGraph


def generate_graph_html(
    graph: RailwayGraph = None,
    risk_scores: Dict[str, float] = None,
    title: str = "Railway Network Graph"
) -> str:
    """Generate an interactive HTML visualization of the railway graph."""
    topology = graph.topology if graph else DEFAULT_TOPOLOGY
    
    type_colors = {
        ComponentType.TRACK: "#4A90D9",
        ComponentType.SWITCH: "#F5A623",
        ComponentType.SIGNAL: "#7ED321",
        ComponentType.BRIDGE: "#9B59B6"
    }
    
    risk_colors = {0: "#27AE60", 1: "#F1C40F", 2: "#E67E22", 3: "#E74C3C", 4: "#8E44AD"}
    
    min_x = min(n.position[0] for n in topology.nodes)
    max_x = max(n.position[0] for n in topology.nodes)
    min_y = min(n.position[1] for n in topology.nodes)
    max_y = max(n.position[1] for n in topology.nodes)
    
    width, height, padding = 900, 500, 80
    
    def scale_x(x):
        return padding + (x - min_x) / (max_x - min_x + 0.01) * (width - 2 * padding)
    
    def scale_y(y):
        return padding + (max_y - y) / (max_y - min_y + 0.01) * (height - 2 * padding)
    
    node_pos = {n.node_id: (scale_x(n.position[0]), scale_y(n.position[1])) for n in topology.nodes}
    
    svg_elements = []
    edge_colors = {EdgeType.PHYSICAL: "#666", EdgeType.ELECTRICAL: "#3498DB",
                   EdgeType.LOAD_SHARING: "#E74C3C", EdgeType.SIGNAL_CONTROL: "#27AE60"}
    
    for edge in topology.edges:
        x1, y1 = node_pos[edge.source]
        x2, y2 = node_pos[edge.target]
        color = edge_colors.get(edge.edge_type, "#999")
        svg_elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" opacity="0.6"/>')
    
    for node in topology.nodes:
        x, y = node_pos[node.node_id]
        fill = risk_colors.get(int(risk_scores.get(node.node_id, 0)), type_colors.get(node.component_type, "#95A5A6")) if risk_scores else type_colors.get(node.component_type, "#95A5A6")
        
        if node.component_type == ComponentType.SWITCH:
            svg_elements.append(f'<polygon points="{x},{y-18} {x+18},{y} {x},{y+18} {x-18},{y}" fill="{fill}" stroke="#333" stroke-width="2"/>')
        elif node.component_type == ComponentType.SIGNAL:
            svg_elements.append(f'<polygon points="{x},{y-16} {x+16},{y+16} {x-16},{y+16}" fill="{fill}" stroke="#333" stroke-width="2"/>')
        elif node.component_type == ComponentType.BRIDGE:
            svg_elements.append(f'<rect x="{x-18}" y="{y-12}" width="36" height="24" fill="{fill}" stroke="#333" stroke-width="2" rx="4"/>')
        else:
            svg_elements.append(f'<circle cx="{x}" cy="{y}" r="15" fill="{fill}" stroke="#333" stroke-width="2"/>')
        
        svg_elements.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" font-size="10" font-weight="bold" fill="white">{node.node_id}</text>')
    
    svg_content = "\n".join(svg_elements)
    
    return f'''<!DOCTYPE html>
<html><head><title>{title}</title>
<style>body{{font-family:Arial;margin:20px;background:#f5f5f5}}.container{{background:white;padding:20px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1)}}h1{{color:#333}}svg{{display:block;margin:0 auto}}.info{{margin-top:20px;padding:15px;background:#f0f0f0;border-radius:5px}}</style>
</head><body><div class="container"><h1>{title}</h1>
<svg width="{width}" height="{height}" style="border:1px solid #ddd;border-radius:5px"><rect width="100%" height="100%" fill="#fafafa"/>{svg_content}</svg>
<div class="info"><strong>Nodes:</strong> {len(topology.nodes)} | <strong>Edges:</strong> {len(topology.edges)}</div></div></body></html>'''


def save_graph_visualization(output_path: str, graph: RailwayGraph = None, risk_scores: Dict[str, float] = None):
    """Save graph visualization to HTML file"""
    html = generate_graph_html(graph, risk_scores)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Visualization saved to {output_path}")


def print_graph_summary(graph: RailwayGraph = None):
    """Print summary of graph structure"""
    topology = graph.topology if graph else DEFAULT_TOPOLOGY
    
    print("\n" + "=" * 60)
    print("RAILWAY GRAPH SUMMARY")
    print("=" * 60)
    print(f"\nNodes: {len(topology.nodes)}")
    
    by_type = {}
    for node in topology.nodes:
        t = node.component_type.value
        by_type[t] = by_type.get(t, 0) + 1
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")
    
    print(f"\nEdges: {len(topology.edges)}")
    edge_types = {}
    for edge in topology.edges:
        t = edge.edge_type.value
        edge_types[t] = edge_types.get(t, 0) + 1
    for t, count in sorted(edge_types.items()):
        print(f"  {t}: {count}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    graph = RailwayGraph(use_temporal=False)
    print_graph_summary(graph)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    save_graph_visualization(os.path.join(output_dir, "railway_graph.html"), graph)