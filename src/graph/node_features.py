"""
Node Feature Management for Railway Graph.
Handles dynamic updates to node features based on sensor data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from .topology import RailwayTopology, DEFAULT_TOPOLOGY, ComponentType


@dataclass
class NodeFeatureConfig:
    """Configuration for node feature computation"""
    num_sensor_features: int = 4
    num_derived_features: int = 8
    num_component_features: int = 4
    rolling_window_short: int = 5
    rolling_window_medium: int = 20
    rolling_window_long: int = 50
    sensor_means: Dict[str, float] = field(default_factory=lambda: {
        "vibration": 0.5, "temperature": 38.0,
        "load": 1.0, "electrical_current": 0.55
    })
    sensor_stds: Dict[str, float] = field(default_factory=lambda: {
        "vibration": 0.2, "temperature": 6.0,
        "load": 0.15, "electrical_current": 0.15
    })


class NodeFeatureManager:
    """Manages node features for the railway graph."""
    
    def __init__(self, topology: RailwayTopology = None, config: NodeFeatureConfig = None):
        self.topology = topology or DEFAULT_TOPOLOGY
        self.config = config or NodeFeatureConfig()
        self.num_nodes = len(self.topology.nodes)
        self.node_ids = self.topology.node_ids
        self.history: Dict[str, Dict[str, deque]] = {}
        self._init_history()
        self.component_type_encoding = self._create_component_encoding()
        self.feature_dim = (self.config.num_sensor_features + 
                           self.config.num_derived_features + 
                           self.config.num_component_features)
        print(f"NodeFeatureManager: {self.num_nodes} nodes, {self.feature_dim} features")
    
    def _init_history(self):
        max_window = self.config.rolling_window_long
        sensors = ["vibration", "temperature", "load", "electrical_current"]
        for node_id in self.node_ids:
            self.history[node_id] = {s: deque(maxlen=max_window) for s in sensors}
    
    def _create_component_encoding(self) -> Dict[str, np.ndarray]:
        type_to_idx = {ComponentType.TRACK: 0, ComponentType.SWITCH: 1,
                       ComponentType.SIGNAL: 2, ComponentType.BRIDGE: 3}
        encodings = {}
        for node in self.topology.nodes:
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[type_to_idx[node.component_type]] = 1.0
            encodings[node.node_id] = one_hot
        return encodings
    
    def normalize_sensor(self, sensor: str, value: float) -> float:
        mean = self.config.sensor_means.get(sensor, 0.0)
        std = self.config.sensor_stds.get(sensor, 1.0)
        return (value - mean) / std
    
    def update_node(self, node_id: str, sensor_values: Dict[str, float]):
        if node_id not in self.history:
            return
        for sensor, value in sensor_values.items():
            if sensor in self.history[node_id]:
                self.history[node_id][sensor].append(value)
    
    def update_from_dataframe_row(self, row: pd.Series):
        node_id = row["component_id"]
        sensor_values = {"vibration": row["vibration"], "temperature": row["temperature"],
                        "load": row["load"], "electrical_current": row["electrical_current"]}
        self.update_node(node_id, sensor_values)
    
    def compute_rolling_stats(self, values: List[float], window: int) -> Tuple[float, float]:
        if len(values) < 2:
            return 0.0, 0.0
        window_vals = list(values)[-window:]
        mean = np.mean(window_vals)
        std = np.std(window_vals) if len(window_vals) > 1 else 0.0
        return mean, std
    
    def compute_trend(self, values: List[float], window: int = 10) -> float:
        if len(values) < 3:
            return 0.0
        window_vals = list(values)[-window:]
        if len(window_vals) < 3:
            return 0.0
        x = np.arange(len(window_vals))
        slope, _ = np.polyfit(x, window_vals, 1)
        return slope
    
    def get_node_features(self, node_id: str) -> np.ndarray:
        features = []
        sensors = ["vibration", "temperature", "load", "electrical_current"]
        
        for sensor in sensors:
            history = list(self.history[node_id][sensor])
            if history:
                normalized = self.normalize_sensor(sensor, history[-1])
            else:
                normalized = 0.0
            features.append(normalized)
        
        for sensor in ["vibration", "temperature"]:
            history = list(self.history[node_id][sensor])
            mean, std = self.compute_rolling_stats(history, self.config.rolling_window_short)
            features.append(self.normalize_sensor(sensor, mean))
            features.append(std / self.config.sensor_stds.get(sensor, 1.0))
        
        for sensor in ["vibration", "temperature"]:
            history = list(self.history[node_id][sensor])
            trend = self.compute_trend(history)
            features.append(trend * 10)
        
        features.extend(self.component_type_encoding[node_id])
        return np.array(features, dtype=np.float32)
    
    def get_all_node_features(self) -> np.ndarray:
        features = [self.get_node_features(nid) for nid in self.node_ids]
        return np.stack(features)
    
    def bulk_update_from_dataframe(self, df: pd.DataFrame, timestamp: str = None):
        if timestamp:
            df = df[df["timestamp"] == timestamp]
        for _, row in df.iterrows():
            self.update_from_dataframe_row(row)
    
    def get_feature_names(self) -> List[str]:
        return ["vibration_norm", "temperature_norm", "load_norm", "current_norm",
                "vibration_roll_mean", "vibration_roll_std", "temperature_roll_mean",
                "temperature_roll_std", "vibration_trend", "temperature_trend",
                "is_track", "is_switch", "is_signal", "is_bridge"]
    
    def reset(self):
        self._init_history()


class TemporalFeatureManager(NodeFeatureManager):
    """Extended manager with temporal sequence storage for TGNN."""
    
    def __init__(self, topology: RailwayTopology = None, config: NodeFeatureConfig = None,
                 sequence_length: int = 12):
        super().__init__(topology, config)
        self.sequence_length = sequence_length
        self.temporal_features: deque = deque(maxlen=sequence_length)
    
    def snapshot(self):
        current_features = self.get_all_node_features()
        self.temporal_features.append(current_features)
    
    def get_temporal_features(self) -> np.ndarray:
        if len(self.temporal_features) == 0:
            return np.zeros((self.sequence_length, self.num_nodes, self.feature_dim), dtype=np.float32)
        features = np.stack(list(self.temporal_features))
        if len(features) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(features), self.num_nodes, self.feature_dim), dtype=np.float32)
            features = np.concatenate([padding, features], axis=0)
        return features
    
    def reset(self):
        super().reset()
        self.temporal_features.clear()