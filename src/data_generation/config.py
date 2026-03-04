"""
Configuration for sensor data generation.
Defines component types, sensor ranges, and simulation parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


class ComponentType(Enum):
    """Types of railway components"""
    TRACK = "track"
    SWITCH = "switch"
    SIGNAL = "signal"
    BRIDGE = "bridge"


class FaultType(Enum):
    """Types of faults that can occur"""
    NONE = "none"
    VIBRATION_ANOMALY = "vibration_anomaly"
    OVERHEATING = "overheating"
    OVERLOAD = "overload"
    ELECTRICAL_FAULT = "electrical_fault"
    MECHANICAL_WEAR = "mechanical_wear"
    PROPAGATED = "propagated"


class RiskLevel(Enum):
    """Risk classification levels"""
    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SensorRanges:
    """Defines normal, warning, and critical ranges for each sensor type"""
    vibration_normal: Tuple[float, float] = (0.2, 0.8)
    vibration_warning: Tuple[float, float] = (0.8, 1.2)
    vibration_critical: Tuple[float, float] = (1.2, 2.5)
    
    temperature_normal: Tuple[float, float] = (30.0, 45.0)
    temperature_warning: Tuple[float, float] = (45.0, 60.0)
    temperature_critical: Tuple[float, float] = (60.0, 85.0)
    
    load_normal: Tuple[float, float] = (0.8, 1.2)
    load_warning: Tuple[float, float] = (1.2, 1.5)
    load_critical: Tuple[float, float] = (1.5, 2.0)
    
    current_normal: Tuple[float, float] = (0.3, 0.7)
    current_warning: Tuple[float, float] = (0.7, 0.9)
    current_critical: Tuple[float, float] = (0.9, 1.5)


@dataclass
class ComponentConfig:
    """Configuration for railway components"""
    
    components: Dict[str, Tuple[ComponentType, str]] = field(default_factory=lambda: {
        "T01": (ComponentType.TRACK, "Track_1"),
        "T02": (ComponentType.TRACK, "Track_2"),
        "T03": (ComponentType.TRACK, "Track_3"),
        "T04": (ComponentType.TRACK, "Track_4"),
        "T05": (ComponentType.TRACK, "Track_5"),
        "T06": (ComponentType.TRACK, "Track_6"),
        "T07": (ComponentType.TRACK, "Track_7"),
        "T08": (ComponentType.TRACK, "Track_8"),
        "T09": (ComponentType.TRACK, "Track_9"),
        "T10": (ComponentType.TRACK, "Track_10"),
        "SW1": (ComponentType.SWITCH, "Switch_1"),
        "SW2": (ComponentType.SWITCH, "Switch_2"),
        "SW3": (ComponentType.SWITCH, "Switch_3"),
        "SW4": (ComponentType.SWITCH, "Switch_4"),
        "SG1": (ComponentType.SIGNAL, "Signal_1"),
        "SG2": (ComponentType.SIGNAL, "Signal_2"),
        "SG3": (ComponentType.SIGNAL, "Signal_3"),
        "BR1": (ComponentType.BRIDGE, "Bridge_1"),
        "BR2": (ComponentType.BRIDGE, "Bridge_2"),
        "BR3": (ComponentType.BRIDGE, "Bridge_3"),
    })
    
    adjacency: Dict[str, List[str]] = field(default_factory=lambda: {
        "T01": ["T02", "SW1", "SG1"],
        "T02": ["T01", "T03", "SW1"],
        "T03": ["T02", "T04", "SW2", "BR1"],
        "T04": ["T03", "T05", "SW2"],
        "T05": ["T04", "T06", "SW3", "SG2"],
        "T06": ["T05", "T07", "SW3"],
        "T07": ["T06", "T08", "SW4", "BR2"],
        "T08": ["T07", "T09", "SW4"],
        "T09": ["T08", "T10", "BR3"],
        "T10": ["T09", "SG3"],
        "SW1": ["T01", "T02", "SG1"],
        "SW2": ["T03", "T04", "BR1"],
        "SW3": ["T05", "T06", "SG2"],
        "SW4": ["T07", "T08", "BR2"],
        "SG1": ["T01", "SW1"],
        "SG2": ["T05", "SW3"],
        "SG3": ["T10", "BR3"],
        "BR1": ["T03", "SW2"],
        "BR2": ["T07", "SW4"],
        "BR3": ["T09", "SG3"],
    })
    
    baseline_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "BR1": {"vibration": 1.1, "temperature": 1.0, "load": 1.15, "current": 1.0},
        "BR2": {"vibration": 1.15, "temperature": 1.0, "load": 1.2, "current": 1.0},
        "BR3": {"vibration": 1.1, "temperature": 1.0, "load": 1.1, "current": 1.0},
        "SW1": {"vibration": 1.0, "temperature": 1.1, "load": 1.0, "current": 1.2},
        "SW2": {"vibration": 1.0, "temperature": 1.15, "load": 1.0, "current": 1.25},
        "SW3": {"vibration": 1.0, "temperature": 1.1, "load": 1.0, "current": 1.2},
        "SW4": {"vibration": 1.0, "temperature": 1.1, "load": 1.0, "current": 1.15},
        "SG1": {"vibration": 0.8, "temperature": 1.05, "load": 0.9, "current": 1.3},
        "SG2": {"vibration": 0.8, "temperature": 1.1, "load": 0.9, "current": 1.35},
        "SG3": {"vibration": 0.8, "temperature": 1.05, "load": 0.9, "current": 1.3},
    })


@dataclass
class SensorConfig:
    """Main configuration for sensor simulation"""
    total_records: int = 100_000
    sensor_ranges: SensorRanges = field(default_factory=SensorRanges)
    component_config: ComponentConfig = field(default_factory=ComponentConfig)
    
    phases: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "healthy": (0, 30_000),
        "degradation_start": (30_000, 50_000),
        "propagation": (50_000, 60_000),
        "stabilization": (60_000, 80_000),
        "new_degradation": (80_000, 100_000),
    })
    
    noise_std: float = 0.05
    start_timestamp: str = "2024-01-01 00:00:00"
    time_interval_seconds: int = 30
    degradation_rate: float = 0.0001
    propagation_delay: int = 500
    propagation_factor: float = 0.6
    random_seed: int = 42


DEFAULT_SENSOR_CONFIG = SensorConfig()
DEFAULT_COMPONENT_CONFIG = ComponentConfig()
DEFAULT_SENSOR_RANGES = SensorRanges()