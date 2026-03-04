"""
Degradation Pattern Generator for Railway Components.
Defines realistic failure patterns and propagation behaviors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .config import FaultType, RiskLevel, SensorRanges


@dataclass
class DegradationEvent:
    """Represents a single degradation event"""
    component_id: str
    start_row: int
    end_row: int
    fault_type: FaultType
    severity_curve: str  # linear, exponential, sudden, oscillating
    max_severity: float  # 0.0 to 1.0
    affected_sensors: List[str]


class DegradationPatternGenerator:
    """Generates realistic degradation patterns for railway components"""
    
    def __init__(self, sensor_ranges: SensorRanges, random_seed: int = 42):
        self.sensor_ranges = sensor_ranges
        self.rng = np.random.default_rng(random_seed)
        self.degradation_events = self._define_degradation_events()
    
    def _define_degradation_events(self) -> List[DegradationEvent]:
        """Define all degradation events across phases"""
        events = []
        
        # Phase 2: Track_5 gradual degradation (30k-50k)
        events.append(DegradationEvent(
            component_id="T05",
            start_row=30_000,
            end_row=50_000,
            fault_type=FaultType.MECHANICAL_WEAR,
            severity_curve="exponential",
            max_severity=0.8,
            affected_sensors=["vibration", "temperature"]
        ))
        
        # Phase 3: Propagation to connected components (50k-60k)
        events.append(DegradationEvent(
            component_id="SW3",
            start_row=50_500,
            end_row=60_000,
            fault_type=FaultType.PROPAGATED,
            severity_curve="linear",
            max_severity=0.5,
            affected_sensors=["vibration", "current"]
        ))
        
        events.append(DegradationEvent(
            component_id="SG2",
            start_row=51_000,
            end_row=60_000,
            fault_type=FaultType.PROPAGATED,
            severity_curve="linear",
            max_severity=0.4,
            affected_sensors=["current", "temperature"]
        ))
        
        events.append(DegradationEvent(
            component_id="T04",
            start_row=52_000,
            end_row=58_000,
            fault_type=FaultType.PROPAGATED,
            severity_curve="linear",
            max_severity=0.3,
            affected_sensors=["vibration"]
        ))
        
        events.append(DegradationEvent(
            component_id="T06",
            start_row=52_000,
            end_row=58_000,
            fault_type=FaultType.PROPAGATED,
            severity_curve="linear",
            max_severity=0.3,
            affected_sensors=["vibration"]
        ))
        
        # Phase 5: New degradation pattern (80k-100k)
        events.append(DegradationEvent(
            component_id="BR2",
            start_row=80_000,
            end_row=95_000,
            fault_type=FaultType.OVERHEATING,
            severity_curve="exponential",
            max_severity=0.7,
            affected_sensors=["temperature", "load"]
        ))
        
        events.append(DegradationEvent(
            component_id="SW4",
            start_row=85_000,
            end_row=98_000,
            fault_type=FaultType.ELECTRICAL_FAULT,
            severity_curve="oscillating",
            max_severity=0.6,
            affected_sensors=["current", "temperature"]
        ))
        
        events.append(DegradationEvent(
            component_id="T07",
            start_row=87_000,
            end_row=100_000,
            fault_type=FaultType.PROPAGATED,
            severity_curve="linear",
            max_severity=0.4,
            affected_sensors=["vibration", "load"]
        ))
        
        return events
    
    def get_severity(self, event: DegradationEvent, row_idx: int) -> float:
        """Calculate severity for a given row based on degradation curve"""
        if row_idx < event.start_row or row_idx > event.end_row:
            return 0.0
        
        progress = (row_idx - event.start_row) / (event.end_row - event.start_row)
        
        if event.severity_curve == "linear":
            return progress * event.max_severity
        elif event.severity_curve == "exponential":
            return (np.exp(progress * 3) - 1) / (np.exp(3) - 1) * event.max_severity
        elif event.severity_curve == "sudden":
            return event.max_severity if progress > 0.7 else progress * 0.2 * event.max_severity
        elif event.severity_curve == "oscillating":
            base = progress * event.max_severity
            oscillation = 0.2 * np.sin(progress * 20 * np.pi) * event.max_severity
            return max(0, base + oscillation)
        return 0.0
    
    def get_sensor_modifier(self, component_id: str, sensor_type: str, row_idx: int) -> Tuple[float, FaultType]:
        """Get the sensor value modifier for a component at a given row."""
        total_modifier = 0.0
        active_fault = FaultType.NONE
        
        for event in self.degradation_events:
            if event.component_id != component_id:
                continue
            if sensor_type not in event.affected_sensors:
                continue
            
            severity = self.get_severity(event, row_idx)
            if severity > 0:
                modifier = self._severity_to_modifier(sensor_type, severity)
                total_modifier += modifier
                active_fault = event.fault_type
        
        return total_modifier, active_fault
    
    def _severity_to_modifier(self, sensor_type: str, severity: float) -> float:
        """Convert severity (0-1) to actual sensor value modifier"""
        modifiers = {
            "vibration": severity * 0.8,
            "temperature": severity * 25.0,
            "load": severity * 0.5,
            "current": severity * 0.4
        }
        return modifiers.get(sensor_type, 0.0)
    
    def calculate_risk_label(self, vibration: float, temperature: float, load: float, current: float) -> int:
        """Calculate risk label based on sensor values"""
        risk_score = 0
        
        if vibration > self.sensor_ranges.vibration_critical[0]:
            risk_score += 2
        elif vibration > self.sensor_ranges.vibration_warning[0]:
            risk_score += 1
        
        if temperature > self.sensor_ranges.temperature_critical[0]:
            risk_score += 2
        elif temperature > self.sensor_ranges.temperature_warning[0]:
            risk_score += 1
        
        if load > self.sensor_ranges.load_critical[0]:
            risk_score += 2
        elif load > self.sensor_ranges.load_warning[0]:
            risk_score += 1
        
        if current > self.sensor_ranges.current_critical[0]:
            risk_score += 2
        elif current > self.sensor_ranges.current_warning[0]:
            risk_score += 1
        
        if risk_score >= 6:
            return RiskLevel.CRITICAL.value
        elif risk_score >= 4:
            return RiskLevel.HIGH.value
        elif risk_score >= 2:
            return RiskLevel.MEDIUM.value
        elif risk_score >= 1:
            return RiskLevel.LOW.value
        return RiskLevel.NORMAL.value
    
    def get_active_events(self, row_idx: int) -> List[DegradationEvent]:
        """Get all active degradation events for a given row"""
        return [e for e in self.degradation_events if e.start_row <= row_idx <= e.end_row]
    
    def get_root_cause_component(self, row_idx: int) -> Optional[str]:
        """Identify the root cause component for failures at a given row."""
        active_events = self.get_active_events(row_idx)
        original_faults = [e for e in active_events if e.fault_type != FaultType.PROPAGATED]
        
        if original_faults:
            return min(original_faults, key=lambda e: e.start_row).component_id
        return None