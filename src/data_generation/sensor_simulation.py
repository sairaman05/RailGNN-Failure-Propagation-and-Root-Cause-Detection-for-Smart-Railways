"""
Railway Sensor Data Simulator.
Generates 100,000 realistic sensor records with degradation patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os
import json

from .config import (
    SensorConfig, 
    FaultType,
    DEFAULT_SENSOR_CONFIG
)
from .degradation_patterns import DegradationPatternGenerator


class RailwaySensorSimulator:
    """
    Generates realistic railway sensor data with:
    - Normal operating conditions
    - Gradual degradation patterns
    - Failure propagation across connected components
    - Realistic noise and variations
    """
    
    def __init__(self, config: Optional[SensorConfig] = None):
        self.config = config or DEFAULT_SENSOR_CONFIG
        self.rng = np.random.default_rng(self.config.random_seed)
        
        self.degradation_generator = DegradationPatternGenerator(
            sensor_ranges=self.config.sensor_ranges,
            random_seed=self.config.random_seed
        )
        
        self.components = list(self.config.component_config.components.keys())
        self.num_components = len(self.components)
        
        print(f"Initialized simulator with {self.num_components} components")
        print(f"Components: {self.components}")
    
    def _generate_base_value(self, sensor_type: str, component_id: str) -> float:
        """Generate a base sensor value within normal range"""
        ranges = self.config.sensor_ranges
        multipliers = self.config.component_config.baseline_multipliers
        
        range_map = {
            "vibration": ranges.vibration_normal,
            "temperature": ranges.temperature_normal,
            "load": ranges.load_normal,
            "current": ranges.current_normal
        }
        
        low, high = range_map[sensor_type]
        base = self.rng.uniform(low, high)
        
        if component_id in multipliers:
            mult = multipliers[component_id].get(sensor_type, 1.0)
            base *= mult
        
        return base
    
    def _add_noise(self, value: float, sensor_type: str) -> float:
        """Add realistic noise to sensor value"""
        noise_scales = {
            "vibration": 0.05,
            "temperature": 1.0,
            "load": 0.03,
            "current": 0.02
        }
        scale = noise_scales.get(sensor_type, 0.05)
        noise = self.rng.normal(0, scale * self.config.noise_std * 10)
        return value + noise
    
    def _add_temporal_variation(self, value: float, row_idx: int, sensor_type: str) -> float:
        """Add time-based variations (daily cycles, etc.)"""
        if sensor_type == "temperature":
            hour = (row_idx * self.config.time_interval_seconds / 3600) % 24
            daily_variation = 3 * np.sin(2 * np.pi * hour / 24)
            value += daily_variation
        elif sensor_type == "load":
            hour = (row_idx * self.config.time_interval_seconds / 3600) % 24
            if 6 <= hour <= 9 or 16 <= hour <= 19:
                value *= 1.1
        return value
    
    def generate_record(self, row_idx: int, component_id: str, timestamp: datetime) -> Dict:
        """Generate a single sensor record"""
        
        vibration = self._generate_base_value("vibration", component_id)
        temperature = self._generate_base_value("temperature", component_id)
        load = self._generate_base_value("load", component_id)
        current = self._generate_base_value("current", component_id)
        
        vibration = self._add_temporal_variation(vibration, row_idx, "vibration")
        temperature = self._add_temporal_variation(temperature, row_idx, "temperature")
        load = self._add_temporal_variation(load, row_idx, "load")
        current = self._add_temporal_variation(current, row_idx, "current")
        
        fault_type = FaultType.NONE
        
        vib_mod, vib_fault = self.degradation_generator.get_sensor_modifier(component_id, "vibration", row_idx)
        temp_mod, temp_fault = self.degradation_generator.get_sensor_modifier(component_id, "temperature", row_idx)
        load_mod, load_fault = self.degradation_generator.get_sensor_modifier(component_id, "load", row_idx)
        curr_mod, curr_fault = self.degradation_generator.get_sensor_modifier(component_id, "current", row_idx)
        
        vibration += vib_mod
        temperature += temp_mod
        load += load_mod
        current += curr_mod
        
        for ft in [vib_fault, temp_fault, load_fault, curr_fault]:
            if ft != FaultType.NONE:
                fault_type = ft
                break
        
        vibration = self._add_noise(vibration, "vibration")
        temperature = self._add_noise(temperature, "temperature")
        load = self._add_noise(load, "load")
        current = self._add_noise(current, "current")
        
        vibration = max(0.1, vibration)
        temperature = max(20.0, temperature)
        load = max(0.5, load)
        current = max(0.1, current)
        
        risk_label = self.degradation_generator.calculate_risk_label(vibration, temperature, load, current)
        
        comp_type, comp_name = self.config.component_config.components[component_id]
        
        return {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "component_id": component_id,
            "component_name": comp_name,
            "component_type": comp_type.value,
            "vibration": round(vibration, 4),
            "temperature": round(temperature, 2),
            "load": round(load, 4),
            "electrical_current": round(current, 4),
            "risk_label": risk_label,
            "fault_type": fault_type.value
        }
    
    def generate_dataset(self, output_path: Optional[str] = None, show_progress: bool = True) -> pd.DataFrame:
        """Generate the complete dataset with 100,000 records."""
        records = []
        records_per_component = self.config.total_records // self.num_components
        
        start_time = datetime.strptime(self.config.start_timestamp, "%Y-%m-%d %H:%M:%S")
        
        print(f"\nGenerating {self.config.total_records} sensor records...")
        print(f"Records per component: {records_per_component}")
        print(f"Time span: {records_per_component * self.config.time_interval_seconds / 3600:.1f} hours")
        
        iterator = range(records_per_component)
        progress_interval = max(1, records_per_component // 20)
        
        for time_idx in iterator:
            if show_progress and time_idx % progress_interval == 0:
                pct = (time_idx / records_per_component) * 100
                sys.stdout.write(f"\rProgress: {pct:.0f}%")
                sys.stdout.flush()
            
            timestamp = start_time + timedelta(seconds=time_idx * self.config.time_interval_seconds)
            
            for comp_idx, component_id in enumerate(self.components):
                row_idx = time_idx * self.num_components + comp_idx
                record = self.generate_record(row_idx, component_id, timestamp)
                records.append(record)
        
        if show_progress:
            print("\rProgress: 100%")
        
        df = pd.DataFrame(records)
        df = df.sort_values(["timestamp", "component_id"]).reset_index(drop=True)
        df["record_id"] = range(len(df))
        
        columns = [
            "record_id", "timestamp", "component_id", "component_name",
            "component_type", "vibration", "temperature", "load",
            "electrical_current", "risk_label", "fault_type"
        ]
        df = df[columns]
        
        print(f"\nGenerated {len(df)} records")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
        
        return df
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the dataset"""
        summary = {
            "total_records": len(df),
            "unique_components": int(df["component_id"].nunique()),
            "time_range": {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max())
            },
            "sensor_stats": {
                "vibration": {
                    "min": float(df["vibration"].min()),
                    "max": float(df["vibration"].max()),
                    "mean": float(df["vibration"].mean()),
                    "std": float(df["vibration"].std())
                },
                "temperature": {
                    "min": float(df["temperature"].min()),
                    "max": float(df["temperature"].max()),
                    "mean": float(df["temperature"].mean()),
                    "std": float(df["temperature"].std())
                },
                "load": {
                    "min": float(df["load"].min()),
                    "max": float(df["load"].max()),
                    "mean": float(df["load"].mean()),
                    "std": float(df["load"].std())
                },
                "electrical_current": {
                    "min": float(df["electrical_current"].min()),
                    "max": float(df["electrical_current"].max()),
                    "mean": float(df["electrical_current"].mean()),
                    "std": float(df["electrical_current"].std())
                }
            },
            "risk_distribution": {int(k): int(v) for k, v in df["risk_label"].value_counts().to_dict().items()},
            "fault_distribution": {str(k): int(v) for k, v in df["fault_type"].value_counts().to_dict().items()}
        }
        return summary
    
    def print_summary(self, df: pd.DataFrame):
        """Print formatted summary of the dataset"""
        summary = self.generate_summary(df)
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        print(f"\nTotal Records: {summary['total_records']:,}")
        print(f"Unique Components: {summary['unique_components']}")
        print(f"Time Range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        
        print("\n--- Sensor Statistics ---")
        for sensor, stats in summary["sensor_stats"].items():
            print(f"\n{sensor.upper()}:")
            print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        
        print("\n--- Risk Distribution ---")
        risk_names = {0: "NORMAL", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
        for risk, count in sorted(summary["risk_distribution"].items()):
            pct = count / summary["total_records"] * 100
            print(f"  {risk_names.get(risk, risk)}: {count:,} ({pct:.2f}%)")
        
        print("\n--- Fault Distribution ---")
        for fault, count in summary["fault_distribution"].items():
            pct = count / summary["total_records"] * 100
            print(f"  {fault}: {count:,} ({pct:.2f}%)")
        
        print("\n" + "="*60)


def main():
    """Main function to generate the dataset"""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "railway_sensor_data.csv")
    
    simulator = RailwaySensorSimulator()
    df = simulator.generate_dataset(output_path=output_path)
    simulator.print_summary(df)
    
    summary = simulator.generate_summary(df)
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    return df


if __name__ == "__main__":
    main()