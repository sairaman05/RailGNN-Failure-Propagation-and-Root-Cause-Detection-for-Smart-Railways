"""
Rolling Feature Engineering for Streaming Records.
Computes window-based statistics per component in real time.
"""

import math
from collections import deque
from typing import Dict, List, Optional


class ComponentBuffer:
    """
    Maintains a sliding window of raw sensor readings for one component
    and computes derived features on demand.
    """

    SENSORS = ["vibration", "temperature", "load", "current"]

    def __init__(self, component_id: str, window_size: int = 20):
        self.component_id = component_id
        self.window_size = window_size
        self._buffers: Dict[str, deque] = {
            s: deque(maxlen=window_size) for s in self.SENSORS
        }
        self._record_count = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, record: dict):
        """Push one record into each sensor buffer."""
        for sensor in self.SENSORS:
            val = record.get(sensor)
            if val is not None:
                self._buffers[sensor].append(float(val))
        self._record_count += 1

    def ready(self) -> bool:
        """True once we have at least window_size / 2 readings."""
        return self._record_count >= max(2, self.window_size // 2)

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(values) -> float:
        lst = list(values)
        return sum(lst) / len(lst) if lst else 0.0

    @staticmethod
    def _std(values) -> float:
        lst = list(values)
        if len(lst) < 2:
            return 0.0
        m = sum(lst) / len(lst)
        variance = sum((x - m) ** 2 for x in lst) / (len(lst) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _trend(values) -> float:
        """Simple linear slope (rise / run) over the window."""
        lst = list(values)
        n = len(lst)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(lst) / n
        numerator   = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(lst))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator != 0 else 0.0

    @staticmethod
    def _z_score(values) -> float:
        """Z-score of the most recent value."""
        lst = list(values)
        if len(lst) < 2:
            return 0.0
        m = sum(lst) / len(lst)
        s = ComponentBuffer._std(lst)
        return (lst[-1] - m) / s if s > 0 else 0.0

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def compute_features(self, record: dict) -> dict:
        """
        Returns a feature dict for the current record, enriched with
        rolling statistics for each sensor.
        """
        features: dict = {
            "component_id": self.component_id,
            "timestamp": record.get("timestamp"),
            "record_id": record.get("record_id"),
            "component_type": record.get("component_type"),
            "risk_level": record.get("risk_level"),
            "risk_score": record.get("risk_score", 0.0),
            "health_index": record.get("health_index", 1.0),
            "is_anomaly": record.get("is_anomaly", False),
            "is_degrading": record.get("is_degrading", False),
            "failure_mode": record.get("failure_mode", "normal"),
            "produced_at": record.get("produced_at"),
            "buffer_fill": self._record_count,
        }

        for sensor in self.SENSORS:
            buf = self._buffers[sensor]
            raw = record.get(sensor, 0.0)

            features[sensor] = raw
            features[f"{sensor}_mean"]   = self._mean(buf)
            features[f"{sensor}_std"]    = self._std(buf)
            features[f"{sensor}_trend"]  = self._trend(buf)
            features[f"{sensor}_zscore"] = self._z_score(buf)
            features[f"{sensor}_min"]    = min(buf) if buf else raw
            features[f"{sensor}_max"]    = max(buf) if buf else raw

        # Composite anomaly score: average |z-score| across sensors
        z_scores = [abs(features[f"{s}_zscore"]) for s in self.SENSORS]
        features["composite_anomaly_score"] = sum(z_scores) / len(z_scores)

        # Rate-of-change flag (any sensor trending sharply)
        trends = [abs(features[f"{s}_trend"]) for s in self.SENSORS]
        features["max_trend_magnitude"] = max(trends)

        return features


class RollingFeatureEngine:
    """
    Manages one ComponentBuffer per component_id and routes incoming
    records to the correct buffer before computing features.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._buffers: Dict[str, ComponentBuffer] = {}
        self._total_processed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, record: dict) -> Optional[dict]:
        """
        Process one raw record. Returns enriched feature dict, or None
        if the buffer is not yet warm enough to compute reliable stats.
        """
        cid = record.get("component_id", "unknown")

        if cid not in self._buffers:
            self._buffers[cid] = ComponentBuffer(cid, self.window_size)

        buf = self._buffers[cid]
        buf.add(record)
        self._total_processed += 1

        if not buf.ready():
            return None          # warming up — skip until buffer has data

        return buf.compute_features(record)

    def process_always(self, record: dict) -> dict:
        """
        Like process() but always returns features (with zeros during warm-up).
        Useful when downstream consumers can't tolerate None.
        """
        result = self.process(record)
        if result is not None:
            return result

        # Return minimal feature dict during warm-up
        cid = record.get("component_id", "unknown")
        minimal = {k: record.get(k) for k in record}
        minimal["component_id"] = cid
        minimal["composite_anomaly_score"] = 0.0
        minimal["max_trend_magnitude"]     = 0.0
        minimal["buffer_fill"]             = self._buffers[cid]._record_count
        for s in ComponentBuffer.SENSORS:
            for suffix in ("mean", "std", "trend", "zscore", "min", "max"):
                minimal[f"{s}_{suffix}"] = 0.0
        return minimal

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def component_ids(self) -> List[str]:
        return list(self._buffers.keys())

    def stats(self) -> dict:
        return {
            "total_processed": self._total_processed,
            "components_tracked": len(self._buffers),
            "window_size": self.window_size,
        }