"""
Alert Manager — generates structured alerts from processed feature records.
Respects cooldown windows to avoid alert storms.
"""

import time
from datetime import datetime
from typing import List, Optional, Dict
from src.kafka.config import RISK_THRESHOLDS, STREAM_CONFIG


class Alert:
    """Represents a single alert event."""

    LEVELS = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def __init__(self, component_id: str, level: str, message: str,
                 risk_score: float, features: dict):
        self.component_id  = component_id
        self.level         = level
        self.message       = message
        self.risk_score    = risk_score
        self.timestamp     = datetime.utcnow().isoformat()
        self.record_id     = features.get("record_id")
        self.failure_mode  = features.get("failure_mode", "normal")
        self.health_index  = features.get("health_index", 1.0)
        self.anomaly_score = features.get("composite_anomaly_score", 0.0)

    def to_dict(self) -> dict:
        return {
            "alert_level":     self.level,
            "component_id":    self.component_id,
            "message":         self.message,
            "risk_score":      round(self.risk_score, 4),
            "failure_mode":    self.failure_mode,
            "health_index":    round(self.health_index, 4),
            "anomaly_score":   round(self.anomaly_score, 4),
            "record_id":       self.record_id,
            "timestamp":       self.timestamp,
        }

    def __repr__(self):
        return (f"[{self.level}] {self.component_id} | "
                f"risk={self.risk_score:.2f} | {self.message}")


class AlertManager:
    """
    Generates and deduplicates alerts from processed feature dicts.
    Each component is subject to a cooldown period so we don't flood
    Kafka with repeated HIGH alerts every 100 ms.
    """

    def __init__(self, cooldown_seconds: int = None):
        self.cooldown = cooldown_seconds or STREAM_CONFIG["alert_cooldown_seconds"]
        self._last_alert_time: Dict[str, float] = {}   # component_id → epoch
        self._counts: Dict[str, int]            = {}   # level → count
        self._history: List[Alert]              = []

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, features: dict) -> Optional[Alert]:
        """
        Check features and emit an Alert if thresholds are breached.
        Returns None if no alert or still in cooldown.
        """
        cid        = features.get("component_id", "unknown")
        risk       = float(features.get("risk_score", 0.0))
        anomaly    = float(features.get("composite_anomaly_score", 0.0))
        is_anomaly = features.get("is_anomaly", False)
        failure    = features.get("failure_mode", "normal")

        level = self._classify_level(risk, anomaly, is_anomaly)
        if level == "INFO":
            return None          # not worth alerting

        if self._in_cooldown(cid):
            return None

        message = self._build_message(cid, level, risk, failure, features)
        alert   = Alert(cid, level, message, risk, features)

        self._record(cid, level, alert)
        return alert

    def evaluate_batch(self, features_list: List[dict]) -> List[Alert]:
        """Evaluate a batch; returns all triggered alerts."""
        alerts = []
        for f in features_list:
            a = self.evaluate(f)
            if a:
                alerts.append(a)
        return alerts

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_level(self, risk: float, anomaly: float,
                        is_anomaly: bool) -> str:
        if risk >= RISK_THRESHOLDS["critical"] or anomaly > 5.0:
            return "CRITICAL"
        if risk >= RISK_THRESHOLDS["high"] or (is_anomaly and anomaly > 3.0):
            return "HIGH"
        if risk >= RISK_THRESHOLDS["medium"]:
            return "MEDIUM"
        if risk >= RISK_THRESHOLDS["low"]:
            return "LOW"
        return "INFO"

    @staticmethod
    def _build_message(cid: str, level: str, risk: float,
                       failure: str, features: dict) -> str:
        trend = features.get("vibration_trend", 0.0)
        temp  = features.get("temperature", 0.0)
        parts = [f"{level} alert on {cid}",
                 f"risk={risk:.2f}",
                 f"mode={failure}"]
        if abs(trend) > 0.05:
            direction = "rising" if trend > 0 else "falling"
            parts.append(f"vibration {direction} (trend={trend:.3f})")
        if temp > 85:
            parts.append(f"high temp={temp:.1f}°C")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Cooldown + bookkeeping
    # ------------------------------------------------------------------

    def _in_cooldown(self, component_id: str) -> bool:
        last = self._last_alert_time.get(component_id, 0.0)
        return (time.time() - last) < self.cooldown

    def _record(self, component_id: str, level: str, alert: Alert):
        self._last_alert_time[component_id] = time.time()
        self._counts[level] = self._counts.get(level, 0) + 1
        self._history.append(alert)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "total_alerts":      sum(self._counts.values()),
            "by_level":          dict(self._counts),
            "components_alerted": len(self._last_alert_time),
        }

    def recent_alerts(self, n: int = 10) -> List[dict]:
        return [a.to_dict() for a in self._history[-n:]]