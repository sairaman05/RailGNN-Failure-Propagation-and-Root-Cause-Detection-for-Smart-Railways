"""
Live Sensor Simulator
=====================
Generates realistic sensor readings for all 20 railway components
and pushes them directly to Kafka as individual JSON messages.

No CSV files involved. Each message = one sensor reading from one component.

Features:
  - Realistic baseline readings per component type
  - Gaussian noise on every reading
  - Injected degradation events (so model sees high-risk predictions)
  - Configurable interval between readings
  - Graceful shutdown on Ctrl-C

Usage:
  python -m src.kafka.live_sensor_simulator
  python -m src.kafka.live_sensor_simulator --interval 2 --degrade T05
"""

import json
import math
import time
import random
import argparse
import threading
from datetime import datetime, timezone

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from src.kafka.config import KAFKA_BOOTSTRAP_SERVERS, TOPICS


# ── Component baseline sensor profiles ──────────────────────────────
# (vibration, temperature, load, current) — mean values per type
BASELINES = {
    # Tracks: moderate vibration, moderate temp, high load
    "T01": (0.35, 42.0, 65.0, 14.5),
    "T02": (0.33, 41.0, 63.0, 14.2),
    "T03": (0.36, 43.0, 67.0, 14.8),
    "T04": (0.34, 42.5, 64.0, 14.3),
    "T05": (0.37, 44.0, 68.0, 15.0),
    "T06": (0.35, 43.0, 66.0, 14.6),
    "T07": (0.36, 42.0, 65.0, 14.4),
    "T08": (0.34, 41.5, 63.5, 14.1),
    "T09": (0.35, 43.5, 67.0, 14.7),
    "T10": (0.33, 41.0, 62.0, 14.0),
    # Switches: higher vibration, moderate temp
    "SW1": (0.55, 48.0, 45.0, 18.5),
    "SW2": (0.53, 47.0, 44.0, 18.2),
    "SW3": (0.56, 49.0, 46.0, 18.8),
    "SW4": (0.54, 48.5, 45.5, 18.4),
    # Signals: low vibration, low load, stable current
    "SG1": (0.15, 35.0, 20.0, 8.5),
    "SG2": (0.14, 34.5, 19.5, 8.3),
    "SG3": (0.16, 35.5, 20.5, 8.6),
    # Bridges: low vibration, higher temp (thermal mass), very high load
    "BR1": (0.25, 55.0, 85.0, 22.0),
    "BR2": (0.24, 54.0, 83.0, 21.5),
    "BR3": (0.26, 56.0, 87.0, 22.5),
}

# Noise std devs per sensor type
NOISE = {
    "vibration":   0.03,
    "temperature": 1.5,
    "load":        3.0,
    "current":     0.5,
}

COMPONENT_TYPES = {
    **{f"T0{i}": "track"  for i in range(1, 10)},
    "T10": "track",
    **{f"SW{i}": "switch" for i in range(1, 5)},
    **{f"SG{i}": "signal" for i in range(1, 4)},
    **{f"BR{i}": "bridge" for i in range(1, 4)},
}


# ════════════════════════════════════════════════════════════════════
# Degradation Engine
# ════════════════════════════════════════════════════════════════════

class DegradationEngine:
    """
    Manages active degradation events for components.
    A degradation event gradually increases sensor readings
    over time to simulate developing faults.
    """

    def __init__(self):
        # component_id → {start_time, duration_s, severity, mode}
        self._events: dict = {}
        self._lock = threading.Lock()

    def inject(self, component_id: str, severity: float = 0.8,
               duration_s: float = 120.0, mode: str = "mechanical_wear"):
        """Start a degradation event on a component."""
        with self._lock:
            self._events[component_id] = {
                "start":    time.time(),
                "duration": duration_s,
                "severity": severity,
                "mode":     mode,
            }
        print(f"[Simulator] ⚡ Degradation injected: {component_id} "
              f"severity={severity:.1f} mode={mode} duration={duration_s}s")

    def get_multiplier(self, component_id: str) -> tuple:
        """
        Returns (vib_mult, temp_mult, load_mult, curr_mult, risk_score, mode).
        All multipliers = 1.0 when no degradation active.
        """
        with self._lock:
            event = self._events.get(component_id)

        if event is None:
            return 1.0, 1.0, 1.0, 1.0, 0.05, "normal"

        elapsed  = time.time() - event["start"]
        duration = event["duration"]

        if elapsed > duration:
            with self._lock:
                self._events.pop(component_id, None)
            return 1.0, 1.0, 1.0, 1.0, 0.05, "normal"

        # Progress 0→1 over the event duration
        progress = elapsed / duration
        # Exponential ramp for realism
        ramp     = 1.0 - math.exp(-3.0 * progress)
        sev      = event["severity"]
        mode     = event["mode"]

        if mode == "mechanical_wear":
            return (1.0 + ramp * sev * 2.5,   # vibration spikes
                    1.0 + ramp * sev * 0.8,   # temperature rises moderately
                    1.0 + ramp * sev * 0.3,   # load increases slightly
                    1.0 + ramp * sev * 0.5,   # current rises
                    ramp * sev,
                    mode)
        elif mode == "overheating":
            return (1.0 + ramp * sev * 0.5,
                    1.0 + ramp * sev * 3.0,   # temperature spikes
                    1.0 + ramp * sev * 0.4,
                    1.0 + ramp * sev * 1.5,   # current rises significantly
                    ramp * sev,
                    mode)
        elif mode == "electrical_fault":
            return (1.0 + ramp * sev * 0.3,
                    1.0 + ramp * sev * 1.0,
                    1.0 - ramp * sev * 0.2,   # load drops (circuit issue)
                    1.0 + ramp * sev * 3.0,   # current spikes dangerously
                    ramp * sev,
                    mode)
        else:
            return (1.0 + ramp * sev * 1.0,
                    1.0 + ramp * sev * 1.0,
                    1.0 + ramp * sev * 1.0,
                    1.0 + ramp * sev * 1.0,
                    ramp * sev,
                    mode)

    def active_events(self) -> list:
        with self._lock:
            return list(self._events.keys())


# ════════════════════════════════════════════════════════════════════
# Live Sensor Simulator
# ════════════════════════════════════════════════════════════════════

class LiveSensorSimulator:
    """
    Generates one sensor reading per component every `interval` seconds
    and pushes each reading as a JSON message to Kafka.
    """

    def __init__(self,
                 bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
                 interval:          float = 2.0,
                 auto_degrade:      bool  = True):
        self.bootstrap_servers = bootstrap_servers
        self.interval          = interval
        self.auto_degrade      = auto_degrade
        self.topic             = TOPICS["raw_sensors"]
        self.producer          = None
        self.degradation       = DegradationEngine()
        self._running          = False
        self._step             = 0
        self._stats            = {"sent": 0, "errors": 0}

        # Schedule automatic degradation events if enabled
        if auto_degrade:
            self._schedule_auto_degradation()

    # ------------------------------------------------------------------
    # Auto-degradation schedule
    # ------------------------------------------------------------------

    def _schedule_auto_degradation(self):
        """
        Schedule realistic degradation events at intervals so the
        model sees actual high-risk predictions during demo.
        """
        events = [
            # (delay_seconds, component, severity, duration, mode)
            (30,  "T05", 0.85, 90,  "mechanical_wear"),
            (60,  "BR2", 0.90, 80,  "overheating"),
            (90,  "SW3", 0.80, 70,  "electrical_fault"),
            (150, "T05", 0.95, 120, "mechanical_wear"),   # escalation
            (180, "SG2", 0.75, 60,  "electrical_fault"),
            (240, "BR2", 0.92, 100, "overheating"),
            (300, "T07", 0.80, 90,  "mechanical_wear"),
        ]

        def _fire(delay, cid, sev, dur, mode):
            time.sleep(delay)
            if self._running:
                self.degradation.inject(cid, sev, dur, mode)

        for delay, cid, sev, dur, mode in events:
            t = threading.Thread(
                target=_fire,
                args=(delay, cid, sev, dur, mode),
                daemon=True
            )
            t.start()

    # ------------------------------------------------------------------
    # Kafka connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            print("[Simulator] kafka-python not installed.")
            print("            pip install kafka-python")
            return False
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks=1,
                linger_ms=5,
                retries=3,
            )
            print(f"[Simulator] ✅ Connected to Kafka: {self.bootstrap_servers}")
            return True
        except NoBrokersAvailable:
            print(f"[Simulator] ❌ No Kafka broker at {self.bootstrap_servers}")
            print("            Start Kafka: docker-compose up -d")
            return False
        except Exception as e:
            print(f"[Simulator] ❌ Connection error: {e}")
            return False

    def disconnect(self):
        self._running = False
        if self.producer:
            self.producer.flush()
            self.producer.close()
        print(f"\n[Simulator] Stopped. Sent {self._stats['sent']:,} messages.")

    # ------------------------------------------------------------------
    # Reading generation
    # ------------------------------------------------------------------

    def _generate_reading(self, component_id: str) -> dict:
        """Generate one realistic sensor reading for a component."""
        base_vib, base_temp, base_load, base_curr = BASELINES[component_id]

        # Get degradation multipliers
        vm, tm, lm, cm, risk_score, mode = \
            self.degradation.get_multiplier(component_id)

        # Apply multipliers + Gaussian noise
        vibration   = max(0.0, base_vib  * vm + random.gauss(0, NOISE["vibration"]))
        temperature = max(0.0, base_temp * tm + random.gauss(0, NOISE["temperature"]))
        load        = max(0.0, base_load * lm + random.gauss(0, NOISE["load"]))
        current     = max(0.0, base_curr * cm + random.gauss(0, NOISE["current"]))

        # Derive health index from risk score
        health_index = max(0.0, 1.0 - risk_score)

        # Risk level from risk score
        if risk_score >= 0.8:
            risk_level = "high"
        elif risk_score >= 0.6:
            risk_level = "medium"
        elif risk_score >= 0.3:
            risk_level = "low"
        else:
            risk_level = "normal"

        return {
            "component_id":   component_id,
            "component_type": COMPONENT_TYPES[component_id],
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "time_step":      self._step,
            "record_id":      self._step * len(BASELINES) + list(BASELINES.keys()).index(component_id),
            "vibration":      round(vibration,   4),
            "temperature":    round(temperature, 4),
            "load":           round(load,        4),
            "current":        round(current,     4),
            "risk_score":     round(risk_score,  4),
            "health_index":   round(health_index,4),
            "risk_level":     risk_level,
            "failure_mode":   mode,
            "is_anomaly":     risk_score >= 0.6,
            "is_degrading":   risk_score >= 0.3,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Start generating and sending sensor readings."""
        self._running = True

        print(f"\n{'='*60}")
        print(f"  LIVE SENSOR SIMULATOR")
        print(f"{'='*60}")
        print(f"  Components   : {len(BASELINES)}")
        print(f"  Interval     : {self.interval}s per cycle")
        print(f"  Topic        : {self.topic}")
        print(f"  Auto-degrade : {self.auto_degrade}")
        print(f"  Press Ctrl-C to stop")
        print(f"{'='*60}\n")

        try:
            while self._running:
                cycle_start = time.time()

                # Send one reading per component
                for component_id in BASELINES:
                    record = self._generate_reading(component_id)
                    self.producer.send(
                        self.topic,
                        key=component_id,
                        value=record
                    ).add_errback(
                        lambda e: self._stats.update(
                            {"errors": self._stats["errors"] + 1}
                        )
                    )
                    self._stats["sent"] += 1

                self._step += 1

                # Status every 10 cycles
                if self._step % 10 == 0:
                    active = self.degradation.active_events()
                    elapsed = self._step * self.interval
                    print(f"  [t={elapsed:.0f}s]  step={self._step}  "
                          f"sent={self._stats['sent']:,}  "
                          f"errors={self._stats['errors']}  "
                          f"degrading={active if active else 'none'}")

                # Sleep for remainder of interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0.0, self.interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            self.disconnect()

    # ------------------------------------------------------------------
    # Manual degradation injection (for testing)
    # ------------------------------------------------------------------

    def inject(self, component_id: str, severity: float = 0.9,
               duration_s: float = 60.0, mode: str = "mechanical_wear"):
        self.degradation.inject(component_id, severity, duration_s, mode)


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Railway Live Sensor Simulator — pushes JSON to Kafka"
    )
    parser.add_argument("--interval",  type=float, default=2.0,
                        help="Seconds between reading cycles (default: 2)")
    parser.add_argument("--brokers",   default=KAFKA_BOOTSTRAP_SERVERS,
                        help="Kafka bootstrap servers")
    parser.add_argument("--no-degrade", action="store_true",
                        help="Disable automatic degradation events")
    parser.add_argument("--degrade",   type=str, default=None,
                        help="Immediately inject degradation on this component e.g. T05")
    parser.add_argument("--severity",  type=float, default=0.9,
                        help="Degradation severity 0-1 (default: 0.9)")
    parser.add_argument("--mode",      default="mechanical_wear",
                        choices=["mechanical_wear", "overheating", "electrical_fault"],
                        help="Degradation mode")
    args = parser.parse_args()

    sim = LiveSensorSimulator(
        bootstrap_servers=args.brokers,
        interval=args.interval,
        auto_degrade=not args.no_degrade,
    )

    if not sim.connect():
        return

    # Manual immediate injection if requested
    if args.degrade:
        sim.inject(args.degrade, args.severity, 120.0, args.mode)

    sim.run()


if __name__ == "__main__":
    main()