"""
Kafka Consumer for Railway Sensor Stream.

Pipeline:
  railway.sensors.raw  →  [RollingFeatureEngine]  →  railway.sensors.processed
                                                   →  railway.alerts  (when triggered)
"""

import json
import time
import argparse
import threading
from datetime import datetime
from collections import defaultdict
from typing import Optional

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from src.kafka.config import (
    KAFKA_BOOTSTRAP_SERVERS, TOPICS, CONSUMER_CONFIG, STREAM_CONFIG
)
from src.kafka.feature_engineer import RollingFeatureEngine
from src.kafka.alert_manager  import AlertManager


class RailwayStreamProcessor:
    """
    Consumes raw sensor records, enriches them with rolling features,
    publishes processed records and alerts back to Kafka.
    Prints a live dashboard to stdout every N records.
    """

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
                 window_size: int = None, print_every: int = 500):
        self.bootstrap_servers = bootstrap_servers
        self.window_size  = window_size or STREAM_CONFIG["rolling_window_size"]
        self.print_every  = print_every

        self.consumer: Optional[KafkaConsumer]  = None
        self.producer: Optional[KafkaProducer]  = None
        self.feature_engine = RollingFeatureEngine(self.window_size)
        self.alert_manager  = AlertManager()

        # Stats
        self._stats = defaultdict(int)
        self._start_time: Optional[float] = None
        self._running = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            print("[Consumer] kafka-python not installed.")
            return False
        try:
            self.consumer = KafkaConsumer(
                TOPICS["raw_sensors"],
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                group_id=CONSUMER_CONFIG["group_id"],
                auto_offset_reset=CONSUMER_CONFIG["auto_offset_reset"],
                enable_auto_commit=True,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
                max_poll_records=500,
            )
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks=1,          # lower acks for downstream topics (speed)
                linger_ms=5,
            )
            print(f"[Consumer] Connected to Kafka at {self.bootstrap_servers}")
            print(f"[Consumer] Subscribed to: {TOPICS['raw_sensors']}")
            return True
        except NoBrokersAvailable:
            print(f"[Consumer] ERROR: No Kafka broker at {self.bootstrap_servers}")
            return False
        except Exception as exc:
            print(f"[Consumer] ERROR: {exc}")
            return False

    def disconnect(self):
        self._running = False
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.flush()
            self.producer.close()
        print("[Consumer] Disconnected.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_records: int = None):
        """Start consuming. Blocks until max_records reached or Ctrl-C."""
        self._running    = True
        self._start_time = time.time()
        total            = 0

        print(f"\n[Consumer] Listening for messages...")
        print(f"[Consumer] Window size : {self.window_size}")
        print(f"[Consumer] Max records : {max_records or 'unlimited'}")
        print("-" * 60)

        try:
            for message in self.consumer:
                if not self._running:
                    break
                if max_records and total >= max_records:
                    break

                raw_record = message.value
                self._process_message(raw_record)
                total += 1

                if total % self.print_every == 0:
                    self._print_dashboard(total)

        except KeyboardInterrupt:
            print("\n[Consumer] Interrupted.")
        finally:
            self.disconnect()
            self._print_final_summary(total)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _process_message(self, raw_record: dict):
        """Core pipeline: feature engineering → publish → alert check."""
        # 1. Enrich with rolling features
        features = self.feature_engine.process_always(raw_record)

        # 2. Publish processed record
        cid = features.get("component_id", "unknown")
        self.producer.send(
            TOPICS["processed_features"],
            key=cid,
            value=features
        )
        self._stats["processed"] += 1

        # 3. Alert evaluation
        alert = self.alert_manager.evaluate(features)
        if alert:
            self.producer.send(
                TOPICS["alerts"],
                key=cid,
                value=alert.to_dict()
            )
            self._stats[f"alert_{alert.level}"] += 1
            print(f"\n  ⚠  {alert}")

        # 4. Track risk distribution
        risk_level = raw_record.get("risk_level", "normal")
        self._stats[f"risk_{risk_level}"] += 1

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _print_dashboard(self, total: int):
        elapsed   = time.time() - self._start_time
        rate      = total / elapsed if elapsed > 0 else 0
        eng_stats = self.feature_engine.stats()
        alert_sum = self.alert_manager.summary()

        print(f"\n{'='*60}")
        print(f"  STREAM PROCESSOR — {datetime.utcnow().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"  Records consumed : {total:,}")
        print(f"  Throughput       : {rate:.0f} rec/s")
        print(f"  Components seen  : {eng_stats['components_tracked']}")
        print(f"  Elapsed          : {elapsed:.1f}s")
        print(f"\n  Risk distribution:")
        for level in ["normal", "low", "medium", "high"]:
            count = self._stats.get(f"risk_{level}", 0)
            bar   = "█" * min(30, count // max(total // 30, 1))
            print(f"    {level:<8} {count:>6,}  {bar}")
        print(f"\n  Alerts fired     : {alert_sum['total_alerts']}")
        for lvl, cnt in alert_sum.get("by_level", {}).items():
            print(f"    {lvl:<10} {cnt}")
        if alert_sum["total_alerts"] > 0:
            print(f"\n  Recent alerts:")
            for a in self.alert_manager.recent_alerts(3):
                print(f"    [{a['alert_level']}] {a['component_id']} "
                      f"risk={a['risk_score']:.2f} {a['failure_mode']}")
        print(f"{'='*60}\n")

    def _print_final_summary(self, total: int):
        elapsed   = time.time() - (self._start_time or time.time())
        alert_sum = self.alert_manager.summary()
        print(f"\n{'='*60}")
        print(f"  FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"  Total consumed  : {total:,}")
        print(f"  Total processed : {self._stats['processed']:,}")
        print(f"  Total alerts    : {alert_sum['total_alerts']}")
        print(f"  Elapsed         : {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Avg throughput  : {total / elapsed:.0f} rec/s")
        print(f"{'='*60}\n")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Railway Stream Processor (Consumer)")
    parser.add_argument("--max",     type=int, default=None,
                        help="Max records to consume (default: unlimited)")
    parser.add_argument("--window",  type=int, default=20,
                        help="Rolling window size (default: 20)")
    parser.add_argument("--brokers", default=KAFKA_BOOTSTRAP_SERVERS,
                        help="Kafka bootstrap servers")
    parser.add_argument("--every",   type=int, default=500,
                        help="Print dashboard every N records")
    args = parser.parse_args()

    processor = RailwayStreamProcessor(
        bootstrap_servers=args.brokers,
        window_size=args.window,
        print_every=args.every,
    )

    if not processor.connect():
        return

    processor.run(max_records=args.max)


if __name__ == "__main__":
    main()