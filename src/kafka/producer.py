"""
Kafka Producer for Railway Sensor Data
Streams records from the generated CSV as if they were live sensors.
"""

import json
import time
import csv
import argparse
from datetime import datetime
from pathlib import Path

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from src.kafka.config import (
    KAFKA_BOOTSTRAP_SERVERS, TOPICS, PRODUCER_CONFIG, STREAM_CONFIG
)


class RailwaySensorProducer:
    """
    Reads railway_sensor_data.csv and publishes records to Kafka.
    Simulates a real-time feed at configurable speed.
    """

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
                 simulation_speed_ms: int = None):
        self.bootstrap_servers = bootstrap_servers
        self.topic = TOPICS["raw_sensors"]
        self.speed_ms = simulation_speed_ms or STREAM_CONFIG["simulation_speed_ms"]
        self.producer = None
        self._stats = {"sent": 0, "errors": 0, "start_time": None}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            print("[Producer] kafka-python not installed. Run: pip install kafka-python")
            return False
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                batch_size=16384,
                linger_ms=10,
                compression_type="gzip",
            )
            print(f"[Producer] Connected to Kafka at {self.bootstrap_servers}")
            return True
        except NoBrokersAvailable:
            print(f"[Producer] ERROR: No Kafka broker at {self.bootstrap_servers}")
            print("           Start Kafka first (see docker-compose.yml).")
            return False
        except Exception as exc:
            print(f"[Producer] ERROR connecting: {exc}")
            return False

    def disconnect(self):
        if self.producer:
            self.producer.flush()
            self.producer.close()
            print("[Producer] Disconnected.")

    # ------------------------------------------------------------------
    # Sending helpers
    # ------------------------------------------------------------------

    def _on_send_success(self, record_metadata):
        self._stats["sent"] += 1

    def _on_send_error(self, exc):
        self._stats["errors"] += 1
        print(f"[Producer] Send error: {exc}")

    def send_record(self, record: dict):
        """Send a single sensor record. Key = component_id for partitioning."""
        key = record.get("component_id", "unknown")
        self.producer.send(self.topic, key=key, value=record) \
            .add_callback(self._on_send_success) \
            .add_errback(self._on_send_error)

    # ------------------------------------------------------------------
    # Main streaming loop
    # ------------------------------------------------------------------

    def stream_from_csv(self, csv_path: str, limit: int = None,
                        speed_ms: int = None, verbose_every: int = 1000):
        """
        Stream records from csv_path to Kafka.

        Args:
            csv_path:      Path to railway_sensor_data.csv
            limit:         Stop after N records (None = all)
            speed_ms:      Override simulation speed
            verbose_every: Print progress every N records
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        delay = (speed_ms or self.speed_ms) / 1000.0
        self._stats["start_time"] = time.time()

        print(f"\n[Producer] Streaming from: {csv_path}")
        print(f"[Producer] Topic: {self.topic}")
        print(f"[Producer] Speed: {speed_ms or self.speed_ms} ms / record")
        print(f"[Producer] Limit: {limit or 'all'}")
        print("-" * 60)

        count = 0
        try:
            with open(path, "r") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if limit and count >= limit:
                        break

                    record = self._parse_row(row)
                    self.send_record(record)
                    count += 1

                    if count % verbose_every == 0:
                        elapsed = time.time() - self._stats["start_time"]
                        rate = count / elapsed
                        print(f"  Sent: {count:,}  |  Errors: {self._stats['errors']}  "
                              f"|  Rate: {rate:.0f} rec/s  |  Elapsed: {elapsed:.1f}s")

                    if delay > 0:
                        time.sleep(delay)

        except KeyboardInterrupt:
            print("\n[Producer] Interrupted by user.")
        finally:
            if self.producer:
                self.producer.flush()
            elapsed = time.time() - self._stats["start_time"]
            print(f"\n[Producer] Done. Sent {self._stats['sent']:,} records in {elapsed:.1f}s.")
            print(f"[Producer] Errors: {self._stats['errors']}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_row(row: dict) -> dict:
        """Convert CSV string values to proper Python types."""
        float_cols = ["vibration", "temperature", "load", "current",
                      "risk_score", "health_index"]
        int_cols   = ["record_id", "time_step"]
        bool_cols  = ["is_anomaly", "is_degrading"]

        parsed = dict(row)
        for col in float_cols:
            if col in parsed:
                try:
                    parsed[col] = float(parsed[col])
                except (ValueError, TypeError):
                    parsed[col] = 0.0
        for col in int_cols:
            if col in parsed:
                try:
                    parsed[col] = int(parsed[col])
                except (ValueError, TypeError):
                    parsed[col] = 0
        for col in bool_cols:
            if col in parsed:
                parsed[col] = str(parsed[col]).lower() in ("true", "1", "yes")

        # Add producer timestamp
        parsed["produced_at"] = datetime.utcnow().isoformat()
        return parsed

    def print_stats(self):
        elapsed = time.time() - (self._stats["start_time"] or time.time())
        print(f"\n--- Producer Stats ---")
        print(f"  Records sent : {self._stats['sent']:,}")
        print(f"  Errors       : {self._stats['errors']}")
        print(f"  Elapsed      : {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Rate         : {self._stats['sent'] / elapsed:.0f} rec/s")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Railway Sensor Kafka Producer")
    parser.add_argument("--csv",   default="data/raw/railway_sensor_data.csv",
                        help="Path to sensor CSV")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records to send (default: all)")
    parser.add_argument("--speed", type=int, default=50,
                        help="ms between records (default: 50, 0=max speed)")
    parser.add_argument("--brokers", default=KAFKA_BOOTSTRAP_SERVERS,
                        help="Kafka bootstrap servers")
    args = parser.parse_args()

    producer = RailwaySensorProducer(
        bootstrap_servers=args.brokers,
        simulation_speed_ms=args.speed
    )

    if not producer.connect():
        return

    try:
        producer.stream_from_csv(args.csv, limit=args.limit, speed_ms=args.speed)
    finally:
        producer.disconnect()


if __name__ == "__main__":
    main()