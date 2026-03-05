"""
Phase 3 (updated) — Kafka Producer with Folder Watcher

Two modes:
  1. watch  — monitors data/incoming/ for new CSV files, streams them automatically
  2. stream — manually stream a specific CSV file (original behaviour)

Usage:
  python -m src.kafka.producer --mode watch --folder data/incoming/
  python -m src.kafka.producer --mode stream --csv data/raw/railway_sensor_data.csv
"""

import csv
import json
import time
import os
import argparse
from pathlib import Path
from datetime import datetime

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from src.kafka.config import KAFKA_BOOTSTRAP_SERVERS, TOPICS, STREAM_CONFIG


class RailwaySensorProducer:

    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS):
        self.bootstrap_servers = bootstrap_servers
        self.topic    = TOPICS["raw_sensors"]
        self.producer = None
        self._stats   = {"sent": 0, "errors": 0, "files_processed": 0}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            print("[Producer] kafka-python not installed: pip install kafka-python")
            return False
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all", retries=3, linger_ms=10, compression_type="gzip",
            )
            print(f"[Producer] Connected → {self.bootstrap_servers}")
            return True
        except NoBrokersAvailable:
            print(f"[Producer] No broker at {self.bootstrap_servers}. Start Kafka first.")
            return False

    def disconnect(self):
        if self.producer:
            self.producer.flush()
            self.producer.close()
        print(f"[Producer] Done. Sent {self._stats['sent']:,} records "
              f"from {self._stats['files_processed']} file(s).")

    # ------------------------------------------------------------------
    # Mode 1: Folder Watcher
    # ------------------------------------------------------------------

    def watch_folder(self, folder: str = "data/incoming",
                     poll_interval: int = 10,
                     speed_ms: int = 50):
        """
        Watch a folder. When a new CSV appears:
          1. Stream all its records to Kafka
          2. Move the file to data/processed_incoming/ so it's not re-sent
        """
        watch_dir    = Path(folder)
        done_dir     = Path("data/processed_incoming")
        watch_dir.mkdir(parents=True, exist_ok=True)
        done_dir.mkdir(parents=True, exist_ok=True)

        seen = set()

        print(f"\n[Producer] 👁  Watching folder: {watch_dir.resolve()}")
        print(f"[Producer]    Drop any CSV file there to trigger live inference.")
        print(f"[Producer]    Poll interval: {poll_interval}s")
        print(f"[Producer]    Press Ctrl-C to stop.\n")

        try:
            while True:
                csv_files = sorted(watch_dir.glob("*.csv"))
                for f in csv_files:
                    if f.name not in seen:
                        seen.add(f.name)
                        print(f"[Producer] 📂 New file detected: {f.name}")
                        count = self._stream_file(f, speed_ms=speed_ms)
                        # Move to done folder
                        dest = done_dir / f.name
                        f.rename(dest)
                        print(f"[Producer] ✅ Streamed {count:,} records. "
                              f"File moved → {dest}")
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[Producer] Watcher stopped.")

    # ------------------------------------------------------------------
    # Mode 2: Stream a specific file
    # ------------------------------------------------------------------

    def stream_file(self, csv_path: str, speed_ms: int = 50):
        p = Path(csv_path)
        if not p.exists():
            print(f"[Producer] File not found: {csv_path}")
            return
        count = self._stream_file(p, speed_ms=speed_ms)
        print(f"[Producer] Streamed {count:,} records from {csv_path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stream_file(self, path: Path, speed_ms: int = 50) -> int:
        delay = speed_ms / 1000.0
        count = 0
        start = time.time()

        with open(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                record = self._parse_row(row)
                record["source_file"] = path.name
                cid = record.get("component_id", "unknown")
                self.producer.send(self.topic, key=cid, value=record) \
                    .add_errback(lambda e: self._stats.update({"errors": self._stats["errors"]+1}))
                count += 1
                if delay > 0:
                    time.sleep(delay)

        self.producer.flush()
        self._stats["sent"]            += count
        self._stats["files_processed"] += 1
        elapsed = time.time() - start
        print(f"[Producer]   {path.name}: {count:,} records in {elapsed:.1f}s "
              f"({count/elapsed:.0f} rec/s)")
        return count

    @staticmethod
    def _parse_row(row: dict) -> dict:
        float_cols = ["vibration","temperature","load","current","risk_score","health_index"]
        int_cols   = ["record_id","time_step"]
        bool_cols  = ["is_anomaly","is_degrading"]
        parsed = dict(row)
        for c in float_cols:
            try:    parsed[c] = float(parsed[c])
            except: parsed[c] = 0.0
        for c in int_cols:
            try:    parsed[c] = int(parsed[c])
            except: parsed[c] = 0
        for c in bool_cols:
            parsed[c] = str(parsed.get(c,"")).lower() in ("true","1","yes")
        parsed["produced_at"] = datetime.utcnow().isoformat()
        return parsed


def main():
    parser = argparse.ArgumentParser(description="Railway Kafka Producer")
    parser.add_argument("--mode",     default="watch", choices=["watch","stream"])
    parser.add_argument("--folder",   default="data/incoming")
    parser.add_argument("--csv",      default="data/raw/railway_sensor_data.csv")
    parser.add_argument("--speed",    type=int, default=50)
    parser.add_argument("--interval", type=int, default=10,
                        help="Folder poll interval in seconds")
    parser.add_argument("--brokers",  default=KAFKA_BOOTSTRAP_SERVERS)
    args = parser.parse_args()

    p = RailwaySensorProducer(bootstrap_servers=args.brokers)
    if not p.connect():
        return
    try:
        if args.mode == "watch":
            p.watch_folder(args.folder, poll_interval=args.interval, speed_ms=args.speed)
        else:
            p.stream_file(args.csv, speed_ms=args.speed)
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()