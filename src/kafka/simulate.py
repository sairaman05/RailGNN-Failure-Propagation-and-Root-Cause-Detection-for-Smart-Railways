"""
Offline Simulation — runs the full Phase 3 pipeline without Kafka.

Use this to test feature engineering and alerts locally before
setting up a Kafka broker.

Usage:
    python -m src.kafka.simulate
    python -m src.kafka.simulate --csv data/raw/railway_sensor_data.csv --limit 5000
"""

import csv
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

from src.kafka.feature_engineer import RollingFeatureEngine
from src.kafka.alert_manager  import AlertManager
from src.kafka.config import STREAM_CONFIG


def simulate(csv_path: str = "data/raw/railway_sensor_data.csv",
             limit: int = 5000,
             window_size: int = 20,
             print_every: int = 1000,
             save_output: bool = True):
    """
    Read sensor CSV → feature engine → alert manager → save outputs.
    No Kafka needed.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[Simulate] ERROR: {csv_path} not found.")
        print("           Run Phase 1 first: python -m src.data_generation.sensor_simulator")
        return

    engine    = RollingFeatureEngine(window_size=window_size)
    alerter   = AlertManager(cooldown_seconds=30)

    processed_records = []
    all_alerts        = []
    start             = time.time()

    print(f"\n{'='*60}")
    print(f"  PHASE 3 — OFFLINE SIMULATION")
    print(f"{'='*60}")
    print(f"  CSV        : {csv_path}")
    print(f"  Limit      : {limit}")
    print(f"  Window     : {window_size}")
    print(f"  Started    : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    count = 0
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if count >= limit:
                break

            record = _parse_row(row)
            features = engine.process_always(record)
            processed_records.append(features)

            alert = alerter.evaluate(features)
            if alert:
                all_alerts.append(alert.to_dict())
                print(f"  ⚠  {alert}")

            count += 1
            if count % print_every == 0:
                elapsed = time.time() - start
                print(f"  [{count:>6,}] Components: {len(engine.component_ids()):<3}  "
                      f"Alerts: {alerter.summary()['total_alerts']:<4}  "
                      f"Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = alerter.summary()
    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Records processed : {count:,}")
    print(f"  Components tracked: {len(engine.component_ids())}")
    print(f"  Total alerts      : {summary['total_alerts']}")
    for lvl, cnt in summary.get("by_level", {}).items():
        print(f"    {lvl:<10} {cnt}")
    print(f"  Elapsed           : {elapsed:.2f}s")
    print(f"  Throughput        : {count / elapsed:.0f} rec/s")

    # Sample feature vector
    if processed_records:
        sample = processed_records[-1]
        print(f"\n  Sample feature record (last):")
        important = ["component_id", "timestamp", "vibration", "vibration_mean",
                     "vibration_std", "vibration_trend", "vibration_zscore",
                     "temperature", "temperature_mean",
                     "composite_anomaly_score", "risk_score", "risk_level"]
        for k in important:
            v = sample.get(k, "N/A")
            if isinstance(v, float):
                v = f"{v:.4f}"
            print(f"    {k:<30} {v}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if save_output:
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save processed features (last 1000 to keep size manageable)
        features_path = out_dir / "processed_features_sample.json"
        with open(features_path, "w") as fh:
            json.dump(processed_records[-1000:], fh, indent=2, default=str)
        print(f"\n  Saved processed features  → {features_path}")

        # Save all alerts
        alerts_path = out_dir / "alerts.json"
        with open(alerts_path, "w") as fh:
            json.dump(all_alerts, fh, indent=2, default=str)
        print(f"  Saved alerts              → {alerts_path}")

        # Save summary
        summary_path = out_dir / "phase3_summary.json"
        with open(summary_path, "w") as fh:
            json.dump({
                "records_processed": count,
                "elapsed_seconds": round(elapsed, 2),
                "throughput_rec_per_sec": round(count / elapsed, 1),
                "components_tracked": len(engine.component_ids()),
                "component_ids": engine.component_ids(),
                "alert_summary": summary,
                "feature_engine_stats": engine.stats(),
            }, fh, indent=2)
        print(f"  Saved phase3 summary      → {summary_path}")

    print(f"\n{'='*60}\n")
    return processed_records, all_alerts


def _parse_row(row: dict) -> dict:
    float_cols = ["vibration", "temperature", "load", "current",
                  "risk_score", "health_index"]
    int_cols   = ["record_id", "time_step"]
    bool_cols  = ["is_anomaly", "is_degrading"]

    parsed = dict(row)
    for c in float_cols:
        if c in parsed:
            try:   parsed[c] = float(parsed[c])
            except: parsed[c] = 0.0
    for c in int_cols:
        if c in parsed:
            try:   parsed[c] = int(parsed[c])
            except: parsed[c] = 0
    for c in bool_cols:
        if c in parsed:
            parsed[c] = str(parsed[c]).lower() in ("true", "1", "yes")
    parsed["produced_at"] = datetime.utcnow().isoformat()
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Offline Simulation")
    parser.add_argument("--csv",    default="data/raw/railway_sensor_data.csv")
    parser.add_argument("--limit",  type=int, default=5000)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--every",  type=int, default=1000,
                        help="Print progress every N records")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving output files")
    args = parser.parse_args()

    simulate(
        csv_path=args.csv,
        limit=args.limit,
        window_size=args.window,
        print_every=args.every,
        save_output=not args.no_save,
    )


if __name__ == "__main__":
    main()