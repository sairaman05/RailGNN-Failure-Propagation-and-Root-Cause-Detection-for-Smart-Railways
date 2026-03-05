"""
Inference Consumer
==================
Consumes raw sensor JSON from Kafka, enriches with rolling features,
runs TGNN inference, saves predictions, publishes alerts.

Pipeline:
  railway.sensors.raw
      → feature_engine  (rolling stats per component)
      → TGNN model      (risk score + root cause per node)
      → live_predictions.json  (dashboard reads this)
      → railway.alerts  (structured alert events)

Usage:
  python -m src.kafka.consumer
  python -m src.kafka.consumer --checkpoint checkpoints/best_model.pt
"""

import json
import time
import argparse
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

import torch

from src.kafka.config         import KAFKA_BOOTSTRAP_SERVERS, TOPICS
from src.kafka.feature_engineer import RollingFeatureEngine
from src.kafka.alert_manager  import AlertManager
from src.model.data_loader    import (
    COMPONENT_ORDER, COMP_INDEX, FEATURE_COLS,
    N_NODES, N_FEATURES, build_edge_index, _safe_float,
)
from src.model.tgnn           import build_model

RISK_LABELS   = ["normal", "low", "medium", "high"]
PRED_PATH     = Path("data/predictions/live_predictions.json")
BUFFER_KEEP   = 200   # keep last N predictions in memory


class InferenceConsumer:

    def __init__(self,
                 checkpoint:        str   = "checkpoints/best_model.pt",
                 bootstrap_servers: str   = KAFKA_BOOTSTRAP_SERVERS,
                 window_size:       int   = 20,
                 seq_len:           int   = 12,
                 print_every:       int   = 20):

        self.checkpoint        = checkpoint
        self.bootstrap_servers = bootstrap_servers
        self.window_size       = window_size
        self.seq_len           = seq_len
        self.print_every       = print_every

        self.consumer   = None
        self.producer   = None
        self.model      = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_index = build_edge_index().to(self.device)

        self.feature_engine = RollingFeatureEngine(window_size)
        self.alert_manager  = AlertManager()

        # Per-node temporal buffer: last seq_len enriched feature vectors
        self._node_buffers = {c: deque(maxlen=seq_len) for c in COMPONENT_ORDER}
        self._lock         = threading.Lock()

        # Circular buffer of recent predictions for the dashboard
        self._pred_buf = deque(maxlen=BUFFER_KEEP)

        PRED_PATH.parent.mkdir(parents=True, exist_ok=True)

        self._stats = {
            "consumed":  0,
            "predicted": 0,
            "alerts":    0,
            "start":     None,
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        p = Path(self.checkpoint)
        if not p.exists():
            print(f"[Consumer] ❌ Checkpoint not found: {self.checkpoint}")
            print("           Train the model first:")
            print("           python -m src.model.trainer --source json --epochs 50")
            return False

        ckpt = torch.load(p, map_location="cpu")
        self.model = build_model(ckpt.get("model_config"))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        print(f"[Consumer] ✅ Model loaded from: {self.checkpoint}")
        print(f"[Consumer]    Params     : {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"[Consumer]    Val loss   : {ckpt.get('best_val_loss', 'N/A')}")
        print(f"[Consumer]    Device     : {self.device}")
        return True

    def connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            print("[Consumer] ❌ kafka-python not installed: pip install kafka-python")
            return False
        try:
            self.consumer = KafkaConsumer(
                TOPICS["raw_sensors"],
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                group_id="railway-inference-consumer",
                auto_offset_reset="latest",   # only new messages from now
                enable_auto_commit=True,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
                max_poll_records=100,
            )
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks=1,
                linger_ms=5,
            )
            print(f"[Consumer] ✅ Connected to Kafka: {self.bootstrap_servers}")
            print(f"[Consumer]    Listening on: {TOPICS['raw_sensors']}")
            return True
        except NoBrokersAvailable:
            print(f"[Consumer] ❌ No broker at {self.bootstrap_servers}")
            print("           Start Kafka: docker-compose up -d")
            return False
        except Exception as e:
            print(f"[Consumer] ❌ Error: {e}")
            return False

    def disconnect(self):
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.flush()
            self.producer.close()
        self._flush_predictions()
        elapsed = time.time() - (self._stats["start"] or time.time())
        print(f"\n[Consumer] Stopped.")
        print(f"[Consumer] consumed={self._stats['consumed']:,}  "
              f"predicted={self._stats['predicted']:,}  "
              f"alerts={self._stats['alerts']}  "
              f"elapsed={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_records: int = None):
        self._stats["start"] = time.time()

        print(f"\n{'='*60}")
        print(f"  INFERENCE CONSUMER RUNNING")
        print(f"{'='*60}")
        print(f"  Predictions → {PRED_PATH}")
        print(f"  Seq length  : {self.seq_len} steps")
        print(f"  Window size : {self.window_size} records")
        print(f"  Waiting for {self.seq_len} readings per component before predicting...")
        print(f"{'='*60}\n")

        try:
            for message in self.consumer:
                raw = message.value

                # Skip malformed messages
                if not isinstance(raw, dict) or "component_id" not in raw:
                    continue

                self._process(raw)
                self._stats["consumed"] += 1

                if max_records and self._stats["consumed"] >= max_records:
                    print(f"[Consumer] Reached max_records={max_records}. Stopping.")
                    break

                if self._stats["consumed"] % self.print_every == 0:
                    self._print_status()

        except KeyboardInterrupt:
            print("\n[Consumer] Interrupted by user.")
        finally:
            self.disconnect()

    # ------------------------------------------------------------------
    # Per-record pipeline
    # ------------------------------------------------------------------

    def _process(self, raw: dict):
        """Full pipeline: feature engineering → TGNN → save → alert."""

        # 1. Rolling feature engineering
        features = self.feature_engine.process_always(raw)
        cid      = features.get("component_id", "T01")

        if cid not in COMP_INDEX:
            return   # unknown component — skip

        # 2. Build feature vector and update this node's temporal buffer
        vec = [_safe_float(features.get(col, 0.0)) for col in FEATURE_COLS]
        with self._lock:
            self._node_buffers[cid].append(vec)

        # 3. Check if ALL nodes have enough history for the TGNN
        with self._lock:
            ready = all(len(self._node_buffers[c]) >= self.seq_len
                        for c in COMPONENT_ORDER)

        if not ready or self.model is None:
            return   # still warming up

        # 4. Build (1, T, N, F) tensor
        x = torch.zeros(1, self.seq_len, N_NODES, N_FEATURES)
        with self._lock:
            for n, comp in enumerate(COMPONENT_ORDER):
                buf = list(self._node_buffers[comp])
                for t, fvec in enumerate(buf):
                    x[0, t, n] = torch.tensor(fvec, dtype=torch.float32)
        x = x.to(self.device)

        # 5. TGNN inference
        with torch.no_grad():
            out = self.model(x, self.edge_index)

        cls_labels = out["risk_cls"][0].argmax(dim=-1)  # (N,)
        reg_scores = out["risk_reg"][0]                  # (N,)
        root_cause = out["root_cause"][0]                # (N,)

        # 6. Build and store prediction
        pred = self._build_prediction(raw, cls_labels, reg_scores, root_cause)
        self._pred_buf.append(pred)
        self._stats["predicted"] += 1

        # 7. Flush predictions to disk every 10 predictions
        if self._stats["predicted"] % 10 == 0:
            self._flush_predictions()

        # 8. Evaluate alerts and publish to Kafka
        self._handle_alerts(cls_labels, reg_scores, features)

        # 9. Publish enriched record to processed topic
        try:
            n_idx = COMP_INDEX[cid]
            enriched = {
                **features,
                "tgnn_risk_class": RISK_LABELS[cls_labels[n_idx].item()],
                "tgnn_risk_score": round(reg_scores[n_idx].item(), 4),
                "tgnn_root_cause": round(root_cause[n_idx].item(), 5),
            }
            self.producer.send(TOPICS["processed_features"], key=cid, value=enriched)
        except Exception:
            pass   # non-critical

    # ------------------------------------------------------------------
    # Build prediction record
    # ------------------------------------------------------------------

    def _build_prediction(self, raw, cls_labels, reg_scores, root_cause) -> dict:
        per_node = {}
        for n, comp in enumerate(COMPONENT_ORDER):
            per_node[comp] = {
                "risk_class":      RISK_LABELS[cls_labels[n].item()],
                "risk_score":      round(reg_scores[n].item(), 4),
                "root_cause_attn": round(root_cause[n].item(), 5),
            }

        # Sort by root cause attention
        rc_ranked = sorted(
            COMPONENT_ORDER,
            key=lambda c: -root_cause[COMP_INDEX[c]].item()
        )

        high_risk = [
            c for c in COMPONENT_ORDER
            if cls_labels[COMP_INDEX[c]].item() >= 2
        ]

        return {
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "trigger_component":  raw.get("component_id"),
            "most_likely_source": rc_ranked[0],
            "high_risk_nodes":    high_risk,
            "per_node":           per_node,
            "step":               raw.get("time_step", 0),
        }

    # ------------------------------------------------------------------
    # Alert handling
    # ------------------------------------------------------------------

    def _handle_alerts(self, cls_labels, reg_scores, features):
        for n, comp in enumerate(COMPONENT_ORDER):
            risk_score = reg_scores[n].item()
            risk_class = RISK_LABELS[cls_labels[n].item()]

            feat_for_alert = {
                "component_id":            comp,
                "risk_score":              risk_score,
                "risk_level":              risk_class,
                "is_anomaly":              cls_labels[n].item() >= 2,
                "composite_anomaly_score": features.get("composite_anomaly_score", 0.0),
                "failure_mode":            features.get("failure_mode", "normal"),
                "health_index":            features.get("health_index", 1.0),
                "record_id":               features.get("record_id", 0),
            }
            alert = self.alert_manager.evaluate(feat_for_alert)
            if alert:
                self._stats["alerts"] += 1
                alert_dict = alert.to_dict()
                print(f"  ⚠  [{alert_dict['alert_level']}] {comp}  "
                      f"risk={risk_score:.3f}  mode={alert_dict['failure_mode']}")
                try:
                    self.producer.send(TOPICS["alerts"], key=comp, value=alert_dict)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Save + status
    # ------------------------------------------------------------------

    def _flush_predictions(self):
        try:
            data = list(self._pred_buf)
            with open(PRED_PATH, "w") as fh:
                json.dump(data, fh, indent=2, default=str)
        except Exception as e:
            print(f"[Consumer] Warning: could not write predictions: {e}")

    def _print_status(self):
        elapsed = time.time() - (self._stats["start"] or time.time())
        rate    = self._stats["consumed"] / elapsed if elapsed > 0 else 0
        ready_count = sum(
            1 for c in COMPONENT_ORDER
            if len(self._node_buffers[c]) >= self.seq_len
        )
        print(f"  [{datetime.now().strftime('%H:%M:%S')}]  "
              f"consumed={self._stats['consumed']:,}  "
              f"predicted={self._stats['predicted']:,}  "
              f"alerts={self._stats['alerts']}  "
              f"rate={rate:.1f}/s  "
              f"nodes_ready={ready_count}/{N_NODES}")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Railway Inference Consumer")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--brokers",    default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--window",     type=int, default=20)
    parser.add_argument("--seq-len",    type=int, default=12)
    parser.add_argument("--max",        type=int, default=None)
    parser.add_argument("--every",      type=int, default=20)
    args = parser.parse_args()

    consumer = InferenceConsumer(
        checkpoint=args.checkpoint,
        bootstrap_servers=args.brokers,
        window_size=args.window,
        seq_len=args.seq_len,
        print_every=args.every,
    )

    if not consumer.load_model():
        return
    if not consumer.connect():
        return

    consumer.run(max_records=args.max)


if __name__ == "__main__":
    main()