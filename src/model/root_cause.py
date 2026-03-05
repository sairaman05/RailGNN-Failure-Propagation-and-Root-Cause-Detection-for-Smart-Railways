"""
Phase 4 — Root Cause Inference
Load a trained TGNN checkpoint and run inference on:
  - A single snapshot (dict of component readings)
  - A batch from the test dataset
  - Live records from the Kafka processed topic (bridging Phase 3 → 4)

Outputs per-node risk scores + root cause ranking.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch

from src.model.tgnn import TGNN, build_model
from src.model.data_loader import (
    COMPONENT_ORDER, COMP_INDEX, FEATURE_COLS, N_NODES, N_FEATURES,
    build_edge_index, _safe_float, _risk_label
)

RISK_LABELS = ["normal", "low", "medium", "high"]


# ════════════════════════════════════════════════════════════════════
# Checkpoint loader
# ════════════════════════════════════════════════════════════════════

def load_checkpoint(path: str = "checkpoints/best_model.pt",
                    device: str = None) -> TGNN:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train first:  python -m src.model.trainer"
        )
    ckpt   = torch.load(p, map_location="cpu")
    model  = build_model(ckpt.get("model_config"))
    model.load_state_dict(ckpt["model_state"])
    dev    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()
    print(f"[Inference] Loaded checkpoint: {path}")
    print(f"[Inference] Val loss at save : {ckpt.get('best_val_loss','?')}")
    return model


# ════════════════════════════════════════════════════════════════════
# Single-snapshot inference
# ════════════════════════════════════════════════════════════════════

def infer_snapshot(model: TGNN,
                   component_records: Dict[str, dict],
                   seq_len: int = 12,
                   device: str = None) -> dict:
    """
    Run inference on a single time snapshot.
    component_records: {component_id: feature_dict}

    If fewer than seq_len history steps are available,
    the snapshot is repeated to fill the window (cold-start).

    Returns dict with per-node risk scores and root cause ranking.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)

    # Build (1, seq_len, N, F) tensor
    x = torch.zeros(1, seq_len, N_NODES, N_FEATURES)
    for cid, rec in component_records.items():
        if cid not in COMP_INDEX:
            continue
        n   = COMP_INDEX[cid]
        vec = torch.tensor([_safe_float(rec.get(col, 0.0)) for col in FEATURE_COLS])
        for t in range(seq_len):
            x[0, t, n] = vec       # repeat snapshot across time steps

    edge_index = build_edge_index().to(dev)
    x = x.to(dev)

    with torch.no_grad():
        out = model(x, edge_index)

    cls_probs  = torch.softmax(out["risk_cls"][0], dim=-1)  # (N, 4)
    cls_labels = cls_probs.argmax(dim=-1)                   # (N,)
    reg_scores = out["risk_reg"][0]                         # (N,)
    rc_attn    = out["root_cause"][0]                       # (N,)

    results = {}
    for n, cid in enumerate(COMPONENT_ORDER):
        results[cid] = {
            "risk_class":     RISK_LABELS[cls_labels[n].item()],
            "risk_score":     round(reg_scores[n].item(), 4),
            "class_probs":    {RISK_LABELS[i]: round(cls_probs[n,i].item(), 4) for i in range(4)},
            "root_cause_attn": round(rc_attn[n].item(), 5),
        }

    # Root cause ranking
    rc_ranking = sorted(
        [(cid, results[cid]["root_cause_attn"]) for cid in COMPONENT_ORDER],
        key=lambda x: -x[1]
    )

    return {
        "per_node":           results,
        "root_cause_ranking": [{"component": c, "score": s} for c, s in rc_ranking],
        "most_likely_source": rc_ranking[0][0],
        "high_risk_nodes":    [c for c, d in results.items() if d["risk_class"] in ("high","medium")],
    }


# ════════════════════════════════════════════════════════════════════
# Batch inference on a list of processed records (Phase 3 output)
# ════════════════════════════════════════════════════════════════════

def infer_from_records(model: TGNN,
                       records: List[dict],
                       seq_len: int = 12,
                       device: str = None) -> List[dict]:
    """
    Run inference over a stream of processed records (like Phase 3 output).
    Groups records by time_step, builds sliding windows, returns predictions.
    """
    from collections import defaultdict
    from src.model.data_loader import SequenceBuilder

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    edge_index = build_edge_index().to(dev)

    builder = SequenceBuilder(seq_len=seq_len, stride=1)
    X, y_cls, y_reg = builder.build(records)

    if len(X) == 0:
        print("[Inference] No sequences built from records.")
        return []

    model.eval()
    predictions = []
    batch_size  = 64

    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size].to(dev)
        with torch.no_grad():
            out = model(xb, edge_index)

        for b in range(len(xb)):
            cls_labels = out["risk_cls"][b].argmax(dim=-1)
            reg_scores = out["risk_reg"][b]
            rc         = out["root_cause"][b]
            rc_ranked  = sorted(zip(COMPONENT_ORDER, rc.tolist()), key=lambda x:-x[1])
            predictions.append({
                "window_index": i + b,
                "per_node": {
                    cid: {
                        "risk_class": RISK_LABELS[cls_labels[n].item()],
                        "risk_score": round(reg_scores[n].item(), 4),
                        "root_cause_attn": round(rc[n].item(), 5),
                    }
                    for n, cid in enumerate(COMPONENT_ORDER)
                },
                "most_likely_source": rc_ranked[0][0],
                "high_risk_nodes": [
                    cid for n, cid in enumerate(COMPONENT_ORDER)
                    if cls_labels[n].item() >= 2
                ],
            })

    return predictions


# ════════════════════════════════════════════════════════════════════
# CLI — quick demo inference
# ════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TGNN Inference — Phase 4")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--records",    default="data/processed/processed_features_sample.json",
                        help="Path to Phase 3 processed features JSON")
    parser.add_argument("--limit",      type=int, default=500)
    parser.add_argument("--save",       default="data/processed/inference_output.json")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)

    # Load Phase 3 records
    rpath = Path(args.records)
    if not rpath.exists():
        print(f"[Inference] {args.records} not found.")
        print("           Run:  python -m src.kafka.simulate")
        return

    with open(rpath) as fh:
        records = json.load(fh)[:args.limit]

    print(f"[Inference] Running on {len(records)} records...")
    preds = infer_from_records(model, records)

    # Save
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as fh:
        json.dump(preds[:100], fh, indent=2)  # save first 100 windows
    print(f"[Inference] Saved {len(preds)} predictions → {args.save}")

    # Print sample
    if preds:
        sample = preds[0]
        print(f"\n  Sample prediction (window 0):")
        print(f"  Most likely failure source : {sample['most_likely_source']}")
        print(f"  High/medium risk nodes     : {sample['high_risk_nodes']}")
        print(f"\n  Per-node summary:")
        for cid, d in sample["per_node"].items():
            if d["risk_class"] != "normal":
                print(f"    {cid:<6}  {d['risk_class']:<8}  "
                      f"score={d['risk_score']:.3f}  "
                      f"rc_attn={d['root_cause_attn']:.4f}")


if __name__ == "__main__":
    main()