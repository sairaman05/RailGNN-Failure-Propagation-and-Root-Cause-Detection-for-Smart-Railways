"""
Phase 4 — Evaluator
Computes classification and regression metrics on test set.
Saves results to data/processed/evaluation_results.json.
No sklearn required — all metrics computed with PyTorch/pure Python.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from typing import List

import torch
from torch.utils.data import DataLoader

from src.model.tgnn        import TGNN
from src.model.data_loader import RailwayDataset, COMPONENT_ORDER

RISK_LABELS = ["normal", "low", "medium", "high"]


# ════════════════════════════════════════════════════════════════════
# Pure-Python metric helpers (no sklearn)
# ════════════════════════════════════════════════════════════════════

def _confusion_matrix(y_true: List[int], y_pred: List[int], n: int) -> List[List[int]]:
    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def _precision_recall_f1(cm: List[List[int]], cls: int):
    tp = cm[cls][cls]
    fp = sum(cm[r][cls] for r in range(len(cm))) - tp
    fn = sum(cm[cls][c] for c in range(len(cm))) - tp
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p+r) if (p+r) > 0 else 0.0
    return round(p,4), round(r,4), round(f1,4)


def _roc_auc_binary(scores: List[float], labels: List[int]) -> float:
    """Compute AUC-ROC for binary classification (positive = class > 0)."""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = auc = 0.0
    prev_fp = prev_tp = 0.0
    for score, label in paired:
        if label > 0:
            tp += 1
        else:
            fp += 1
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_fp, prev_tp = fp, tp
    return round(auc / (n_pos * n_neg), 4)


# ════════════════════════════════════════════════════════════════════
# Evaluator
# ════════════════════════════════════════════════════════════════════

class Evaluator:

    def __init__(self, model: TGNN, dataset: RailwayDataset,
                 batch_size: int = 64,
                 output_path: str = "data/processed/evaluation_results.json",
                 device: str = None):

        self.model   = model
        self.dataset = dataset
        self.loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.out_path = Path(output_path)
        self.device   = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self._edge_index = dataset.edge_index.to(self.device)

    # ------------------------------------------------------------------
    # Collect all predictions
    # ------------------------------------------------------------------

    def _collect(self):
        self.model.eval()
        all_cls_true, all_cls_pred = [], []
        all_reg_true, all_reg_pred = [], []
        all_root_cause             = []

        with torch.no_grad():
            for x, y_cls, y_reg, _ in self.loader:
                x     = x.to(self.device)
                y_cls = y_cls.to(self.device)
                y_reg = y_reg.to(self.device)

                out = self.model(x, self._edge_index)

                cls_pred = out["risk_cls"].argmax(dim=-1)  # (B, N)

                all_cls_true.append(y_cls.cpu())
                all_cls_pred.append(cls_pred.cpu())
                all_reg_true.append(y_reg.cpu())
                all_reg_pred.append(out["risk_reg"].cpu())
                all_root_cause.append(out["root_cause"].cpu())

        cls_true   = torch.cat(all_cls_true,  dim=0)  # (total, N)
        cls_pred   = torch.cat(all_cls_pred,  dim=0)
        reg_true   = torch.cat(all_reg_true,  dim=0)
        reg_pred   = torch.cat(all_reg_pred,  dim=0)
        root_cause = torch.cat(all_root_cause, dim=0)

        return cls_true, cls_pred, reg_true, reg_pred, root_cause

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self, verbose: bool = True) -> dict:
        cls_true, cls_pred, reg_true, reg_pred, root_cause = self._collect()

        N = cls_true.shape[1]
        total_samples = cls_true.shape[0]

        # ── Flatten for overall metrics ──────────────────────────────
        ct_flat = cls_true.view(-1).tolist()
        cp_flat = cls_pred.view(-1).tolist()
        rt_flat = reg_true.view(-1).tolist()
        rp_flat = reg_pred.view(-1).tolist()

        # ── Overall accuracy ─────────────────────────────────────────
        overall_acc = sum(a == b for a, b in zip(ct_flat, cp_flat)) / len(ct_flat)

        # ── Confusion matrix ─────────────────────────────────────────
        cm = _confusion_matrix(ct_flat, cp_flat, 4)

        # ── Per-class P/R/F1 ─────────────────────────────────────────
        per_class = {}
        for i, label in enumerate(RISK_LABELS):
            p, r, f1 = _precision_recall_f1(cm, i)
            per_class[label] = {"precision": p, "recall": r, "f1": f1}

        # ── AUC-ROC (binary: any risk vs normal) ─────────────────────
        binary_labels = [1 if t > 0 else 0 for t in ct_flat]
        # Score = probability of non-normal = 1 - P(class=0)
        # Approximate with predicted class > 0
        binary_scores = [1.0 if p > 0 else 0.0 for p in cp_flat]
        auc = _roc_auc_binary(binary_scores, binary_labels)

        # ── Regression metrics ───────────────────────────────────────
        mae  = sum(abs(a-b) for a,b in zip(rt_flat, rp_flat)) / len(rt_flat)
        mse  = sum((a-b)**2 for a,b in zip(rt_flat, rp_flat)) / len(rt_flat)
        rmse = math.sqrt(mse)

        # ── Per-node metrics ─────────────────────────────────────────
        per_node = {}
        for n, cid in enumerate(COMPONENT_ORDER):
            nt = cls_true[:, n].tolist()
            np_ = cls_pred[:, n].tolist()
            acc = sum(a==b for a,b in zip(nt,np_)) / len(nt)
            reg_mae = (reg_true[:, n] - reg_pred[:, n]).abs().mean().item()
            per_node[cid] = {
                "accuracy": round(acc, 4),
                "reg_mae":  round(reg_mae, 5),
            }

        # ── Root cause analysis ───────────────────────────────────────
        # Average root cause attention across all windows
        mean_rc = root_cause.mean(dim=0)  # (N,)
        rc_ranked = sorted(
            zip(COMPONENT_ORDER, mean_rc.tolist()),
            key=lambda x: -x[1]
        )
        root_cause_ranking = [
            {"component": c, "score": round(s, 5)} for c, s in rc_ranked
        ]

        results = {
            "overall": {
                "accuracy":    round(overall_acc, 4),
                "auc_roc":     auc,
                "mae":         round(mae,  5),
                "rmse":        round(rmse, 5),
                "total_samples": total_samples,
                "total_nodes":   N,
            },
            "per_class":          per_class,
            "confusion_matrix":   cm,
            "per_node":           per_node,
            "root_cause_ranking": root_cause_ranking,
        }

        # ── Save ─────────────────────────────────────────────────────
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as fh:
            json.dump(results, fh, indent=2)

        if verbose:
            self._print(results)

        print(f"\n[Evaluator] Results saved → {self.out_path}")
        return results

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def _print(self, r: dict):
        o = r["overall"]
        print(f"\n{'='*60}")
        print(f"  PHASE 4 — EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Overall accuracy : {o['accuracy']:.4f}")
        print(f"  AUC-ROC (binary) : {o['auc_roc']:.4f}")
        print(f"  MAE (reg)        : {o['mae']:.5f}")
        print(f"  RMSE (reg)       : {o['rmse']:.5f}")
        print(f"\n  Per-class metrics:")
        print(f"  {'Class':<10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
        print(f"  {'-'*42}")
        for cls, m in r["per_class"].items():
            print(f"  {cls:<10}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  {m['f1']:>8.4f}")

        print(f"\n  Top root cause components:")
        for entry in r["root_cause_ranking"][:5]:
            bar = "█" * int(entry["score"] * 200)
            print(f"    {entry['component']:<6}  {entry['score']:.4f}  {bar}")

        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        labels = ["NRM","LOW","MED","HGH"]
        print(f"         {'  '.join(f'{l:>4}' for l in labels)}")
        for i, row in enumerate(r["confusion_matrix"]):
            print(f"  {labels[i]:>4}  {'  '.join(f'{v:>4}' for v in row)}")
        print(f"{'='*60}\n")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    import argparse
    from src.model.data_loader import load_from_json, load_from_csv

    parser = argparse.ArgumentParser(description="Evaluate TGNN — Phase 4")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--source",     default="json", choices=["json","csv"])
    parser.add_argument("--seq-len",    type=int, default=12)
    args = parser.parse_args()

    # Load data
    if args.source == "json":
        _, _, test_ds = load_from_json(seq_len=args.seq_len)
    else:
        _, _, test_ds = load_from_csv(seq_len=args.seq_len)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    from src.model.tgnn import build_model
    model = build_model(ckpt.get("model_config"))
    model.load_state_dict(ckpt["model_state"])
    print(f"[Eval] Loaded checkpoint: {args.checkpoint}")
    print(f"[Eval] Best val loss was: {ckpt.get('best_val_loss','?')}")

    ev = Evaluator(model, test_ds)
    ev.evaluate()


if __name__ == "__main__":
    main()