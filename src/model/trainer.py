"""
Phase 4 — Trainer
Full training loop with:
  - Combined classification + regression loss
  - Early stopping
  - Best-checkpoint saving
  - Per-epoch metric logging to JSON
"""

import json
import time
import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.tgnn        import TGNN, build_model
from src.model.data_loader import RailwayDataset, load_from_json, load_from_csv


# ════════════════════════════════════════════════════════════════════
# Loss
# ════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    cls_weight * CrossEntropy(risk_cls, y_cls)
  + reg_weight * MSE(risk_reg, y_reg)
    """

    def __init__(self, cls_weight: float = 0.7, reg_weight: float = 0.3,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ce  = nn.CrossEntropyLoss(weight=class_weights)
        self.mse = nn.MSELoss()

    def forward(self, pred: dict, y_cls: torch.Tensor,
                y_reg: torch.Tensor) -> torch.Tensor:
        B, N, C = pred["risk_cls"].shape

        # Reshape for CrossEntropy: (B*N, C) and (B*N,)
        cls_loss = self.ce(
            pred["risk_cls"].view(B * N, C),
            y_cls.view(B * N)
        )
        reg_loss = self.mse(pred["risk_reg"], y_reg)

        return self.cls_weight * cls_loss + self.reg_weight * reg_loss


# ════════════════════════════════════════════════════════════════════
# Metrics helper
# ════════════════════════════════════════════════════════════════════

def _accuracy(risk_cls: torch.Tensor, y_cls: torch.Tensor) -> float:
    """Node-level classification accuracy."""
    preds = risk_cls.argmax(dim=-1)          # (B, N)
    return (preds == y_cls).float().mean().item()


def _mae(risk_reg: torch.Tensor, y_reg: torch.Tensor) -> float:
    return (risk_reg - y_reg).abs().mean().item()


# ════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════

class Trainer:

    def __init__(self,
                 model:         TGNN,
                 train_dataset: RailwayDataset,
                 val_dataset:   RailwayDataset,
                 # Training hyperparams
                 epochs:        int   = 50,
                 batch_size:    int   = 32,
                 lr:            float = 1e-3,
                 weight_decay:  float = 1e-4,
                 # Loss weights
                 cls_weight:    float = 0.7,
                 reg_weight:    float = 0.3,
                 # Early stopping
                 patience:      int   = 8,
                 min_delta:     float = 1e-4,
                 # Paths
                 checkpoint_dir: str  = "checkpoints",
                 log_path:       str  = "data/processed/training_log.json",
                 # Device
                 device:        str   = None):

        self.model     = model
        self.epochs    = epochs
        self.patience  = patience
        self.min_delta = min_delta
        self.log_path  = Path(log_path)
        self.ckpt_dir  = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        print(f"[Trainer] Device       : {self.device}")
        print(f"[Trainer] Model params : {model.count_params():,}")

        # Class weights computed dynamically from training data distribution
        # Formula: w_c = total_samples / (n_classes * count_c)
        # This automatically handles any imbalance ratio
        all_labels = train_dataset.y_cls.view(-1).tolist()
        n_total    = len(all_labels)
        n_classes  = 4
        counts     = [max(1, all_labels.count(c)) for c in range(n_classes)]
        weights    = [n_total / (n_classes * cnt) for cnt in counts]
        # Cap max weight at 20x to avoid instability on very rare classes
        max_w      = max(weights)
        weights    = [min(w, 20.0) for w in weights]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        print(f"[Trainer] Class weights: "
              f"normal={weights[0]:.2f}  low={weights[1]:.2f}  "
              f"medium={weights[2]:.2f}  high={weights[3]:.2f}")
        self.criterion = CombinedLoss(cls_weight, reg_weight, class_weights)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=4, factor=0.5
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True
        )
        self.val_loader   = DataLoader(
            val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False
        )

        # Edge index to device (same for all batches)
        self._edge_index = train_dataset.edge_index.to(self.device)

        self.history = []

    # ------------------------------------------------------------------
    # One epoch
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss = total_acc = total_mae = 0.0
        n_batches  = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y_cls, y_reg, _ in loader:
                x     = x.to(self.device)          # (B, T, N, F)
                y_cls = y_cls.to(self.device)       # (B, N)
                y_reg = y_reg.to(self.device)       # (B, N)

                pred = self.model(x, self._edge_index)
                loss = self.criterion(pred, y_cls, y_reg)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                total_acc  += _accuracy(pred["risk_cls"], y_cls)
                total_mae  += _mae(pred["risk_reg"], y_reg)
                n_batches  += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "acc":  total_acc  / max(n_batches, 1),
            "mae":  total_mae  / max(n_batches, 1),
        }

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def train(self) -> dict:
        best_val_loss  = float("inf")
        best_state     = None
        patience_count = 0
        start          = time.time()

        print(f"\n{'='*60}")
        print(f"  PHASE 4 — TGNN TRAINING")
        print(f"{'='*60}")
        print(f"  Epochs     : {self.epochs}")
        print(f"  Batch size : {self.train_loader.batch_size}")
        print(f"  Train size : {len(self.train_loader.dataset)}")
        print(f"  Val size   : {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            t_metrics = self._run_epoch(self.train_loader, train=True)
            v_metrics = self._run_epoch(self.val_loader,   train=False)

            self.scheduler.step(v_metrics["loss"])

            improved = v_metrics["loss"] < best_val_loss - self.min_delta
            if improved:
                best_val_loss  = v_metrics["loss"]
                best_state     = copy.deepcopy(self.model.state_dict())
                patience_count = 0
                tag = " ✓ best"
            else:
                patience_count += 1
                tag = f" (patience {patience_count}/{self.patience})"

            row = {
                "epoch":     epoch,
                "train_loss": round(t_metrics["loss"], 5),
                "train_acc":  round(t_metrics["acc"],  4),
                "train_mae":  round(t_metrics["mae"],  5),
                "val_loss":   round(v_metrics["loss"], 5),
                "val_acc":    round(v_metrics["acc"],  4),
                "val_mae":    round(v_metrics["mae"],  5),
            }
            self.history.append(row)

            print(f"  Epoch {epoch:03d}/{self.epochs}  "
                  f"train_loss={t_metrics['loss']:.4f}  acc={t_metrics['acc']:.3f}  "
                  f"| val_loss={v_metrics['loss']:.4f}  acc={v_metrics['acc']:.3f}{tag}")

            if patience_count >= self.patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch}.")
                break

        # ── Restore best weights ─────────────────────────────────────
        if best_state:
            self.model.load_state_dict(best_state)

        elapsed = time.time() - start
        print(f"\n[Trainer] Training complete in {elapsed:.1f}s")
        print(f"[Trainer] Best val loss: {best_val_loss:.5f}")

        # ── Save checkpoint ──────────────────────────────────────────
        ckpt_path = self.ckpt_dir / "best_model.pt"
        torch.save({
            "model_state":  self.model.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.history),
            "model_config": {
                "in_features": self.model.in_features,
                "gcn_hidden":  self.model.gcn_hidden,
                "gru_hidden":  self.model.gru_hidden,
                "n_classes":   self.model.n_classes,
            }
        }, ckpt_path)
        print(f"[Trainer] Checkpoint saved → {ckpt_path}")

        # ── Save training log ─────────────────────────────────────────
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as fh:
            json.dump(self.history, fh, indent=2)
        print(f"[Trainer] Training log  → {self.log_path}")

        return {"best_val_loss": best_val_loss, "history": self.history}


# ════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train TGNN — Phase 4")
    parser.add_argument("--source",   default="json",
                        choices=["json","csv"], help="Data source")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--patience", type=int,   default=8)
    parser.add_argument("--seq-len",  type=int,   default=12)
    parser.add_argument("--stride",   type=int,   default=1)
    args = parser.parse_args()

    # Load data
    if args.source == "json":
        train_ds, val_ds, test_ds = load_from_json(seq_len=args.seq_len, stride=args.stride, horizon=6)
    else:
        train_ds, val_ds, test_ds = load_from_csv(seq_len=args.seq_len, stride=args.stride, horizon=6)

    # Build model
    model = build_model()
    print(f"[Main] TGNN architecture:\n{model}")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
    )
    trainer.train()

    # Quick test evaluation
    from src.model.evaluator import Evaluator
    ev = Evaluator(model, test_ds)
    ev.evaluate()


if __name__ == "__main__":
    main()