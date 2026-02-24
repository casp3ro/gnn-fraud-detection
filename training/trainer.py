"""
Training loop for GNN and MLP: same interface, BCEWithLogitsLoss with pos_weight,
early stopping on validation metric, and evaluation on test set.
"""

from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data

from training.metrics import compute_confusion_matrix, compute_metrics, logits_to_pred_and_prob


def get_pos_weight(y_train: torch.Tensor) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    pos_weight = (num_negatives / num_positives) so that the loss weight for
    the positive class is higher and the model is encouraged to catch frauds.
    """
    y = y_train.ravel()
    num_pos = (y == 1).sum().item()
    num_neg = (y == 0).sum().item()
    if num_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor([num_neg / num_pos], dtype=torch.float32)


class Trainer:
    """
    Generic trainer for any model that takes PyG Data and returns logits (N, 1).

    Trains on data.train_mask, validates on data.val_mask, and evaluates on data.test_mask.
    Uses early stopping based on validation F1 (or another metric).
    """

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        device: str,
        lr: float = 0.01,
        max_epochs: int = 200,
        patience: int = 25,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience

        # Loss: BCEWithLogitsLoss with pos_weight for imbalance.
        if pos_weight is None:
            pos_weight = get_pos_weight(data.y[data.train_mask]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = Adam(model.parameters(), lr=lr)

    def train_epoch(self) -> float:
        """One training epoch; returns average training loss."""
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.data)
        # Loss only on training nodes.
        loss = self.criterion(logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, mask: torch.Tensor) -> Dict[str, float]:
        """Compute metrics on nodes where mask is True."""
        self.model.eval()
        logits = self.model(self.data)
        logits_m = logits[mask]
        y_true = self.data.y[mask].cpu().numpy().ravel()
        y_pred, y_prob = logits_to_pred_and_prob(logits_m)
        return compute_metrics(y_true, y_pred, y_prob)

    def run(self) -> Dict[str, float]:
        """
        Full training with early stopping on validation F1.
        Returns test-set metrics from the best epoch (by val F1).
        """
        best_val_f1 = -1.0
        best_state: Optional[Dict] = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.data.val_mask)
            val_f1 = val_metrics["f1"]

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1:3d}  loss={train_loss:.4f}  val_f1={val_f1:.4f}  "
                    f"val_recall={val_metrics['recall']:.4f}  val_precision={val_metrics['precision']:.4f}"
                )

            if epochs_no_improve >= self.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Restore best model and evaluate on test set.
        if best_state is not None:
            self.model.load_state_dict(best_state)
        test_metrics = self.evaluate(self.data.test_mask)
        return test_metrics

    def test_confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix on test set (after run())."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data)
        mask = self.data.test_mask
        _, y_pred = logits_to_pred_and_prob(logits[mask])
        y_pred_bin = (y_pred >= 0.5).astype(np.int64)
        y_true = self.data.y[mask].cpu().numpy().ravel().astype(np.int64)
        return compute_confusion_matrix(y_true, y_pred_bin)
