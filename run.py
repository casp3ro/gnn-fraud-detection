"""
Entry point: load config, build graph dataset, train GNN and MLP, compare results.

Run from the fraud-detection project directory:
    cd projects/fraud-detection && python run.py
"""

import numpy as np
import torch

from config import Config
from data import prepare_fraud_data
from models import FraudGNN, FraudMLP
from training import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = Config.from_defaults()
    set_seed(cfg.training.seed)
    device = cfg.training.device
    print(f"Device: {device}")

    # Load and prepare graph data (normalize, split, k-NN graph).
    print("Loading dataset and building k-NN graph...")
    data = prepare_fraud_data(
        csv_path=cfg.data.data_path,
        k_neighbors=cfg.data.k_neighbors,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.training.seed,
    )
    data = data.to(device)
    print(
        f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, "
        f"Train/Val/Test: {data.train_mask.sum().item()}/{data.val_mask.sum().item()}/{data.test_mask.sum().item()}"
    )

    # Shared model config.
    model_cfg = cfg.model
    hidden_dims = tuple(model_cfg.hidden_dims)

    # --- Train GNN ---
    print("\n--- Training GNN (graph-based) ---")
    gnn = FraudGNN(
        input_dim=model_cfg.input_dim,
        hidden_dims=hidden_dims,
        dropout=model_cfg.dropout,
    )
    trainer_gnn = Trainer(
        model=gnn,
        data=data,
        device=device,
        lr=cfg.training.learning_rate,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
    )
    gnn_metrics = trainer_gnn.run()
    gnn_cm = trainer_gnn.test_confusion_matrix()

    # --- Train MLP baseline ---
    print("\n--- Training MLP baseline (non-graph) ---")
    mlp = FraudMLP(
        input_dim=model_cfg.input_dim,
        hidden_dims=hidden_dims,
        dropout=model_cfg.dropout,
    )
    trainer_mlp = Trainer(
        model=mlp,
        data=data,
        device=device,
        lr=cfg.training.learning_rate,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
    )
    mlp_metrics = trainer_mlp.run()
    mlp_cm = trainer_mlp.test_confusion_matrix()

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON (test set)")
    print("=" * 60)
    print(f"{'Metric':<12}  {'GNN':>10}  {'MLP':>10}")
    print("-" * 36)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        gv = gnn_metrics.get(key, 0.0)
        mv = mlp_metrics.get(key, 0.0)
        print(f"{key:<12}  {gv:>10.4f}  {mv:>10.4f}")
    print("\nConfusion matrices (rows=true, cols=predicted, order=[legit, fraud]):")
    print("GNN:")
    print(gnn_cm)
    print("MLP:")
    print(mlp_cm)
    print("\nDone.")


if __name__ == "__main__":
    main()
