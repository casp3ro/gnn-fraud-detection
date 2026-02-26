"""
Entry point: load config, build graph dataset, train GNN and MLP, compare results,
and optionally generate Matplotlib visualizations.

Run from the fraud-detection project directory:
    cd projects/fraud-detection && python run.py
"""

from pathlib import Path

import numpy as np
import torch

from config import Config
from data import prepare_fraud_data
from models import FraudGNN, FraudMLP
from training import Trainer
from visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_roc_pr_curves,
    plot_sample_graph,
    plot_training_curves,
    save_history,
    save_predictions,
)


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

    # --- Save histories and predictions for later analysis ---
    vis_cfg = cfg.visualization
    metrics_dir: Path = vis_cfg.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    gnn_history = trainer_gnn.get_history()
    mlp_history = trainer_mlp.get_history()
    save_history(gnn_history, metrics_dir / "gnn_history.npz")
    save_history(mlp_history, metrics_dir / "mlp_history.npz")

    gnn_preds = trainer_gnn.get_test_predictions()
    mlp_preds = trainer_mlp.get_test_predictions()
    save_predictions(gnn_preds, metrics_dir / "gnn_test_preds.npz")
    save_predictions(mlp_preds, metrics_dir / "mlp_test_preds.npz")

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

    # --- Visualization (optional) ---
    if vis_cfg.enable_plots:
        plots_dir: Path = vis_cfg.plots_dir
        plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating plots in {plots_dir} ...")

        plot_training_curves(
            history_gnn=gnn_history,
            history_mlp=mlp_history,
            out_dir=plots_dir,
            figsize=vis_cfg.figsize,
        )
        plot_confusion_matrices(
            gnn_cm=gnn_cm,
            mlp_cm=mlp_cm,
            out_dir=plots_dir,
            figsize=(vis_cfg.figsize[0] * 1.5, vis_cfg.figsize[1]),
        )
        plot_roc_pr_curves(
            gnn_probs=gnn_preds["y_prob"],
            mlp_probs=mlp_preds["y_prob"],
            y_true=gnn_preds["y_true"],
            out_dir=plots_dir,
            figsize=vis_cfg.figsize,
        )
        plot_class_distribution(
            data=data,
            out_dir=plots_dir,
            figsize=vis_cfg.figsize,
        )
        # This plot may be relatively heavy; ignore return value.
        plot_sample_graph(
            data=data,
            out_dir=plots_dir,
            num_nodes=200,
        )

        print("Plots saved. You can explore them and the .npz files for deeper analysis.")
    print("\nDone.")


if __name__ == "__main__":
    main()
