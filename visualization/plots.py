from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _ensure_dir(path: Path | str) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _to_array(seq: Sequence[float]) -> np.ndarray:
    return np.asarray(list(seq), dtype=float)


def plot_training_curves(
    history_gnn: Mapping[str, Sequence[float]],
    history_mlp: Mapping[str, Sequence[float]],
    out_dir: Path | str,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> Path:
    """
    Plot training/validation loss and validation F1 curves for GNN and MLP.
    Expects histories as produced by Trainer.get_history().
    """
    out_dir = _ensure_dir(out_dir)
    epochs_gnn = _to_array(history_gnn.get("epoch", []))
    epochs_mlp = _to_array(history_mlp.get("epoch", []))

    plt.figure(figsize=(figsize[0], figsize[1] * 1.2))

    # Loss subplot.
    plt.subplot(2, 1, 1)
    if len(epochs_gnn) > 0:
        plt.plot(epochs_gnn, _to_array(history_gnn.get("train_loss", [])), label="GNN train loss")
        plt.plot(epochs_gnn, _to_array(history_gnn.get("val_loss", [])), label="GNN val loss")
    if len(epochs_mlp) > 0:
        plt.plot(epochs_mlp, _to_array(history_mlp.get("train_loss", [])), "--", label="MLP train loss")
        plt.plot(epochs_mlp, _to_array(history_mlp.get("val_loss", [])), "--", label="MLP val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1 subplot.
    plt.subplot(2, 1, 2)
    if len(epochs_gnn) > 0:
        plt.plot(epochs_gnn, _to_array(history_gnn.get("val_f1", [])), label="GNN val F1")
    if len(epochs_mlp) > 0:
        plt.plot(epochs_mlp, _to_array(history_mlp.get("val_f1", [])), "--", label="MLP val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("Validation F1 over epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_confusion_matrices(
    gnn_cm: np.ndarray,
    mlp_cm: np.ndarray,
    out_dir: Path | str,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> Path:
    """
    Plot side-by-side confusion matrices for GNN and MLP.
    Assumes format [[TN, FP], [FN, TP]] with label order [0, 1].
    """
    out_dir = _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    labels = ["legit (0)", "fraud (1)"]
    sns.heatmap(
        gnn_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("GNN confusion matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        mlp_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("MLP confusion matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    out_path = out_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_roc_pr_curves(
    gnn_probs: np.ndarray,
    mlp_probs: np.ndarray,
    y_true: np.ndarray,
    out_dir: Path | str,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> tuple[Path, Path]:
    """
    Plot ROC and Precision-Recall curves for GNN and MLP on the same axes.
    """
    out_dir = _ensure_dir(out_dir)
    y_true = y_true.astype(int).ravel()
    gnn_probs = gnn_probs.ravel()
    mlp_probs = mlp_probs.ravel()

    # Guard against degenerate cases without positive labels.
    if len(np.unique(y_true)) < 2:
        return out_dir / "roc_curves_skipped.png", out_dir / "pr_curves_skipped.png"

    # ROC.
    fpr_gnn, tpr_gnn, _ = roc_curve(y_true, gnn_probs)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_true, mlp_probs)
    auc_gnn = auc(fpr_gnn, tpr_gnn)
    auc_mlp = auc(fpr_mlp, tpr_mlp)

    plt.figure(figsize=figsize)
    plt.plot(fpr_gnn, tpr_gnn, label=f"GNN (AUC={auc_gnn:.3f})")
    plt.plot(fpr_mlp, tpr_mlp, label=f"MLP (AUC={auc_mlp:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_path = out_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Precision-Recall.
    prec_gnn, rec_gnn, _ = precision_recall_curve(y_true, gnn_probs)
    prec_mlp, rec_mlp, _ = precision_recall_curve(y_true, mlp_probs)
    pr_gnn = auc(rec_gnn, prec_gnn)
    pr_mlp = auc(rec_mlp, prec_mlp)

    plt.figure(figsize=figsize)
    plt.plot(rec_gnn, prec_gnn, label=f"GNN (AUPR={pr_gnn:.3f})")
    plt.plot(rec_mlp, prec_mlp, label=f"MLP (AUPR={pr_mlp:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    pr_path = out_dir / "pr_curves.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()

    return roc_path, pr_path


def plot_class_distribution(
    data,
    out_dir: Path | str,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> Path:
    """
    Plot class distribution (legit vs fraud) using data.y from PyG Data.
    """
    out_dir = _ensure_dir(out_dir)
    y = data.y.view(-1).cpu().numpy()
    # Labels are typically 0/1 floats; cast to int for clarity.
    y = y.astype(int)
    unique, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=figsize)
    plt.bar(unique, counts, tick_label=[str(int(c)) for c in unique])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class distribution (0 = legit, 1 = fraud)")
    for x, c in zip(unique, counts):
        plt.text(x, c, str(int(c)), ha="center", va="bottom")
    plt.tight_layout()

    out_path = out_dir / "class_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_sample_graph(
    data,
    out_dir: Path | str,
    num_nodes: int = 200,
    seed: int = 42,
) -> Path | None:
    """
    Plot a small subgraph of the transaction graph.

    Attempts to use torch_geometric.utils.to_networkx and networkx; if they are
    not available, this function silently returns None.
    """
    try:
        import networkx as nx  # type: ignore[import]
        from torch_geometric.data import Data as PyGData
        from torch_geometric.utils import subgraph, to_networkx
    except ImportError:
        # Optional dependency; skip graph plot if missing.
        return None

    out_dir = _ensure_dir(out_dir)

    num_total = int(data.num_nodes)
    if num_total == 0:
        return None

    rng = np.random.default_rng(seed)
    k = min(max(num_nodes, 1), num_total)
    subset_idx = np.sort(rng.choice(num_total, size=k, replace=False))

    node_idx = np.asarray(subset_idx, dtype=np.int64)
    # Long tensor indices on the same device as data.x.
    node_idx_t = torch.as_tensor(node_idx, dtype=torch.long, device=data.x.device)

    sub_edge_index, _ = subgraph(node_idx_t, data.edge_index, relabel_nodes=True)
    sub_data = PyGData(
        x=data.x[node_idx_t],
        edge_index=sub_edge_index,
        y=data.y[node_idx_t],
    )

    G = to_networkx(sub_data, to_undirected=True)
    y_sub = sub_data.y.view(-1).cpu().numpy().astype(int)

    # Map labels to colors: legit=0 -> lightblue, fraud=1 -> orange.
    color_map = ["#4c72b0" if label == 0 else "#dd8452" for label in y_sub]

    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        node_color=color_map,
        node_size=40,
        edge_color="#bbbbbb",
        linewidths=0.5,
        width=0.5,
        with_labels=False,
    )
    plt.title("Sample transaction subgraph (color by class)")
    plt.tight_layout()

    out_path = out_dir / "sample_graph.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_history(history: Mapping[str, Sequence[float]], out_path: Path | str) -> Path:
    """
    Save training history as a compressed .npz file for later analysis.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np_history = {k: _to_array(v) for k, v in history.items()}
    np.savez_compressed(out_path, **np_history)
    return out_path


def save_predictions(predictions: Mapping[str, np.ndarray], out_path: Path | str) -> Path:
    """
    Save test-set predictions (y_true, y_pred, y_prob) as a compressed .npz file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np_preds = {k: np.asarray(v) for k, v in predictions.items()}
    np.savez_compressed(out_path, **np_preds)
    return out_path

