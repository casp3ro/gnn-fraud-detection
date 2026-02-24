"""
Dataset loading and PyG Data construction for fraud detection.

Loads fraud_data.csv, normalizes features (fit on train only to avoid leakage),
splits into train/val/test with stratification, builds k-NN graph, and
returns a single torch_geometric.data.Data object with masks.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Local graph builder: k-NN edges from feature matrix.
from data.graph_builder import build_knn_graph


# Column names in fraud_data.csv: V1..V28, Amount, Class.
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def load_fraud_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load fraud CSV and return feature matrix (X) and labels (y).

    Args:
        csv_path: Path to fraud_data.csv.

    Returns:
        X: DataFrame with columns V1–V28 and Amount.
        y: Series with Class (0 or 1).
    """
    df = pd.read_csv(csv_path)
    # Drop any row with missing values (optional; dataset is usually complete).
    df = df.dropna()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def prepare_fraud_data(
    csv_path: Path,
    k_neighbors: int = 10,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Data:
    """
    Load CSV, normalize features, split with stratification, build k-NN graph,
    and return a PyG Data object with x, edge_index, y, and train/val/test masks.

    Normalization is fit on the training set only, then applied to val and test,
    to avoid data leakage.

    Args:
        csv_path: Path to fraud_data.csv.
        k_neighbors: k for k-NN graph.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed for reproducibility.

    Returns:
        PyG Data with .x, .edge_index, .y, .train_mask, .val_mask, .test_mask.
    """
    X_df, y_series = load_fraud_dataset(csv_path)
    X = X_df.values.astype(np.float32)
    y = y_series.values.astype(np.int64)

    n = len(y)
    # First split: train vs (val + test).
    train_ratio_here = train_ratio
    val_plus_test_ratio = val_ratio + test_ratio
    indices = np.arange(n)
    idx_train, idx_rest = train_test_split(
        indices,
        test_size=val_plus_test_ratio,
        stratify=y,
        random_state=seed,
    )
    # Second split: val vs test.
    val_relative = val_ratio / val_plus_test_ratio if val_plus_test_ratio > 0 else 0.5
    y_rest = y[idx_rest]
    idx_val, idx_test = train_test_split(
        np.arange(len(idx_rest)),
        test_size=1.0 - val_relative,
        stratify=y_rest,
        random_state=seed,
    )
    idx_val = idx_rest[idx_val]
    idx_test = idx_rest[idx_test]

    # Normalize: fit StandardScaler on training data only.
    scaler = StandardScaler()
    X_train_only = X[idx_train]
    scaler.fit(X_train_only)
    X_scaled = scaler.transform(X).astype(np.float32)

    # Build k-NN graph on full scaled features (acceptable for this dataset size).
    edge_index = build_knn_graph(X_scaled, k=k_neighbors)

    # Masks: boolean arrays of shape (num_nodes,).
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    data = Data(
        x=torch.from_numpy(X_scaled),
        edge_index=edge_index,
        y=torch.from_numpy(y).unsqueeze(1).float(),  # (N, 1) for BCEWithLogitsLoss
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
    )
    return data
