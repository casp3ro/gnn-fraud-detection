"""
Configuration for the fraud detection pipeline.

Centralizes all hyperparameters, paths, and run settings so pipeline code
stays free of magic numbers and is easy to reproduce.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# Default project root: directory containing this config file.
_PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataConfig:
    """Paths and parameters for data loading and graph construction."""

    # Path to the CSV dataset (relative to project root or absolute).
    data_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "fraud_data.csv")

    # k for k-NN graph: each node is connected to its k nearest neighbors.
    # Higher k = more edges, more context; lower k = sparser graph. Typical: 5–15.
    k_neighbors: int = 10

    # Train / validation / test split ratios (must sum to 1.0).
    # Stratified so fraud ratio is preserved in each split.
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and training hyperparameters shared by GNN and MLP."""

    # Feature dimension: V1–V28 (28) + Amount (1) = 29.
    input_dim: int = 29

    # Hidden layer sizes. Same capacity for fair GNN vs MLP comparison.
    hidden_dims: Tuple[int, ...] = (64, 32)

    # Dropout probability for regularization.
    dropout: float = 0.3

    # Number of output classes (binary classification).
    num_classes: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop settings."""

    # Maximum number of epochs.
    max_epochs: int = 200

    # Learning rate for Adam optimizer.
    learning_rate: float = 0.01

    # Early stopping: stop if no improvement on val metric for this many epochs.
    patience: int = 25

    # Random seed for reproducibility (numpy, torch, sklearn).
    seed: int = 42

    # Device: "cuda" if available, else "cpu".
    @property
    def device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


@dataclass(frozen=True)
class VisualizationConfig:
    """Settings for Matplotlib-based visualizations and output paths."""

    # Where to save generated plots (relative to project root).
    plots_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "outputs" / "plots")

    # Where to save serialized training histories / metrics (for notebooks).
    metrics_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "outputs" / "metrics")

    # Toggle to enable/disable plotting from run.py.
    enable_plots: bool = True

    # Default figure size for plots (width, height) in inches.
    figsize: Tuple[float, float] = (8.0, 6.0)


@dataclass(frozen=True)
class Config:
    """Top-level config aggregating data, model, training, and visualization settings."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_defaults(cls) -> "Config":
        """Build config with all defaults (no CLI override for now)."""
        return cls()
