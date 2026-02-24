"""
Models: GNN and MLP for fraud detection.
"""

from models.gnn import FraudGNN
from models.baseline_mlp import FraudMLP

__all__ = ["FraudGNN", "FraudMLP"]
