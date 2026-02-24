"""
MLP baseline for fraud classification (no graph).

Same input features and capacity as the GNN, but each node is classified
independently without message passing. Used for fair comparison to show
whether the graph structure adds value.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


class FraudMLP(nn.Module):
    """
    Multi-layer perceptron for binary fraud classification.

    Architecture: input_dim -> hidden_1 -> ReLU -> Dropout -> hidden_2 -> ReLU -> Dropout -> 1.
    Ignores edge_index; processes each node in isolation (non-graph baseline).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        dims = [input_dim] + list(hidden_dims)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], 1)

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass: apply MLP to node features only (no graph).

        Args:
            data: PyG Data; only .x is used.

        Returns:
            Logits of shape (num_nodes, 1).
        """
        x = data.x
        x = self.layers(x)
        logits = self.classifier(x)
        return logits
