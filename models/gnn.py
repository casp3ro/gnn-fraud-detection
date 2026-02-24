"""
Graph Neural Network for node-level fraud classification.

Two-layer GCN: node features are propagated along the k-NN graph edges,
so each node's representation is informed by its neighbors (graph reasoning).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class FraudGNN(nn.Module):
    """
    Two-layer Graph Convolutional Network for binary fraud classification.

    Architecture: input_dim -> hidden_1 -> ReLU -> Dropout -> hidden_2 -> ReLU -> Dropout -> 1.
    Message passing uses the graph structure (edge_index) so that each node
    aggregates information from its k-NN neighbors.
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

        # First GCN layer: input -> first hidden dim.
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        # Second GCN layer: first hidden -> second hidden (or output if only one hidden).
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0])
        # Final projection to single logit per node.
        out_dim = hidden_dims[-1] if hidden_dims else hidden_dims[0]
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass: propagate node features over the graph, then classify.

        Args:
            data: PyG Data with .x (node features) and .edge_index.

        Returns:
            Logits of shape (num_nodes, 1).
        """
        x = data.x
        edge_index = data.edge_index

        # Layer 1: graph convolution + ReLU + dropout.
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: graph convolution + ReLU + dropout.
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Node-level classification: one logit per node.
        logits = self.classifier(x)
        return logits
