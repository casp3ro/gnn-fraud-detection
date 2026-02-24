"""
k-NN graph construction from a feature matrix.

Builds an undirected graph by connecting each node to its k nearest neighbors
in feature space. Used to turn tabular fraud data into a graph for GNNs.
"""

from typing import Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    X: np.ndarray,
    k: int,
    metric: str = "minkowski",
    include_self: bool = False,
) -> torch.Tensor:
    """
    Build an undirected k-NN graph from a feature matrix.

    Each node i is connected to its k nearest neighbors (by Euclidean distance
    when metric='minkowski' with p=2). Edges are symmetrized: if i→j exists then
    j→i is added, so the GNN sees an undirected graph.

    Args:
        X: Node feature matrix of shape (num_nodes, num_features).
        k: Number of nearest neighbors per node.
        metric: Distance metric for NearestNeighbors ('minkowski' => Euclidean).
        include_self: If True, each node may link to itself (k+1 includes self).
            For standard k-NN we use False.

    Returns:
        edge_index: LongTensor of shape (2, num_edges) in COO format, so that
            edge_index[0] are source nodes and edge_index[1] are target nodes.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_nodes = X.shape[0]
    if k >= n_nodes:
        raise ValueError(f"k ({k}) must be < num_nodes ({n_nodes})")

    # k+1 because the first neighbor is typically self (distance 0); we exclude self
    # so each node gets exactly k neighbors.
    n_neighbors = min(k + 1, n_nodes)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto")
    nn.fit(X)

    # Indices of k+1 nearest neighbors for each node (including self at index 0 usually).
    # distances shape (n_nodes, n_neighbors), indices shape (n_nodes, n_neighbors).
    distances, indices = nn.kneighbors(X)

    # Build edge list: for each node i, edges (i, j) for each j in indices[i].
    sources: list[int] = []
    targets: list[int] = []
    for i in range(n_nodes):
        for j in indices[i]:
            if not include_self and j == i:
                continue
            sources.append(i)
            targets.append(j)

    edge_index = np.array([sources, targets], dtype=np.int64)

    # Symmetrize: add reverse edges so graph is undirected.
    # GCN and most PyG layers expect (source, target) and may use both directions.
    edge_index = _symmetrize_edges(edge_index)

    return torch.from_numpy(edge_index).long()


def _symmetrize_edges(edge_index: np.ndarray) -> np.ndarray:
    """
    Make the graph undirected by adding reverse edges.

    If (i, j) exists, ensure (j, i) exists. Removes duplicates so each
    undirected edge appears once as (i, j) with i < j, or we keep both
    (i,j) and (j,i) for PyG (both conventions are used; keeping both
    doubles the edge count but is standard in message passing).
    """
    src, tgt = edge_index[0], edge_index[1]
    # Append reverse edges.
    src_rev = np.concatenate([src, tgt])
    tgt_rev = np.concatenate([tgt, src])
    combined = np.stack([src_rev, tgt_rev], axis=0)
    # Remove duplicates (optional but reduces memory).
    combined = np.unique(combined, axis=1)
    return combined
