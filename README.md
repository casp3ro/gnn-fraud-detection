# Graph-Based Credit Card Fraud Detection

Binary classification of credit card transactions (legitimate vs fraud) using **graph neural networks** (GNN) with PyTorch Geometric, with a non-graph **MLP baseline** for comparison.

## Problem

- **Task**: Imbalanced binary classification — only a small fraction of transactions are fraud (~1.7% in the dataset).
- **Goal**: Use graph-based reasoning (similar transactions connected) and compare against a flat MLP to see if the graph structure helps.

## Approach

1. **Graph formulation** (from tabular data):
   - **Nodes**: One per transaction (row in the CSV).
   - **Node features**: V1–V28 (PCA-derived) + normalized Amount → 29-dim vector.
   - **Edges**: k-NN graph — each node is connected to its k nearest neighbors in feature space; edges are symmetrized for an undirected graph.

2. **Models**:
   - **GNN**: Two-layer GCN; message passing over the k-NN graph so each node’s representation depends on its neighbors.
   - **MLP baseline**: Same architecture (29 → 64 → 32 → 1) but no graph; each node classified from its own features only.

3. **Training**: Same train/val/test split and class weighting (`pos_weight` in BCEWithLogitsLoss); early stopping on validation F1.

## Dataset

- **File**: `fraud_data.csv`
- **Columns**: V1–V28 (anonymized PCA features), Amount, Class (0 = legitimate, 1 = fraud).
- **Size**: ~21.7k rows; highly imbalanced (~1.7% fraud).

## How to run

From the project root:

```bash
pip3 install -r requirements.txt
python3 run.py
```

`run.py` will:

1. Load and normalize the data (scaler fit on train only).
2. Build the k-NN graph and create train/val/test masks.
3. Train the GNN and the MLP with early stopping.
4. Print a comparison table (accuracy, precision, recall, F1, ROC-AUC) and confusion matrices on the test set.

## Results

**GNN vs MLP** (test set, stratified 70/15/15 split):

| Metric    | GNN    | MLP    |
| --------- | ------ | ------ |
| Accuracy  | 0.9862 | 0.9831 |
| Precision | 0.5465 | 0.4896 |
| Recall    | 0.8868 | 0.8868 |
| F1        | 0.6763 | 0.6309 |
| ROC-AUC   | 0.9725 | 0.9667 |

**Confusion matrices** (rows = true, cols = predicted; order = [legit, fraud]):

- **GNN**: TN=3162, FP=39, FN=6, TP=47
- **MLP**: TN=3152, FP=49, FN=6, TP=47

**Interpretation**: The GNN outperforms the MLP on precision, F1, and ROC-AUC while matching recall; it also has fewer false positives (39 vs 49). Using neighborhood information via the k-NN graph helps the model better separate fraud from legitimate transactions without missing more frauds.

## Project structure

```
projects/fraud-detection/
├── fraud_data.csv
├── requirements.txt
├── README.md
├── README-PL.md        # Polish description
├── config.py           # Dataclass config (paths, k, split, seed, device, visualization)
├── data/
│   ├── dataset.py      # Load CSV, normalize, split, build PyG Data
│   └── graph_builder.py # k-NN graph from feature matrix
├── models/
│   ├── gnn.py          # Two-layer GCN (FraudGNN)
│   └── baseline_mlp.py # MLP baseline (FraudMLP)
├── training/
│   ├── metrics.py      # Precision, recall, F1, AUC-ROC, confusion matrix
│   └── trainer.py      # Train/val loop, early stopping, BCEWithLogitsLoss + history
├── visualization/
│   └── plots.py        # Matplotlib/seaborn plots (curves, confusion, ROC/PR, graph, etc.)
└── run.py              # Entry point: load → train both → compare → (optionally) plot
```

## Dependencies

- Python 3.9+
- PyTorch, torch_geometric, scikit-learn, pandas, numpy, matplotlib (see `requirements.txt`).
