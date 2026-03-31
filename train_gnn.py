"""
Train the spatial xG GNN and analyse high-value game states.

Steps:
  1. Load graphs from data/graphs.pkl (built by freeze_frames.py)
  2. Match-stratified 5-fold CV → AUC, Brier score, log loss
  3. Train final model on all data
  4. Save model to models/spatial_xg.pt
  5. Score every shot and dump ranked game states to data/outputs/spatial_xg_values.parquet

High-value game state analysis:
  - Top-N shots by P(goal) → inspect node positions
  - Cluster graph embeddings with k-means → characterise geometry types
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.cluster import KMeans

from gnn_model import SpatialXGModel

os.makedirs("models", exist_ok=True)
os.makedirs("data/outputs", exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIM = 32
HEADS = 4
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 80
BATCH_SIZE = 32
N_FOLDS = 5
N_CLUSTERS = 6  # k-means clusters over graph embeddings
DEVICE = torch.device("cpu")


def load_graphs(path: str = "data/graphs.pkl") -> tuple[list[Data], list[dict]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["graphs"], payload["metadata"]


def get_match_ids(metadata: list[dict]) -> np.ndarray:
    return np.array([m["match_id"] for m in metadata])


def train_epoch(model, loader, optimizer, pos_weight):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        probs = model.predict_proba(batch.x, batch.edge_index, batch.batch)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def cross_validate(graphs: list[Data], metadata: list[dict]) -> dict:
    match_ids = get_match_ids(metadata)
    labels = np.array([g.y.item() for g in graphs])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(graphs, labels, groups=match_ids)):
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)

        model = SpatialXGModel(hidden_dim=HIDDEN_DIM, heads=HEADS, dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, optimizer, pos_weight)

        probs, true = eval_model(model, val_loader)
        auc = roc_auc_score(true, probs) if len(np.unique(true)) > 1 else float("nan")
        brier = brier_score_loss(true, probs)
        ll = log_loss(true, probs)

        fold_metrics.append({"fold": fold + 1, "auc": auc, "brier": brier, "log_loss": ll})
        print(f"  Fold {fold+1}: AUC={auc:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}")

    return fold_metrics


def train_final(graphs: list[Data]) -> SpatialXGModel:
    labels = np.array([g.y.item() for g in graphs])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)

    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    model = SpatialXGModel(hidden_dim=HIDDEN_DIM, heads=HEADS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        loss = train_epoch(model, loader, optimizer, pos_weight)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")

    return model


@torch.no_grad()
def extract_embeddings(model: SpatialXGModel, graphs: list[Data]) -> np.ndarray:
    """Extract graph-level embeddings (before MLP head) for clustering."""
    model.eval()
    loader = DataLoader(graphs, batch_size=BATCH_SIZE)
    embeddings = []
    for batch in loader:
        batch = batch.to(DEVICE)
        from torch_geometric.nn import global_mean_pool, global_max_pool
        import torch.nn.functional as F

        x = batch.x
        x, _ = model.gat1(x, batch.edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, _ = model.gat2(x, batch.edge_index, return_attention_weights=True)
        x = F.elu(x)
        emb = torch.cat([global_mean_pool(x, batch.batch), global_max_pool(x, batch.batch)], dim=-1)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def analyse_game_states(model: SpatialXGModel, graphs: list[Data], metadata: list[dict]) -> pd.DataFrame:
    """Score all shots and cluster their embeddings."""
    loader = DataLoader(graphs, batch_size=BATCH_SIZE)
    all_probs, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            probs = model.predict_proba(batch.x, batch.edge_index, batch.batch)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    embeddings = extract_embeddings(model, graphs)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    meta_df = pd.DataFrame(metadata)
    df = pd.DataFrame({
        "p_goal": all_probs,
        "is_goal": [int(l) for l in all_labels],
        "cluster": clusters,
    })
    df = pd.concat([meta_df.reset_index(drop=True), df], axis=1)

    print("\n── Cluster summary (mean P(goal) per cluster) ──")
    summary = df.groupby("cluster").agg(
        n_shots=("shot_id", "count"),
        mean_p_goal=("p_goal", "mean"),
        goal_rate=("is_goal", "mean"),
    ).sort_values("mean_p_goal", ascending=False)
    print(summary.to_string())

    return df


def main():
    print("Loading graphs...")
    graphs, metadata = load_graphs()
    print(f"  {len(graphs)} shot graphs loaded")

    labels = np.array([g.y.item() for g in graphs])
    print(f"  Goals: {int(labels.sum())} / {len(labels)} ({labels.mean()*100:.1f}%)")

    print(f"\nRunning {N_FOLDS}-fold cross-validation...")
    fold_metrics = cross_validate(graphs, metadata)
    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n── CV Summary ──")
    print(metrics_df.describe().loc[["mean", "std"]].to_string())

    print("\nTraining final model on all data...")
    model = train_final(graphs)
    torch.save(model.state_dict(), "models/spatial_xg.pt")
    print("  Saved to models/spatial_xg.pt")

    print("\nAnalysing game states...")
    results = analyse_game_states(model, graphs, metadata)
    results.to_parquet("data/outputs/spatial_xg_values.parquet", index=False)
    print(f"  Saved to data/outputs/spatial_xg_values.parquet")

    print("\nDone.")


if __name__ == "__main__":
    main()
