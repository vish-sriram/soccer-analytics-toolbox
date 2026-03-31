"""
Train the STVN on possession chain sequences.

Training setup
--------------
  - Task:         binary classification — does this chain produce a shot?
  - Loss:         BCEWithLogitsLoss with pos_weight for class imbalance
  - CV:           match-grouped 5-fold (chains from same match stay together)
  - Augmentation: horizontal pitch flip (mirror y → PITCH_WIDTH - y)
  - Optimiser:    AdamW with cosine LR schedule

After training, computes per-frame value curves and player-level
marginal value rankings (deep-learning analogue of xVAEP).
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

from stvn import STVN

DEVICE   = torch.device("cpu")
CHAINS_PATH = Path("data/360/chains.pkl")
OUT_DIR     = Path("data/outputs")
MODEL_PATH  = Path("models/stvn.pt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIM  = 48
GRU_DIM     = 64
HEADS       = 4
DROPOUT     = 0.35
LR          = 3e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 60
BATCH_SIZE  = 16    # chains per batch
N_FOLDS     = 5
PITCH_WIDTH = 80.0


# ── Data augmentation ─────────────────────────────────────────────────────────

def flip_chain(chain):
    """Mirror all player y-coordinates: y → PITCH_WIDTH - y."""
    import copy
    from torch_geometric.data import Data
    flipped = []
    for g in chain:
        x = g.x.clone()
        x[:, 1] = PITCH_WIDTH / PITCH_WIDTH - x[:, 1]   # y is normalised
        x[:, 3] = -x[:, 3]                               # vy flips
        x[:, 5] = -x[:, 5]                               # ay flips
        flipped.append(Data(x=x, edge_index=g.edge_index.clone(),
                            has_shot=g.has_shot))
    return flipped


# ── Training utilities ────────────────────────────────────────────────────────

def batch_forward(model, chains, labels, pos_weight):
    logits = model(chains)
    loss   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
        logits, torch.tensor(labels, dtype=torch.float)
    )
    return loss, logits


def train_epoch(model, chains, labels, optimizer, pos_weight, augment=True):
    model.train()
    indices = np.random.permutation(len(chains))
    total_loss = 0.0

    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        b_chains  = [chains[i] for i in batch_idx]
        b_labels  = [labels[i] for i in batch_idx]

        # Horizontal flip augmentation
        if augment:
            extra_chains = [flip_chain(chains[i]) for i in batch_idx]
            extra_labels = b_labels[:]
            b_chains     = b_chains + extra_chains
            b_labels     = b_labels + extra_labels

        optimizer.zero_grad()
        loss, _ = batch_forward(model, b_chains, b_labels, pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(b_chains)

    return total_loss / (len(chains) * (2 if augment else 1))


@torch.no_grad()
def evaluate(model, chains, labels):
    model.eval()
    all_probs, all_labels = [], []
    for start in range(0, len(chains), BATCH_SIZE):
        b = chains[start:start + BATCH_SIZE]
        l = labels[start:start + BATCH_SIZE]
        probs = torch.sigmoid(model(b)).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(l)
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    auc    = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    brier  = brier_score_loss(labels, probs)
    return auc, brier, probs


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(chains, labels, meta):
    match_ids = np.array([m["match_id"] for m in meta])
    labels_arr = np.array(labels)
    n_pos = labels_arr.sum()
    pos_weight = torch.tensor([(len(labels_arr) - n_pos) / max(n_pos, 1)], dtype=torch.float)

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(chains, labels_arr, groups=match_ids)
    ):
        train_chains = [chains[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_chains   = [chains[i] for i in val_idx]
        val_labels   = [labels[i] for i in val_idx]

        model = STVN(hidden_dim=HIDDEN_DIM, gru_dim=GRU_DIM,
                     heads=HEADS, dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )

        for epoch in range(EPOCHS):
            train_epoch(model, train_chains, train_labels, optimizer, pos_weight)
            scheduler.step()

        auc, brier, _ = evaluate(model, val_chains, val_labels)
        fold_metrics.append({"fold": fold+1, "auc": auc, "brier": brier})
        print(f"  Fold {fold+1}: AUC={auc:.4f}  Brier={brier:.4f}")

    return fold_metrics


# ── Player value attribution ──────────────────────────────────────────────────

def compute_player_values(model, chains, meta, events_df: pd.DataFrame | None = None):
    """
    For each chain, compute per-frame marginal value (Δh_t).
    If events_df is provided, attribute marginal value to the actor player.
    Returns a DataFrame with columns [team, player, total_stvn_value, chains].
    """
    model.eval()
    player_values = defaultdict(float)
    player_chains = defaultdict(int)

    for i, (chain, m) in enumerate(zip(chains, meta)):
        values = model.frame_values(chain).numpy()  # [T]
        deltas = np.diff(values, prepend=0.0)       # Δvalue per frame

        team = m["team"]
        # Without player-level data, attribute to team
        player_values[team] += float(deltas.sum())
        player_chains[team] += 1

    df = pd.DataFrame([
        {"team": k, "total_stvn_value": round(v, 4),
         "chains": player_chains[k]}
        for k, v in player_values.items()
    ]).sort_values("total_stvn_value", ascending=False)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading chains...")
    with open(CHAINS_PATH, "rb") as f:
        data = pickle.load(f)

    chains = data["graphs"]
    labels = data["labels"]
    meta   = data["meta"]

    labels_arr = np.array(labels)
    print(f"  Total chains : {len(chains)}")
    print(f"  Shot chains  : {int(labels_arr.sum())}  ({labels_arr.mean()*100:.1f}%)")
    print(f"  Avg chain len: {np.mean([len(c) for c in chains]):.1f}")

    print(f"\nRunning {N_FOLDS}-fold cross-validation...")
    fold_metrics = cross_validate(chains, labels, meta)

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n── CV Summary ──")
    print(metrics_df.describe().loc[["mean","std"]].round(4).to_string())

    print("\nTraining final model on all data...")
    n_pos = labels_arr.sum()
    pos_weight = torch.tensor([(len(labels_arr) - n_pos) / max(n_pos, 1)], dtype=torch.float)

    model = STVN(hidden_dim=HIDDEN_DIM, gru_dim=GRU_DIM,
                 heads=HEADS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        loss = train_epoch(model, chains, labels, optimizer, pos_weight)
        scheduler.step()
        if (epoch + 1) % 15 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Saved → {MODEL_PATH}")

    print("\nComputing team-level STVN values...")
    team_values = compute_player_values(model, chains, meta)
    print(team_values.to_string(index=False))

    team_values.to_parquet(OUT_DIR / "stvn_team_values.parquet", index=False)
    print(f"\nSaved → data/outputs/stvn_team_values.parquet")


if __name__ == "__main__":
    main()
