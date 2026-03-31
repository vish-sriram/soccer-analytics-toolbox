"""
VAEP Model Training

Trains two XGBoost binary classifiers:
  - P(scores)   : probability the acting team scores within the next 10 actions
  - P(concedes) : probability the acting team concedes within the next 10 actions

Then computes per-action VAEP values:
  VAEP(a) = [P_scores(a) - P_scores(a-1)] - [P_concedes(a) - P_concedes(a-1)]

Evaluation uses match-based cross-validation (leave-N-games-out) to prevent
data leakage — actions from the same match must not appear in both train and test.

Outputs
-------
  models/vaep_scores.json       XGBoost model for P(scores)
  models/vaep_concedes.json     XGBoost model for P(concedes)
  data/outputs/vaep_values.parquet   per-action VAEP value + components
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import joblib

FEATURES_DIR = Path("data/features")
OUTPUTS_DIR  = Path("data/outputs")
MODELS_DIR   = Path("models")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5   # match-grouped k-fold

# XGBoost hyperparameters (balanced for small dataset + heavy class imbalance)
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=20,
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    print("Loading features and labels...")
    features = pd.read_parquet(FEATURES_DIR / "features.parquet")
    labels   = pd.read_parquet(FEATURES_DIR / "labels.parquet")
    meta     = pd.read_parquet(FEATURES_DIR / "actions_meta.parquet")

    feature_cols = [c for c in features.columns if c != "match_id"]
    X = features[feature_cols].values.astype(np.float32)
    y_scores   = labels["scores"].astype(int).values
    y_concedes = labels["concedes"].astype(int).values
    groups     = features["match_id"].values

    print(f"  X shape      : {X.shape}")
    print(f"  Scores +ve   : {y_scores.sum():,}  ({y_scores.mean():.3%})")
    print(f"  Concedes +ve : {y_concedes.sum():,}  ({y_concedes.mean():.3%})")
    return X, y_scores, y_concedes, groups, feature_cols, meta


# ---------------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------------

def cross_validate(X, y, groups, label_name: str):
    """Match-grouped k-fold CV; returns out-of-fold predicted probabilities."""
    print(f"\n  Cross-validating {label_name} model ({N_FOLDS}-fold, grouped by match)...")
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale positive weight to handle class imbalance
        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

        brier = brier_score_loss(y_val, oof_probs[val_idx])
        auc   = roc_auc_score(y_val, oof_probs[val_idx])
        print(f"    fold {fold}: Brier={brier:.4f}  AUC={auc:.4f}")

    # Overall OOF metrics
    brier_oof = brier_score_loss(y, oof_probs)
    auc_oof   = roc_auc_score(y, oof_probs)
    ll_oof    = log_loss(y, oof_probs)
    print(f"  OOF overall → Brier={brier_oof:.4f}  AUC={auc_oof:.4f}  LogLoss={ll_oof:.4f}")
    return oof_probs


# ---------------------------------------------------------------------------
# Final model training (full dataset)
# ---------------------------------------------------------------------------

def train_final(X, y, label_name: str) -> XGBClassifier:
    """Train on the full dataset for use in production scoring."""
    print(f"\n  Training final {label_name} model on full dataset...")
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, verbose=False)
    print(f"  Done. {model.n_estimators} trees.")
    return model


# ---------------------------------------------------------------------------
# VAEP value computation
# ---------------------------------------------------------------------------

def compute_vaep(oof_scores: np.ndarray, oof_concedes: np.ndarray,
                 meta: pd.DataFrame) -> pd.DataFrame:
    """
    VAEP(a) = [P_scores(a) - P_scores(a-1)] - [P_concedes(a) - P_concedes(a-1)]

    The baseline probability is the probability of the preceding action.
    For the first action of each possession chain we use 0 as the baseline.
    """
    print("\nComputing VAEP values...")
    vaep = meta.copy()
    vaep["prob_scores"]   = oof_scores
    vaep["prob_concedes"] = oof_concedes

    # Shift probabilities within each match to get the previous-action baseline
    vaep = vaep.sort_values(["match_id", "action_id"]).reset_index(drop=True)
    vaep["prev_prob_scores"]   = vaep.groupby("match_id")["prob_scores"].shift(1).fillna(0)
    vaep["prev_prob_concedes"] = vaep.groupby("match_id")["prob_concedes"].shift(1).fillna(0)

    vaep["offensive_value"] = vaep["prob_scores"]   - vaep["prev_prob_scores"]
    vaep["defensive_value"] = vaep["prob_concedes"] - vaep["prev_prob_concedes"]
    vaep["vaep_value"]      = vaep["offensive_value"] - vaep["defensive_value"]

    print(f"  VAEP range: [{vaep['vaep_value'].min():.4f}, {vaep['vaep_value'].max():.4f}]")
    print(f"  Mean VAEP : {vaep['vaep_value'].mean():.6f}  (should be ~0)")
    return vaep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X, y_scores, y_concedes, groups, feature_cols, meta = load_data()

    # --- P(scores) ---
    print("\n=== Scores model ===")
    oof_scores = cross_validate(X, y_scores, groups, "scores")
    model_scores = train_final(X, y_scores, "scores")
    model_scores.save_model(str(MODELS_DIR / "vaep_scores.json"))
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    # --- P(concedes) ---
    print("\n=== Concedes model ===")
    oof_concedes = cross_validate(X, y_concedes, groups, "concedes")
    model_concedes = train_final(X, y_concedes, "concedes")
    model_concedes.save_model(str(MODELS_DIR / "vaep_concedes.json"))

    # --- VAEP values ---
    vaep_df = compute_vaep(oof_scores, oof_concedes, meta)
    vaep_df.to_parquet(OUTPUTS_DIR / "vaep_values.parquet", index=False)

    # --- Top feature importances ---
    print("\nTop 15 features by importance (scores model):")
    feat_imp = pd.Series(model_scores.feature_importances_, index=feature_cols)
    print(feat_imp.nlargest(15).to_string())

    print(f"\nDone. Outputs written to:")
    for d in [MODELS_DIR, OUTPUTS_DIR]:
        for f in sorted(d.iterdir()):
            print(f"  {f}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
