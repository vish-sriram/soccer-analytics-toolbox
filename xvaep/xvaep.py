"""
xVAEP — Expected VAEP

Standard VAEP evaluates actual outcomes, penalising difficult actions that
fail even when the decision itself had positive expected value.

xVAEP fixes this by separating decision quality from outcome luck:

    xVAEP(a) = P_success(a) * VAEP_if_success(a)
             + P_failure(a) * VAEP_if_failure(a)

Where:
  P_success(a)      = probability the action succeeds given context
                      (trained as a separate "action quality" model)
  VAEP_if_success   = counterfactual VAEP if result had been success
  VAEP_if_failure   = counterfactual VAEP if result had been failure

Computing counterfactuals: flip the result_*_a0 columns in the feature
matrix and re-run the saved VAEP XGBoost models. The delta before/after
is then computed with the counterfactual outcome baked in.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEATURES_DIR = Path("data/features")
OUTPUTS_DIR  = Path("data/outputs")
MODELS_DIR   = Path("models")

# Action types where result is definitionally fixed — exclude from
# the success model since they carry no decision uncertainty
FIXED_RESULT_TYPES = {"goalkick", "keeper_save", "keeper_claim",
                      "keeper_punch", "clearance", "bad_touch"}

# Columns that encode the result of a0 (the current action)
RESULT_COLS_A0 = [
    "result_fail_a0", "result_success_a0", "result_offside_a0",
    "result_owngoal_a0", "result_yellow_card_a0", "result_red_card_a0",
]

XGB_SUCCESS_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)


# ── Step 1: Train P(success | action, context) ────────────────────────────────

def build_success_features(spadl: pd.DataFrame) -> pd.DataFrame:
    """
    Construct feature matrix for the action-success model.
    Features: action type (one-hot), bodypart (one-hot),
              start/end x/y, distance & angle to goal, movement.
    """
    s = spadl.reset_index(drop=True)

    type_dummies = pd.get_dummies(s["type_name"], prefix="type")
    body_dummies = pd.get_dummies(s["bodypart_name"], prefix="body")

    feats = pd.concat([
        type_dummies,
        body_dummies,
        s[["start_x", "start_y", "end_x", "end_y", "period_id", "time_seconds"]],
    ], axis=1)

    dx = 105 - s["start_x"].values
    dy = 34  - s["start_y"].values
    feats["dist_to_goal"]  = np.sqrt(dx**2 + dy**2)
    feats["angle_to_goal"] = np.arctan2(np.abs(dy), np.abs(dx))
    feats["movement"]      = np.sqrt(
        (s["end_x"] - s["start_x"])**2 +
        (s["end_y"] - s["start_y"])**2
    )
    return feats.fillna(0)


def train_success_model(spadl: pd.DataFrame) -> tuple[XGBClassifier, list[str]]:
    mask = ~spadl["type_name"].isin(FIXED_RESULT_TYPES)
    sub  = spadl[mask].copy()

    y    = (sub["result_name"] == "success").astype(int).values
    X    = build_success_features(sub)
    feat_cols = X.columns.tolist()
    X    = X.values

    groups  = sub["match_id"].values
    gkf     = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        m = XGBClassifier(**XGB_SUCCESS_PARAMS)
        m.fit(X[train_idx], y[train_idx],
              eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        oof_preds[val_idx] = m.predict_proba(X[val_idx])[:, 1]

    auc = roc_auc_score(y, oof_preds)
    print(f"  Action-success model  CV AUC = {auc:.4f}")

    # Final model on all data
    final = XGBClassifier(**XGB_SUCCESS_PARAMS)
    final.fit(X, y, verbose=False)

    joblib.dump((final, feat_cols), MODELS_DIR / "action_success.pkl")
    return final, feat_cols


# ── Step 2: Counterfactual VAEP via result-column flipping ───────────────────

def load_vaep_models():
    from xgboost import XGBClassifier
    scores_model   = XGBClassifier()
    concedes_model = XGBClassifier()
    scores_model.load_model(MODELS_DIR / "vaep_scores.json")
    concedes_model.load_model(MODELS_DIR / "vaep_concedes.json")
    feat_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    return scores_model, concedes_model, feat_cols


def predict_vaep_components(X_df: pd.DataFrame,
                             scores_model, concedes_model,
                             feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = X_df[feat_cols].values
    p_scores   = scores_model.predict_proba(X)[:, 1]
    p_concedes = concedes_model.predict_proba(X)[:, 1]
    return p_scores, p_concedes


def compute_counterfactual_vaep(X_df: pd.DataFrame,
                                 scores_model, concedes_model,
                                 feat_cols: list[str]):
    """
    For each action compute VAEP under success and failure hypotheticals.

    Returns arrays:
      vaep_if_success   shape (N,)
      vaep_if_failure   shape (N,)
      p_scores_before   shape (N,)  — used as baseline (same as standard VAEP)
      p_concedes_before shape (N,)
    """
    # Baseline: actual P scores/concedes for each action
    p_scores_actual, p_concedes_actual = predict_vaep_components(
        X_df, scores_model, concedes_model, feat_cols
    )

    # The "before" state for action i is the "after" state for action i-1.
    # In the feature matrix, action i's features include its own result (a0).
    # P_before[i] = P_scores_actual[i-1]  (shifted by 1).
    p_scores_before   = np.roll(p_scores_actual,   1)
    p_concedes_before = np.roll(p_concedes_actual, 1)
    # First action in each game has no predecessor — treat baseline as 0
    p_scores_before[0]   = 0.0
    p_concedes_before[0] = 0.0

    # --- Counterfactual: force success ---
    X_success = X_df.copy()
    for col in RESULT_COLS_A0:
        X_success[col] = 1 if col == "result_success_a0" else 0
    p_s_succ, p_c_succ = predict_vaep_components(
        X_success, scores_model, concedes_model, feat_cols
    )

    # --- Counterfactual: force failure ---
    X_failure = X_df.copy()
    for col in RESULT_COLS_A0:
        X_failure[col] = 1 if col == "result_fail_a0" else 0
    p_s_fail, p_c_fail = predict_vaep_components(
        X_failure, scores_model, concedes_model, feat_cols
    )

    vaep_if_success = (p_s_succ - p_scores_before) - (p_c_succ - p_concedes_before)
    vaep_if_failure = (p_s_fail - p_scores_before) - (p_c_fail - p_concedes_before)

    return vaep_if_success, vaep_if_failure, p_scores_before, p_concedes_before


# ── Step 3: Combine into xVAEP ────────────────────────────────────────────────

def compute_xvaep(spadl: pd.DataFrame,
                  X_vaep: pd.DataFrame,
                  success_model: XGBClassifier,
                  success_feat_cols: list[str],
                  scores_model, concedes_model,
                  vaep_feat_cols: list[str]) -> pd.DataFrame:

    # P(success | action, context)  — 0.5 default for fixed-result actions
    success_feats = build_success_features(spadl)
    fixed_mask    = spadl["type_name"].isin(FIXED_RESULT_TYPES).values
    p_success     = np.full(len(spadl), 0.5)

    # Only predict for non-fixed actions
    non_fixed_X = success_feats[~fixed_mask][success_feat_cols].reindex(
        columns=success_feat_cols, fill_value=0
    ).values
    p_success[~fixed_mask] = success_model.predict_proba(non_fixed_X)[:, 1]

    # Counterfactual VAEP values
    vaep_if_s, vaep_if_f, _, _ = compute_counterfactual_vaep(
        X_vaep, scores_model, concedes_model, vaep_feat_cols
    )

    # Standard VAEP (actual outcome)
    actual_vaep = pd.read_parquet(OUTPUTS_DIR / "vaep_values.parquet")["vaep_value"].values

    # xVAEP
    xvaep = p_success * vaep_if_s + (1 - p_success) * vaep_if_f

    return pd.DataFrame({
        "vaep_value":     actual_vaep,
        "xvaep":          xvaep,
        "vaep_if_success": vaep_if_s,
        "vaep_if_failure": vaep_if_f,
        "p_success":      p_success,
    })


# ── Step 4: Player rankings ───────────────────────────────────────────────────

def player_rankings(spadl: pd.DataFrame,
                    results: pd.DataFrame,
                    lineups: pd.DataFrame) -> pd.DataFrame:
    player_lookup = (
        lineups[["player_id", "player_name", "team"]]
        .drop_duplicates("player_id")
        .set_index("player_id")
    )
    df = spadl.copy()
    df["player_name"] = df["player_id"].map(player_lookup["player_name"])
    df["team"]        = df["player_id"].map(player_lookup["team"])
    df["vaep_value"]  = results["vaep_value"].values
    df["xvaep"]       = results["xvaep"].values

    lev = df[df["team"] == "Bayer Leverkusen"]

    rank = (
        lev.groupby("player_name")
        .agg(
            actions          = ("vaep_value", "count"),
            total_vaep       = ("vaep_value", "sum"),
            total_xvaep      = ("xvaep",      "sum"),
            xvaep_per_action = ("xvaep",      "mean"),
        )
        .reset_index()
    )
    rank["vaep_rank"]  = rank["total_vaep"].rank(ascending=False).astype(int)
    rank["xvaep_rank"] = rank["total_xvaep"].rank(ascending=False).astype(int)
    rank["rank_delta"] = rank["vaep_rank"] - rank["xvaep_rank"]
    return rank.sort_values("total_xvaep", ascending=False).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    spadl   = pd.read_parquet(FEATURES_DIR / "spadl_actions.parquet")
    X_vaep  = pd.read_parquet(FEATURES_DIR / "features.parquet")
    lineups = pd.read_parquet("data/raw/lineups.parquet")

    print("\nTraining action-success model...")
    success_model, success_feat_cols = train_success_model(spadl)

    print("Loading VAEP models...")
    scores_model, concedes_model, vaep_feat_cols = load_vaep_models()

    print("Computing counterfactual VAEP and xVAEP...")
    results = compute_xvaep(
        spadl, X_vaep,
        success_model, success_feat_cols,
        scores_model, concedes_model, vaep_feat_cols,
    )

    results.to_parquet(OUTPUTS_DIR / "xvaep_values.parquet", index=False)
    print(f"  Saved → data/outputs/xvaep_values.parquet")

    print("\nComputing player rankings...")
    rankings = player_rankings(spadl, results, lineups)

    pd.set_option("display.width", 130)
    print("\n=== Leverkusen: VAEP vs xVAEP Rankings ===\n")
    cols = ["player_name", "actions", "total_vaep", "vaep_rank",
            "total_xvaep", "xvaep_rank", "rank_delta"]
    print(rankings[cols].round({"total_vaep": 2, "total_xvaep": 3}).to_string(index=False))

    wirtz = rankings[rankings["player_name"].str.contains("Wirtz", na=False)]
    print(f"\nWirtz:  VAEP rank {wirtz['vaep_rank'].values[0]}/23  →  "
          f"xVAEP rank {wirtz['xvaep_rank'].values[0]}/23  "
          f"(Δ = {wirtz['rank_delta'].values[0]:+d})")


if __name__ == "__main__":
    main()
