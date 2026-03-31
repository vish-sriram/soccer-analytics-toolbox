"""
VAEP Feature Engineering for Bayer Leverkusen 2023/24 Bundesliga season.

Pipeline:
  1. Load events via socceraction's StatsBombLoader (handles nested format)
  2. Convert raw events → SPADL actions (standardized action format)
  3. Engineer features over a rolling window of the last N_ACTIONS actions
  4. Generate scoring/conceding labels for each action
  5. Save to data/features/

SPADL (Soccer Player Action Description Language) standardizes events into a
uniform schema: type, start (x,y), end (x,y), result, body part, period, time.

VAEP labels:
  scores:   did the acting team score within the next 10 actions?
  concedes: did the acting team concede within the next 10 actions?
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd

import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.vaep import features as vaep_features
from socceraction.vaep import labels as vaep_labels

COMPETITION_ID = 9    # 1. Bundesliga
SEASON_ID = 281       # 2023/2024
RAW_DIR = Path("data/raw")
FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# VAEP window: features built from the last N_ACTIONS actions
N_ACTIONS = 3


# ---------------------------------------------------------------------------
# Step 1: Load events and convert to SPADL actions
# ---------------------------------------------------------------------------

def load_and_convert() -> pd.DataFrame:
    """Use socceraction's StatsBombLoader to fetch events and convert to SPADL."""
    print("Loading events via StatsBombLoader...")
    loader = StatsBombLoader(getter="remote", creds={"user": "", "passwd": ""})

    games = loader.games(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    print(f"  {len(games)} games found.")

    all_actions = []
    for i, (_, game) in enumerate(games.iterrows(), 1):
        mid = game["game_id"]
        home_team_id = game["home_team_id"]

        events = loader.events(game_id=mid)
        actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
        actions["match_id"] = mid
        all_actions.append(actions)
        print(f"  [{i}/{len(games)}] game {mid}: {len(actions)} actions")

    actions_df = pd.concat(all_actions, ignore_index=True)
    print(f"\n  Total SPADL actions: {len(actions_df):,}")
    return actions_df, games


# ---------------------------------------------------------------------------
# Steps 2 & 3: Features and labels per match
# ---------------------------------------------------------------------------

def build_features_and_labels(actions_df: pd.DataFrame):
    """
    Build VAEP feature matrix and labels.

    Features (per action, using window of last N_ACTIONS):
      For each action slot in the window:
        - actiontype_onehot  (20 categories)
        - result_onehot      (6 categories)
        - bodypart_onehot    (4 categories)
        - start_x, start_y, end_x, end_y
        - start/end distance and angle to goal
        - movement (dx, dy)
        - time_delta (seconds since prior action)
        - team (same team as current action?)

    Labels:
      scores   (bool): acting team scores within next 10 actions
      concedes (bool): acting team concedes within next 10 actions
    """
    print("\nBuilding features and labels per match...")

    feature_frames = []
    label_frames = []
    action_frames = []

    grouped = actions_df.groupby("match_id")
    n_matches = len(grouped)

    for i, (mid, game_actions) in enumerate(grouped, 1):
        game_actions = game_actions.reset_index(drop=True)

        # gamestates returns list of N_ACTIONS DataFrames (shifted views)
        gamestates = vaep_features.gamestates(game_actions, nb_prev_actions=N_ACTIONS)

        feat_df = pd.concat([
            vaep_features.actiontype_onehot(gamestates),
            vaep_features.result_onehot(gamestates),
            vaep_features.bodypart_onehot(gamestates),
            vaep_features.startlocation(gamestates),
            vaep_features.endlocation(gamestates),
            vaep_features.startpolar(gamestates),
            vaep_features.endpolar(gamestates),
            vaep_features.movement(gamestates),
            vaep_features.time_delta(gamestates),
            vaep_features.team(gamestates),
        ], axis=1)
        feat_df["match_id"] = mid

        lbl_df = vaep_labels.scores(game_actions)
        lbl_df["concedes"] = vaep_labels.concedes(game_actions)["concedes"]
        lbl_df["match_id"] = mid

        action_meta = game_actions[[
            "action_id", "match_id", "period_id", "time_seconds",
            "team_id", "player_id", "type_id", "result_id",
        ]].copy()

        feature_frames.append(feat_df)
        label_frames.append(lbl_df)
        action_frames.append(action_meta)

        print(f"  [{i}/{n_matches}] match {mid}: {len(game_actions)} actions, {feat_df.shape[1]-1} features")

    return (
        pd.concat(feature_frames, ignore_index=True),
        pd.concat(label_frames, ignore_index=True),
        pd.concat(action_frames, ignore_index=True),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1
    actions_df, games = load_and_convert()

    # Add human-readable names for inspection
    actions_named = spadl.add_names(actions_df)
    actions_named.to_parquet(FEATURES_DIR / "spadl_actions.parquet", index=False)
    print(f"  SPADL actions saved.")

    print("\nAction type distribution:")
    print(actions_named["type_name"].value_counts().to_string())

    # Steps 2 & 3
    features, labels, actions_meta = build_features_and_labels(actions_named)

    assert len(features) == len(labels) == len(actions_meta), "Row count mismatch!"

    print(f"\nLabel rates  →  scores: {labels['scores'].mean():.3%}  |  concedes: {labels['concedes'].mean():.3%}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature columns:\n  {features.drop(columns='match_id').columns.tolist()}")

    features.to_parquet(FEATURES_DIR / "features.parquet", index=False)
    labels.to_parquet(FEATURES_DIR / "labels.parquet", index=False)
    actions_meta.to_parquet(FEATURES_DIR / "actions_meta.parquet", index=False)

    print("\nDone. Files written to data/features/:")
    for f in sorted(FEATURES_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
