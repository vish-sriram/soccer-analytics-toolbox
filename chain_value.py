"""
Possession Chain Value Attribution (PCVA)

VAEP's core flaw for creative players: it only measures the immediate
probability change of each action, penalising failed attempts even when
those attempts are part of sequences that generate high-value chances.

PCVA fix:
  1. Group actions into possession chains (broken by turnover or period change)
  2. For each chain, compute terminal value = xG of the shot it produced
     (or 0 if no shot)
  3. Distribute that value back to every player in the chain using an
     exponential decay weight (actions closer to the shot get more credit,
     but ALL contributors get some)
  4. Re-rank players on cumulative chain value

This directly captures what VAEP misses: Wirtz dribbles past a defender
and lays off a pass that leads to a shot three actions later — he
contributed to that shot and should receive credit.
"""

import pandas as pd
import numpy as np

DECAY   = 0.85   # credit fades by 15% per step back from the shot
MIN_GAP = 5.0    # seconds gap between actions to break a chain


def build_chains(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a chain_id to every action.
    A new chain begins when:
      - the team changes (turnover / defensive action)
      - the period changes
      - time gap between consecutive actions exceeds MIN_GAP seconds
    """
    df = df.sort_values(["match_id", "period_id", "time_seconds"]).reset_index(drop=True)

    chain_id = 0
    chain_ids = [0]

    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]
        new_chain = (
            curr["match_id"]  != prev["match_id"]  or
            curr["period_id"] != prev["period_id"] or
            curr["team"]      != prev["team"]       or
            (curr["time_seconds"] - prev["time_seconds"]) > MIN_GAP
        )
        if new_chain:
            chain_id += 1
        chain_ids.append(chain_id)

    df["chain_id"] = chain_ids
    return df


def chain_terminal_value(chain: pd.DataFrame) -> float:
    """
    Terminal value of a possession chain = xG of the last shot in it
    (using VAEP's scores probability as the xG proxy), or 0 if no shot.
    """
    shots = chain[chain["type_name"].str.contains("shot", case=False, na=False)]
    if shots.empty:
        return 0.0
    # Use the scores label as a proxy: mean P(scores) around the shot
    return float(shots["scores"].iloc[-1])


def attribute_chain_value(chain: pd.DataFrame, terminal: float) -> pd.Series:
    """
    Distribute terminal value to each action in the chain using
    exponential decay from the end: last action gets weight 1,
    second-to-last DECAY, third-to-last DECAY^2, etc.
    Weights are normalised so they sum to 1.
    """
    n = len(chain)
    weights = np.array([DECAY ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()
    return pd.Series(weights * terminal, index=chain.index)


def compute_pcva(df: pd.DataFrame) -> pd.DataFrame:
    """Build chains and compute per-action PCVA credit."""
    df = build_chains(df)

    df["pcva"] = 0.0
    for _, chain in df.groupby("chain_id"):
        tv = chain_terminal_value(chain)
        if tv > 0:
            df.loc[chain.index, "pcva"] = attribute_chain_value(chain, tv)

    return df


def player_rankings(df: pd.DataFrame, team: str) -> pd.DataFrame:
    lev = df[df["team"] == team].copy()

    vaep_rank = (
        lev.groupby("player_name")
        .agg(total_vaep=("vaep_value", "sum"), actions=("vaep_value", "count"))
        .reset_index()
    )

    pcva_rank = (
        lev.groupby("player_name")
        .agg(total_pcva=("pcva", "sum"))
        .reset_index()
    )

    # Shot chain involvement: what % of shot-producing chains did this player appear in?
    shot_chains = set(
        lev[lev["type_name"].str.contains("shot", case=False, na=False)]["chain_id"]
    )
    total_shot_chains = len(shot_chains)

    involvement = (
        lev[lev["chain_id"].isin(shot_chains)]
        .groupby("player_name")["chain_id"]
        .nunique()
        .reset_index()
        .rename(columns={"chain_id": "shot_chains_involved"})
    )
    involvement["pct_shot_chains"] = (
        involvement["shot_chains_involved"] / total_shot_chains * 100
    ).round(1)

    rankings = (
        vaep_rank
        .merge(pcva_rank, on="player_name")
        .merge(involvement, on="player_name", how="left")
        .fillna(0)
    )
    rankings["vaep_rank"] = rankings["total_vaep"].rank(ascending=False).astype(int)
    rankings["pcva_rank"] = rankings["total_pcva"].rank(ascending=False).astype(int)
    rankings["rank_delta"] = rankings["vaep_rank"] - rankings["pcva_rank"]  # positive = PCVA rewards more
    return rankings.sort_values("total_pcva", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    vaep    = pd.read_parquet("data/outputs/vaep_values.parquet")
    spadl   = pd.read_parquet("data/features/spadl_actions.parquet")
    lineups = pd.read_parquet("data/raw/lineups.parquet")
    labels  = pd.read_parquet("data/features/labels.parquet")

    player_lookup = (
        lineups[["player_id", "player_name", "team"]]
        .drop_duplicates("player_id")
        .set_index("player_id")
    )

    df = spadl.copy()
    df["vaep_value"]  = vaep["vaep_value"].values
    df["scores"]      = labels["scores"].values
    df["player_name"] = df["player_id"].map(player_lookup["player_name"])
    df["team"]        = df["player_id"].map(player_lookup["team"])

    print("Building possession chains and attributing value...")
    df = compute_pcva(df)

    rankings = player_rankings(df, "Bayer Leverkusen")

    pd.set_option("display.width", 120)
    print("\n=== Leverkusen: VAEP vs PCVA Rankings ===\n")
    cols = ["player_name", "actions", "total_vaep", "vaep_rank",
            "total_pcva", "pcva_rank", "rank_delta", "pct_shot_chains"]
    print(rankings[cols].round({"total_vaep": 2, "total_pcva": 3, "pct_shot_chains": 1}).to_string(index=False))

    wirtz = rankings[rankings["player_name"].str.contains("Wirtz", na=False)]
    print(f"\nWirtz VAEP rank: {wirtz['vaep_rank'].values[0]}/23  →  "
          f"PCVA rank: {wirtz['pcva_rank'].values[0]}/23  "
          f"(appears in {wirtz['pct_shot_chains'].values[0]:.1f}% of shot-producing chains)")

    df.to_parquet("data/outputs/pcva_values.parquet", index=False)
    print("\nSaved → data/outputs/pcva_values.parquet")
