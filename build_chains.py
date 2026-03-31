"""
Build possession chain sequences from 360 event data.

A possession chain is a maximal sequence of consecutive actions by the
same team within a period. Each chain becomes one training example:

  Input : sequence of spatial graphs [G_1, ..., G_T]
            - nodes  = visible players in freeze frame
            - edges  = fully connected within team, proximity across teams
            - node features: [x, y, vx, vy, ax, ay,
                               is_actor, is_teammate, is_gk,
                               dist_to_goal, angle_to_goal]
  Label : 1 if the chain contains a shot (attacking team), 0 otherwise
          (shot-ended chains are the positive class — the model learns
           which spatial evolutions lead to shot-generating sequences)

Velocity and acceleration are derived from position deltas between
consecutive frames *within the chain*. First frame vx=vy=0, second
frame ax=ay=0.

Output: data/360/chains.pkl
  {
    "graphs":   List[List[torch_geometric.data.Data]],  # one list per chain
    "labels":   List[int],
    "meta":     List[dict],   # match_id, team, chain_len, has_goal
  }
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch_geometric.data import Data

RAW_DIR   = Path("data/360/raw")
OUT_DIR   = Path("data/360")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PITCH_LENGTH = 120.0
PITCH_WIDTH  = 80.0
GOAL_CENTER  = np.array([120.0, 40.0])

# Chain is broken by a time gap larger than this (seconds)
MAX_TIME_GAP = 8.0
# Cross-team edge if players within this distance (pitch units)
CROSS_TEAM_EDGE_DIST = 12.0
# Chains shorter than this are discarded
MIN_CHAIN_LEN = 2
# Chains longer than this are truncated (avoids outlier memory use)
MAX_CHAIN_LEN = 30


def timestamp_to_seconds(ts: str) -> float:
    """'00:03:12.450' → seconds float."""
    try:
        parts = ts.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0


def split_into_chains(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split a match-period dataframe into possession chains.
    Breaks on: team change OR time gap > MAX_TIME_GAP seconds.
    """
    df = df.sort_values("index").reset_index(drop=True)
    df["time_s"] = df["timestamp"].apply(timestamp_to_seconds)

    chains, current = [], [0]
    for i in range(1, len(df)):
        team_change = df.loc[i, "team"] != df.loc[i-1, "team"]
        time_gap    = df.loc[i, "time_s"] - df.loc[i-1, "time_s"]
        if team_change or time_gap > MAX_TIME_GAP:
            chains.append(df.loc[current].copy())
            current = []
        current.append(i)
    if current:
        chains.append(df.loc[current].copy())

    return [c for c in chains if len(c) >= MIN_CHAIN_LEN]


def build_node_features(players: list[dict],
                        prev_players: list[dict] | None,
                        pprev_players: list[dict] | None) -> torch.Tensor:
    """
    Build node feature matrix for one freeze frame.

    players: list of dicts with keys teammate, actor, keeper, location
    prev/pprev: same structure for t-1 and t-2 frames (for vel/accel)

    Returns tensor [N, 11]
    """
    rows = []
    for j, p in enumerate(players):
        loc  = np.array(p["location"], dtype=np.float32)
        x    = loc[0] / PITCH_LENGTH
        y    = loc[1] / PITCH_WIDTH

        # Velocity: Δpos from previous frame (0 if first frame)
        if prev_players and j < len(prev_players):
            prev_loc = np.array(prev_players[j]["location"], dtype=np.float32)
            vx = (loc[0] - prev_loc[0]) / PITCH_LENGTH
            vy = (loc[1] - prev_loc[1]) / PITCH_WIDTH
        else:
            vx = vy = 0.0

        # Acceleration: Δvel from frame before that (0 if ≤ second frame)
        if pprev_players and prev_players and j < len(prev_players) and j < len(pprev_players):
            pprev_loc = np.array(pprev_players[j]["location"], dtype=np.float32)
            prev_vx   = (prev_loc[0] - pprev_loc[0]) / PITCH_LENGTH
            prev_vy   = (prev_loc[1] - pprev_loc[1]) / PITCH_WIDTH
            ax = vx - prev_vx
            ay = vy - prev_vy
        else:
            ax = ay = 0.0

        # Spatial context
        delta = GOAL_CENTER - loc
        dist  = float(np.linalg.norm(delta)) / PITCH_LENGTH
        angle = float(np.arctan2(abs(delta[1]), delta[0])) / (np.pi / 2)

        is_actor    = float(p.get("actor", False))
        is_teammate = float(p.get("teammate", False))
        is_gk       = float(p.get("keeper", False))

        rows.append([x, y, vx, vy, ax, ay, is_actor, is_teammate, is_gk, dist, angle])

    return torch.tensor(rows, dtype=torch.float)


def build_edges(players: list[dict]) -> torch.Tensor:
    """Fully connect teammates; connect opponents within proximity threshold."""
    n = len(players)
    src, dst = [], []
    locs = [np.array(p["location"], dtype=np.float32) for p in players]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same_team = players[i].get("teammate") == players[j].get("teammate")
            if same_team:
                src.append(i); dst.append(j)
            else:
                if np.linalg.norm(locs[i] - locs[j]) < CROSS_TEAM_EDGE_DIST:
                    src.append(i); dst.append(j)

    if not src:  # fallback: fully connected
        src = [i for i in range(n) for j in range(n) if i != j]
        dst = [j for i in range(n) for j in range(n) if i != j]

    return torch.tensor([src, dst], dtype=torch.long)


def frame_to_graph(row: pd.Series,
                   prev_row: pd.Series | None,
                   pprev_row: pd.Series | None) -> Data | None:
    """Convert one event row (with freeze frame) to a PyG Data object."""
    players = row["freeze_frame"]
    if not hasattr(players, "__len__") or len(players) < 2:
        return None
    players = list(players)

    prev_players  = list(prev_row["freeze_frame"])  if prev_row  is not None else None
    pprev_players = list(pprev_row["freeze_frame"]) if pprev_row is not None else None

    # Align by index (players may differ in count across frames — use min)
    if prev_players:
        n = min(len(players), len(prev_players))
        players       = players[:n]
        prev_players  = prev_players[:n]
        if pprev_players:
            pprev_players = pprev_players[:min(n, len(pprev_players))]

    x          = build_node_features(players, prev_players, pprev_players)
    edge_index = build_edges(players)

    has_shot = int("Shot" in str(row["type"]))
    return Data(x=x, edge_index=edge_index, event_type=row["type"], has_shot=has_shot)


def chain_to_graphs(chain: pd.DataFrame) -> list[Data]:
    """Convert a possession chain DataFrame to a list of PyG graphs."""
    chain = chain.reset_index(drop=True)
    chain = chain.head(MAX_CHAIN_LEN)
    graphs = []
    for i, row in chain.iterrows():
        prev  = chain.iloc[i-1] if i > 0 else None
        pprev = chain.iloc[i-2] if i > 1 else None
        g = frame_to_graph(row, prev, pprev)
        if g is not None:
            graphs.append(g)
    return graphs


def process_file(path: Path) -> tuple[list, list, list]:
    print(f"  Processing {path.name}...")
    df = pd.read_parquet(path)

    all_graphs, all_labels, all_meta = [], [], []

    for (match_id, period), group in df.groupby(["match_id", "period"]):
        chains = split_into_chains(group)
        for chain in chains:
            graphs = chain_to_graphs(chain)
            if len(graphs) < MIN_CHAIN_LEN:
                continue

            has_shot = int(chain["type"].str.contains("Shot", na=False).any())
            has_goal = int(
                chain.apply(
                    lambda r: r["type"] == "Shot" and
                              isinstance(r.get("freeze_frame"), list),
                    axis=1
                ).any()
            )  # refine below from event data if needed

            team = chain["team"].iloc[0]
            all_graphs.append(graphs)
            all_labels.append(has_shot)
            all_meta.append({
                "match_id":  match_id,
                "period":    period,
                "team":      team,
                "chain_len": len(graphs),
                "has_shot":  has_shot,
            })

    n = len(all_graphs)
    s = sum(all_labels)
    print(f"    → {n} chains  (shot chains: {s}  {s/max(n,1)*100:.1f}%)")
    return all_graphs, all_labels, all_meta


def main():
    all_graphs, all_labels, all_meta = [], [], []

    for path in sorted(RAW_DIR.glob("*.parquet")):
        g, l, m = process_file(path)
        all_graphs.extend(g)
        all_labels.extend(l)
        all_meta.extend(m)

    print(f"\nTotal chains : {len(all_graphs)}")
    print(f"Shot chains  : {sum(all_labels)}  ({sum(all_labels)/len(all_labels)*100:.1f}%)")
    print(f"Avg chain len: {sum(c['chain_len'] for c in all_meta)/len(all_meta):.1f}")

    out = OUT_DIR / "chains.pkl"
    with open(out, "wb") as f:
        pickle.dump({"graphs": all_graphs, "labels": all_labels, "meta": all_meta}, f)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
