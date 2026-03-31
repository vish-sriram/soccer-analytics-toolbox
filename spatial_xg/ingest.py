"""
Parses StatsBomb shot freeze frames into PyTorch Geometric graphs.

Each shot becomes a graph:
  - Nodes: all visible players (shooter + teammates + opponents)
  - Edges: fully connected within each team, cross-team edges between nearby players
  - Node features: x, y, dist_to_goal, angle_to_goal, is_goalkeeper, is_teammate, is_shooter
  - Label: 1 if shot resulted in a goal, 0 otherwise
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

GOAL_CENTER = np.array([120.0, 40.0])  # StatsBomb pitch coords
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Proximity threshold for cross-team edges (in pitch units)
CROSS_TEAM_EDGE_DIST = 10.0


def _dist_angle_to_goal(xy: np.ndarray) -> tuple[float, float]:
    """Euclidean distance and angle to goal center from position xy."""
    delta = GOAL_CENTER - xy
    dist = float(np.linalg.norm(delta))
    angle = float(np.arctan2(abs(delta[1]), delta[0]))
    return dist, angle


def shot_to_graph(shot_row: pd.Series) -> Data | None:
    """
    Convert a single shot row (with freeze_frame) into a PyG Data object.
    Returns None if freeze_frame is missing or malformed.
    """
    ff = shot_row["shot_freeze_frame"]
    if not isinstance(ff, (list, np.ndarray)) or len(ff) == 0:
        return None

    shooter_loc = np.array(shot_row["location"], dtype=np.float32)
    is_goal = int(
        shot_row.get("shot_outcome_name", shot_row.get("shot_outcome", "")) == "Goal"
    )
    match_id = shot_row.get("match_id", -1)
    shot_id = shot_row.get("id", -1)

    # Build node list: shooter first, then freeze frame players
    nodes = []

    # Shooter node
    d, a = _dist_angle_to_goal(shooter_loc)
    nodes.append(
        {
            "x": shooter_loc[0] / PITCH_LENGTH,
            "y": shooter_loc[1] / PITCH_WIDTH,
            "dist_to_goal": d / PITCH_LENGTH,
            "angle_to_goal": a / (np.pi / 2),
            "is_goalkeeper": 0.0,
            "is_teammate": 1.0,
            "is_shooter": 1.0,
        }
    )

    for player in ff:
        loc = player.get("location")
        if loc is None:
            continue
        loc = np.array(loc, dtype=np.float32)
        pos_name = ""
        pos = player.get("position")
        if isinstance(pos, dict):
            pos_name = pos.get("name", "")
        is_gk = 1.0 if "Goalkeeper" in pos_name else 0.0
        is_tm = 1.0 if player.get("teammate", False) else 0.0
        d, a = _dist_angle_to_goal(loc)
        nodes.append(
            {
                "x": loc[0] / PITCH_LENGTH,
                "y": loc[1] / PITCH_WIDTH,
                "dist_to_goal": d / PITCH_LENGTH,
                "angle_to_goal": a / (np.pi / 2),
                "is_goalkeeper": is_gk,
                "is_teammate": is_tm,
                "is_shooter": 0.0,
            }
        )

    if len(nodes) < 2:
        return None

    # Node feature matrix [N, 7]
    feat_keys = ["x", "y", "dist_to_goal", "angle_to_goal", "is_goalkeeper", "is_teammate", "is_shooter"]
    x = torch.tensor([[n[k] for k in feat_keys] for n in nodes], dtype=torch.float)

    # Build edges
    src, dst = [], []
    n = len(nodes)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same_team = nodes[i]["is_teammate"] == nodes[j]["is_teammate"]
            if same_team:
                # Fully connect teammates
                src.append(i)
                dst.append(j)
            else:
                # Cross-team edge only if within proximity threshold
                xi = np.array([nodes[i]["x"] * PITCH_LENGTH, nodes[i]["y"] * PITCH_WIDTH])
                xj = np.array([nodes[j]["x"] * PITCH_LENGTH, nodes[j]["y"] * PITCH_WIDTH])
                if np.linalg.norm(xi - xj) < CROSS_TEAM_EDGE_DIST:
                    src.append(i)
                    dst.append(j)

    if len(src) == 0:
        # Fallback: fully connect all nodes
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([is_goal], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(events_path: str) -> list[Data]:
    """Load events parquet and convert all shots with freeze frames to graphs."""
    print("Loading events...")
    events = pd.read_parquet(events_path)

    shots = events[events["type"].apply(
        lambda t: (t.get("name") if isinstance(t, dict) else t) == "Shot"
    )].copy()

    # Flatten shot_outcome to name string
    shots["shot_outcome_name"] = shots["shot_outcome"].apply(
        lambda o: o.get("name") if isinstance(o, dict) else str(o)
    )

    print(f"Total shots: {len(shots)}")
    shots_with_ff = shots[shots["shot_freeze_frame"].apply(
        lambda ff: isinstance(ff, (list, np.ndarray)) and len(ff) > 0
    )]
    print(f"Shots with freeze frames: {len(shots_with_ff)}")

    goals = shots_with_ff[shots_with_ff["shot_outcome_name"] == "Goal"]
    print(f"Goals (with freeze frame): {len(goals)}")

    graphs, metadata = [], []
    for _, row in shots_with_ff.iterrows():
        g = shot_to_graph(row)
        if g is not None:
            graphs.append(g)
            metadata.append({
                "shot_id": str(row.get("id", "")),
                "match_id": int(row.get("match_id", -1)),
                "player": str(row.get("player", "")),
                "team": str(row.get("team", "")),
                "minute": int(row.get("minute", -1)),
                "shot_outcome": row.get("shot_outcome_name", ""),
                "location_x": float(row.get("location", [0, 0])[0]),
                "location_y": float(row.get("location", [0, 0])[1]),
            })

    print(f"Graphs built: {len(graphs)}")
    return graphs, metadata


if __name__ == "__main__":
    import pickle
    graphs, metadata = build_graph_dataset("data/raw/events.parquet")
    with open("data/graphs.pkl", "wb") as f:
        pickle.dump({"graphs": graphs, "metadata": metadata}, f)
    print("Saved to data/graphs.pkl")
