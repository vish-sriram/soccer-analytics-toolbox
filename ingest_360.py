"""
Ingest 360 event data for three competitions:
  - FIFA World Cup 2022       (64 matches)
  - UEFA Euro 2024            (51 matches)
  - Bundesliga 2023/24        (34 matches)

For each match, fetches:
  - events (statsbombpy)
  - 360 freeze frames (raw GitHub JSON, keyed by event_uuid)

Saves per-competition parquet files to data/360/raw/.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import requests
import pandas as pd
from pathlib import Path
from statsbombpy import sb

RAW_DIR = Path("data/360/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_360_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty"

COMPETITIONS = [
    (43, 106, "wc2022"),
    (55, 282, "euro2024"),
    (9,  281, "bundesliga2324"),
]

# Event types to keep — exclude bookkeeping events with no spatial content
KEEP_TYPES = {
    "Pass", "Carry", "Shot", "Dribble", "Ball Receipt*",
    "Pressure", "Interception", "Block", "Clearance",
    "Ball Recovery", "Miscontrol", "Duel", "Foul Committed",
    "Foul Won", "Dispossessed", "Dribbled Past",
}


def fetch_frames(match_id: int) -> dict[str, list]:
    """Download 360 JSON for a match, return {event_uuid: freeze_frame}."""
    url = f"{BASE_360_URL}/{match_id}.json"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return {f["event_uuid"]: f for f in r.json()}


def flatten_event(row: pd.Series, frames: dict) -> dict | None:
    """
    Merge one event row with its 360 freeze frame.
    Returns None if the event has no freeze frame or isn't a kept type.
    """
    type_name = row["type"] if isinstance(row["type"], str) else row["type"].get("name", "")
    if type_name not in KEEP_TYPES:
        return None

    frame = frames.get(row["id"])
    if frame is None:
        return None

    team = row["team"] if isinstance(row["team"], str) else row["team"].get("name", "")
    player = row.get("player", "")
    if isinstance(player, dict):
        player = player.get("name", "")

    location = row.get("location") or [None, None]

    return {
        "event_id":    row["id"],
        "match_id":    row["match_id"],
        "period":      row["period"],
        "timestamp":   row["timestamp"],
        "minute":      row["minute"],
        "second":      row["second"],
        "index":       row["index"],
        "type":        type_name,
        "team":        team,
        "player":      player,
        "location_x":  location[0],
        "location_y":  location[1],
        "freeze_frame": frame["freeze_frame"],
        "visible_area": frame.get("visible_area"),
    }


def ingest_competition(comp_id: int, season_id: int, tag: str):
    out_path = RAW_DIR / f"{tag}.parquet"
    if out_path.exists():
        print(f"  {tag}: already exists, skipping.")
        return

    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    print(f"  {tag}: {len(matches)} matches")

    all_rows = []
    for i, (_, match) in enumerate(matches.iterrows()):
        mid = match["match_id"]
        try:
            events = sb.events(match_id=mid)
            events["match_id"] = mid
            frames = fetch_frames(mid)

            for _, row in events.iterrows():
                rec = flatten_event(row, frames)
                if rec:
                    all_rows.append(rec)

            print(f"    [{i+1}/{len(matches)}] match {mid}: "
                  f"{len(events)} events, {len(frames)} frames")
        except Exception as e:
            print(f"    [{i+1}/{len(matches)}] match {mid}: ERROR — {e}")

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")


def main():
    for comp_id, season_id, tag in COMPETITIONS:
        print(f"\n{'='*50}")
        print(f"Ingesting {tag}...")
        ingest_competition(comp_id, season_id, tag)
    print("\nDone.")


if __name__ == "__main__":
    main()
