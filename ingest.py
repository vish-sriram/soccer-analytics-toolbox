"""
Ingest StatsBomb open data for Bayer Leverkusen's 2023/24 Bundesliga season.

Saves to data/raw/:
  matches.parquet   - match metadata (34 matches)
  events.parquet    - all event data concatenated across matches
  lineups.parquet   - lineup data per match
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
from statsbombpy import sb

COMPETITION_ID = 9    # 1. Bundesliga
SEASON_ID = 281       # 2023/2024
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def ingest():
    print("Fetching matches...")
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    matches.to_parquet(RAW_DIR / "matches.parquet", index=False)
    print(f"  {len(matches)} matches saved.")

    match_ids = matches["match_id"].tolist()

    print("Fetching events...")
    events_list = []
    for i, mid in enumerate(match_ids, 1):
        events = sb.events(match_id=mid)
        events["match_id"] = mid
        events_list.append(events)
        print(f"  [{i}/{len(match_ids)}] match {mid}: {len(events)} events")

    events_df = pd.concat(events_list, ignore_index=True)
    events_df.to_parquet(RAW_DIR / "events.parquet", index=False)
    print(f"  Total events: {len(events_df):,}")

    print("Fetching lineups...")
    lineup_list = []
    for mid in match_ids:
        lineup = sb.lineups(match_id=mid)
        for team_name, players_df in lineup.items():
            players_df["match_id"] = mid
            players_df["team"] = team_name
            lineup_list.append(players_df)

    lineups_df = pd.concat(lineup_list, ignore_index=True)
    lineups_df.to_parquet(RAW_DIR / "lineups.parquet", index=False)
    print(f"  Total lineup rows: {len(lineups_df):,}")

    print("\nDone. Files written to data/raw/:")
    for f in sorted(RAW_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    ingest()
