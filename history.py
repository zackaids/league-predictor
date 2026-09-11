"""Append-only ratings history, so rankings can be tracked over time.

Two entry points:

  record(year)    Append one snapshot from whatever the pipeline just wrote.
                  Called at the end of every pipeline run.

  backfill(year)  Reconstruct the whole season's history at once by replaying
                  Elo over prefixes of the match list. This works because
                  `elo.run_elo` is a single chronological pass: running it over
                  the games up to date D gives exactly the ratings as they stood
                  on D. No need to wait weeks for a timeline to accumulate.

LOOKAHEAD CAVEAT
----------------
Backfilled snapshots use *today's* region prior (region_strength.csv), not the
prior as it stood that week, because the prior is rebuilt from all years at
once. Region strength is slow-moving, so this is fine for a display timeline.
It is NOT acceptable for model evaluation -- backtest.py must recompute the
prior point-in-time, or it will score itself with information from the future.
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

import calibrate
import elo
import paths
from paths import CURRENT_YEAR

RATINGS = "ratings_history"
ODDS = "title_odds_history"

# Bump when rating logic changes, so old rows stay interpretable and the model
# health page can tell "the rating moved" from "the model changed".
PIPELINE_VERSION = 1


def _append(name: str, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Merge `new` into the history table, replacing whole snapshot dates.

    A snapshot is a complete picture of one date, so the entire date partition
    is replaced rather than merged per key. Merging per (date, team) would
    strand rows for teams that dropped out of the ranking between runs -- e.g.
    a team later excluded for not really belonging to a top league would linger
    in an old snapshot forever.
    """
    paths.ensure_dirs()
    path = paths.history(name)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["snapshot_date"].isin(set(new["snapshot_date"]))]
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = (combined
                .drop_duplicates(subset=keys, keep="last")
                .sort_values(keys)
                .reset_index(drop=True))
    combined.to_parquet(path, index=False)
    print(f"  {name}: +{len(new)} rows this snapshot, {len(combined)} total -> {path}")
    return combined


def _stamp(df: pd.DataFrame, snapshot: str, as_of: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "snapshot_date", snapshot)
    df.insert(1, "as_of_date", as_of)
    df["pipeline_version"] = PIPELINE_VERSION
    return df


def ranked(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = df["calibrated"].rank(ascending=False, method="min").astype("Int64")
    return df


def record(year: int = CURRENT_YEAR, snapshot_date: str | None = None) -> pd.DataFrame:
    """Append today's snapshot from the current processed outputs."""
    ratings = pd.read_parquet(paths.processed(year, "calibrated_ratings"))
    tg = pd.read_parquet(paths.processed(year, "team_games"))

    # as_of = latest game actually in the data, which is what the rating reflects.
    # snapshot = when we computed it. They differ, and both matter.
    as_of = pd.to_datetime(tg["date"]).max().date().isoformat()
    snapshot = snapshot_date or dt.date.today().isoformat()

    out = _append(RATINGS, _stamp(ranked(ratings), snapshot, as_of),
                  keys=["snapshot_date", "team"])

    odds_path = paths.processed(year, "worlds_title_odds", "csv")
    if odds_path.exists():
        _append(ODDS, _stamp(pd.read_csv(odds_path), snapshot, as_of),
                keys=["snapshot_date", "team"])
    return out


def backfill(year: int = CURRENT_YEAR, step_days: int = 7,
             k: float = elo.K_DEFAULT, all_leagues: bool = False) -> pd.DataFrame:
    """Reconstruct the season's rating history by replaying Elo over prefixes."""
    tg = elo.scoped_team_games(year, all_leagues)
    matches = elo.build_matches(tg)
    dates = pd.to_datetime(matches["date"])

    region = calibrate.load_region_prior()
    # home_league needs the UNSCOPED table: it decides whether a top-league
    # appearance is a team's real home by comparing against all its domestic
    # play, and ERL leagues like EM/LFL are outside elo's MAJOR_SCOPE.
    tg_all = pd.read_parquet(paths.processed(year, "team_games"))
    tg_all["date"] = pd.to_datetime(tg_all["date"])
    # Home league is a season-level property; computing it once keeps snapshots
    # comparable instead of letting a team's league flicker early in the year.
    homes = calibrate.home_league(tg_all, region, year)

    start, end = dates.min(), dates.max()
    cutoffs = list(pd.date_range(start, end, freq=f"{step_days}D"))
    if not cutoffs or cutoffs[-1] < end:
        cutoffs.append(end)

    print(f"Replaying {len(matches)} matches at {len(cutoffs)} weekly cutoffs "
          f"({start.date()} -> {end.date()})")

    frames = []
    for cutoff in cutoffs:
        prefix = matches[dates <= cutoff]
        if prefix.empty:
            continue
        ratings = calibrate.combine(homes, elo.run_elo(prefix, k), region)
        stamp = cutoff.date().isoformat()
        frames.append(_stamp(ranked(ratings), stamp, stamp))

    if not frames:
        raise ValueError("No snapshots produced -- is the team-games table empty?")

    return _append(RATINGS, pd.concat(frames, ignore_index=True),
                   keys=["snapshot_date", "team"])


def load(name: str = RATINGS) -> pd.DataFrame:
    """Read a history table, or an empty frame if it does not exist yet."""
    path = paths.history(name)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--backfill", action="store_true",
                    help="reconstruct the full season history by replaying Elo")
    ap.add_argument("--step-days", type=int, default=7,
                    help="backfill snapshot interval in days (default 7)")
    args = ap.parse_args()

    if args.backfill:
        backfill(args.year, args.step_days)
    else:
        record(args.year)


if __name__ == "__main__":
    main()
