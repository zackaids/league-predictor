"""Team-season aggregation: turn the clean team-per-game table into one
recency-weighted strength profile per team.

Input : data/processed/2026_team_games.parquet   (from clean.py)
        data/raw/2026_..._OraclesElixir.csv       (for per-game rosters)
Output: data/processed/2026_team_profiles.parquet (+ .csv mirror)

WHY WEIGHTED, NOT A PLAIN SEASON MEAN
-------------------------------------
Two things make an old game less representative of a team's Worlds strength:
  1. Form   - recent games reflect the current meta and how the team is playing.
  2. Roster - lineups change between splits; a game played by a different five
              tells us little about the team that will actually attend Worlds.

So each past game gets a weight = time_decay * roster_overlap:

  time_decay    = 0.5 ** (days_before_latest / HALF_LIFE_DAYS)
                  -> a game HALF_LIFE_DAYS old counts half as much.

  roster_overlap = (# of the team's CURRENT starting five that played in that
                    game) / 5
                  -> a game by a totally different lineup counts ~0; a game
                    where 4/5 current starters played counts 0.8.

Every feature is then a weighted mean over that team's games (weighted by the
above, and skipping games where the feature is null, e.g. timeline stats on
partial-data games). `effective_games` = sum of weights = the weighted sample
size, so we can tell strong signal (many recent, same-roster games) from thin.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import paths
from paths import CURRENT_YEAR

HALF_LIFE_DAYS = 45  # a game this many days old counts half as much

# feature columns to aggregate (from clean.py)
FEATURES = [
    "gspd", "gpr", "ckpm", "team_kpm", "dpm", "vspm", "wcpm", "earned_gpm",
    "gd10", "gd15", "xpd15", "csd15",
    "first_blood", "first_dragon", "first_herald", "first_tower", "first_baron",
    "dragon_diff", "herald_diff", "grub_diff", "baron_diff", "tower_diff",
    "plate_diff", "gamelength_min",
]


def game_rosters(raw: Path | None = None,
                 year: int = CURRENT_YEAR) -> dict[tuple[str, str], frozenset[str]]:
    """{(gameid, teamname): frozenset of 5 player names} from the raw player rows."""
    df = pd.read_csv(raw or paths.raw_csv(year), low_memory=False,
                     usecols=["gameid", "teamname", "position", "playername"])
    players = df[df["position"] != "team"]
    rosters: dict[tuple[str, str], frozenset[str]] = {}
    for (gid, team), grp in players.groupby(["gameid", "teamname"]):
        rosters[(gid, team)] = frozenset(grp["playername"])
    return rosters


def current_rosters(tg: pd.DataFrame, rosters: dict) -> dict[str, frozenset[str]]:
    """Each team's current five = roster of its most recent game that has one."""
    cur: dict[str, frozenset[str]] = {}
    for team, grp in tg.sort_values("date").groupby("team"):
        for gid in reversed(grp["gameid"].tolist()):
            r = rosters.get((gid, team))
            if r:
                cur[team] = r
                break
    return cur


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean ignoring NaN values; NaN if no weight remains."""
    mask = ~np.isnan(values)
    w = weights[mask]
    denom = w.sum()
    if denom <= 0:
        return np.nan
    return float((w * values[mask]).sum() / denom)


def build_profiles(tg: pd.DataFrame, rosters: dict, half_life: float) -> pd.DataFrame:
    as_of = tg["date"].max()
    cur = current_rosters(tg, rosters)

    # per-row weights
    days_ago = (as_of - tg["date"]).dt.total_seconds() / 86400
    tg = tg.assign(w_time=np.power(0.5, days_ago / half_life))

    def overlap(row) -> float:
        r = rosters.get((row["gameid"], row["team"]))
        c = cur.get(row["team"])
        if not r or not c:
            return np.nan  # unknown roster (partial game) -> imputed below
        return len(r & c) / 5.0

    tg = tg.assign(roster_overlap=tg.apply(overlap, axis=1))
    # impute unknown overlap with the team's median known overlap (fallback 1.0)
    tg["roster_overlap"] = tg.groupby("team")["roster_overlap"].transform(
        lambda s: s.fillna(s.median() if s.notna().any() else 1.0)
    )
    tg["weight"] = tg["w_time"] * tg["roster_overlap"]

    rows = []
    for team, g in tg.groupby("team"):
        w = g["weight"].to_numpy()
        recent = g.sort_values("date").iloc[-1]
        rec = {
            "team": team,
            "teamid": g["teamid"].dropna().iloc[-1] if g["teamid"].notna().any() else None,
            "league": recent["league"],
            "n_leagues": g["league"].nunique(),
            "n_games": len(g),
            "effective_games": round(float(w.sum()), 2),
            "last_game_date": recent["date"].date().isoformat(),
            "days_since_last": int((as_of - recent["date"]).days),
            "win_rate": round(float(g["won"].mean()), 4),
            "win_rate_w": round(weighted_mean(g["won"].to_numpy(float), w), 4),
            "current_roster": " / ".join(sorted(cur.get(team, []))),
        }
        for f in FEATURES:
            rec[f] = weighted_mean(g[f].to_numpy(float), w)
        rows.append(rec)

    profiles = pd.DataFrame(rows)
    # round feature columns for readability
    for f in FEATURES:
        profiles[f] = profiles[f].round(3)
    return profiles.sort_values("win_rate_w", ascending=False).reset_index(drop=True)


def run(year: int = CURRENT_YEAR, half_life: float = HALF_LIFE_DAYS) -> pd.DataFrame:
    team_games = paths.processed(year, "team_games")
    print("Loading", team_games)
    tg = pd.read_parquet(team_games)
    tg["date"] = pd.to_datetime(tg["date"])
    print(f"  {len(tg)} team-game rows, {tg['team'].nunique()} teams")

    print("Extracting per-game rosters from raw...")
    rosters = game_rosters(year=year)
    print(f"  {len(rosters)} team-game rosters")

    print(f"Building profiles (half-life={half_life}d)...")
    profiles = build_profiles(tg, rosters, half_life)

    paths.ensure_dirs()
    pq = paths.processed(year, "team_profiles")
    profiles.to_parquet(pq, index=False)
    profiles.to_csv(paths.processed(year, "team_profiles", "csv"), index=False)
    print(f"\nWrote {len(profiles)} team profiles x {profiles.shape[1]} cols -> {pq}")
    return profiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--half-life", type=float, default=HALF_LIFE_DAYS,
                    help=f"time-decay half-life in days (default {HALF_LIFE_DAYS})")
    args = ap.parse_args()
    run(args.year, args.half_life)


if __name__ == "__main__":
    main()
