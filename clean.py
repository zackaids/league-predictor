"""Clean one season of Oracle's Elixir data into a tidy team-per-game table.

Input : data/raw/{year}_LoL_esports_match_data_from_OraclesElixir.csv
Output: data/processed/{year}_team_games.parquet  (+ .csv mirror)

Grain: one row per team per game (2 rows per game). We drop the 10 individual
player rows and keep only the team-aggregate rows (position == "team"), because
we're predicting *team* outcomes.

------------------------------------------------------------------------------
FEATURES WE ACTUALLY WANT (and why), for predicting team strength -> Worlds odds
------------------------------------------------------------------------------
Context / keys (not model inputs, but needed for filtering, ratings, splits):
    gameid, date, league, split, playoffs, patch, side, team, teamid,
    datacompleteness, has_timeline

Target:
    won                 result as bool (what any game-outcome model predicts)

The strength signals, grouped. Differentials (team - opponent) are preferred
over raw counts because they're symmetric and less league-pace dependent.

  Early game / tempo  -- who's ahead before teamfighting decides things
    gd10, gd15          gold diff at 10 / 15 min   (timeline; NaN on partial)
    xpd15, csd15        xp / cs diff at 15 min     (timeline; NaN on partial)
    first_blood         won the first kill          (0/1)
    first_dragon        first dragon                (0/1)
    first_herald        first rift herald           (0/1)
    first_tower         first tower                 (0/1)

  Macro / objectives  -- converting leads into map control
    dragon_diff         dragons - opp_dragons
    herald_diff         heralds - opp_heralds
    grub_diff           void_grubs - opp_void_grubs
    baron_diff          barons - opp_barons
    tower_diff          towers - opp_towers
    plate_diff          turretplates - opp_turretplates
    first_baron         first baron                 (0/1)

  Economy / efficiency -- normalized so it compares across game lengths
    gspd                gold-spent % diff (OE's normalized gold-lead metric)
    earned_gpm          earned gold per minute
    gpr                 gold % rating (timeline; NaN on partial)

  Combat / aggression
    ckpm                combined kills per minute (game pace)
    team_kpm            team kills per minute
    dpm                 damage to champions per minute

  Vision / control
    vspm                vision score per minute
    wcpm                wards cleared per minute

Deliberately excluded: per-player columns (wrong grain), draft/ban columns
(champion-meta modeling is a separate effort), raw at-20/25 timeline splits
(at-15 already captures the early-lead signal; more just adds NaNs), and
`url`/`participantid` (not predictive).
------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import paths
from paths import CURRENT_YEAR

# raw column -> clean name (straight passthroughs)
RENAME = {
    "gamelength": "gamelength_sec",
    "firstblood": "first_blood",
    "firstdragon": "first_dragon",
    "firstherald": "first_herald",
    "firsttower": "first_tower",
    "firstbaron": "first_baron",
    "golddiffat10": "gd10",
    "golddiffat15": "gd15",
    "xpdiffat15": "xpd15",
    "csdiffat15": "csd15",
    "earned gpm": "earned_gpm",
    "team kpm": "team_kpm",
    "teamname": "team",
}

# clean_name -> (col, opp_col) differentials we derive
DIFFS = {
    "dragon_diff": ("dragons", "opp_dragons"),
    "herald_diff": ("heralds", "opp_heralds"),
    "grub_diff": ("void_grubs", "opp_void_grubs"),
    "baron_diff": ("barons", "opp_barons"),
    "tower_diff": ("towers", "opp_towers"),
    "plate_diff": ("turretplates", "opp_turretplates"),
}

CONTEXT = ["gameid", "date", "league", "split", "playoffs", "patch",
           "side", "team", "teamid", "datacompleteness"]

PASSTHROUGH = ["gspd", "gpr", "ckpm", "team_kpm", "dpm", "vspm", "wcpm",
               "earned_gpm", "gd10", "gd15", "xpd15", "csd15",
               "first_blood", "first_dragon", "first_herald",
               "first_tower", "first_baron"]

BINARY = ["first_blood", "first_dragon", "first_herald", "first_tower",
          "first_baron", "playoffs"]


def load_team_rows(path: Path | None = None, year: int = CURRENT_YEAR) -> pd.DataFrame:
    path = path or paths.raw_csv(year)
    df = pd.read_csv(path, low_memory=False)
    team = df[df["position"] == "team"].copy()
    return team


def validate(team: pd.DataFrame) -> None:
    """Sanity checks: 2 team rows/game, opposite sides, exactly one winner."""
    per_game = team.groupby("gameid").size()
    bad = per_game[per_game != 2]
    if len(bad):
        raise ValueError(f"{len(bad)} games do not have exactly 2 team rows: "
                         f"{bad.head().to_dict()}")

    sides = team.groupby("gameid")["side"].apply(lambda s: set(s))
    bad_sides = sides[sides != {"Blue", "Red"}]
    if len(bad_sides):
        raise ValueError(f"{len(bad_sides)} games lack a Blue/Red pair")

    wins = team.groupby("gameid")["result"].sum()
    bad_wins = wins[wins != 1]
    if len(bad_wins):
        raise ValueError(f"{len(bad_wins)} games do not have exactly one winner")

    print(f"  validated {per_game.size} games x 2 team rows")


def build(team: pd.DataFrame) -> pd.DataFrame:
    team = team.rename(columns=RENAME)
    out = team[CONTEXT].copy()

    # types
    out["date"] = pd.to_datetime(out["date"])
    out["playoffs"] = team["playoffs"].astype("int8")

    # target + game length
    out["won"] = team["result"].astype(bool)
    out["gamelength_min"] = (team["gamelength_sec"] / 60).round(2)
    out["has_timeline"] = team["datacompleteness"].eq("complete")

    # passthrough features
    for c in PASSTHROUGH:
        out[c] = pd.to_numeric(team[c], errors="coerce")

    # binary flags -> int8
    for c in BINARY:
        out[c] = out[c].fillna(0).astype("int8")

    # derived differentials
    for name, (a, b) in DIFFS.items():
        out[name] = pd.to_numeric(team[a], errors="coerce") - pd.to_numeric(team[b], errors="coerce")

    return out.reset_index(drop=True)


def run(year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Clean one season and write the team-games table. Returns it for callers."""
    raw = paths.raw_csv(year)
    print("Loading team rows from", raw)
    team = load_team_rows(raw)
    print(f"  {len(team)} team rows, {team['gameid'].nunique()} games")

    print("Validating...")
    validate(team)

    print("Building clean table...")
    clean = build(team)

    paths.ensure_dirs()
    pq = paths.processed(year, "team_games")
    clean.to_parquet(pq, index=False)
    clean.to_csv(paths.processed(year, "team_games", "csv"), index=False)

    n_feat = clean.shape[1] - len(CONTEXT) - 3  # minus context + won/gamelength/has_timeline
    print(f"\nWrote {len(clean)} rows x {clean.shape[1]} cols -> {pq}")
    print(f"  ({n_feat} feature columns)")
    print(f"  timeline available on {clean['has_timeline'].mean()*100:.1f}% of rows")
    return clean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    run(ap.parse_args().year)


if __name__ == "__main__":
    main()
