"""Pure Elo rating layer over every 2026 game.

Input : data/processed/2026_team_games.parquet   (from clean.py)
Output: data/processed/2026_team_elo.parquet      (+ .csv mirror)

Standard Elo, single chronological pass over all 5,935 games:

    expected_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    R_A += K * (score_A - expected_A)          # score = 1 win, 0 loss

Every team starts at 1500. Win/loss only (no margin of victory) -- "pure Elo",
robust and hard to game. Cross-region comparison only works because
international events (First Stand, MSI, EWC, ...) put teams from different
leagues in the same game, chaining the regional pools onto one scale. We report
average Elo per league so you can confirm the pools actually separated.
"""
from __future__ import annotations

import argparse

import pandas as pd

import paths
import region_strength
from paths import CURRENT_YEAR

K_DEFAULT = 30
BASE = 1500.0

# Elo is only valid within a CONNECTED pool. Restricting to major regions + the
# international events (First Stand, EWC) that link them keeps every rated team
# anchored to one scale. Without this, sealed tier-2 pools (e.g. EM) farm Elo
# internally and float above real majors. This is also the Worlds-relevant set.
MAJOR_LEAGUES = {
    "LCK", "LPL", "LEC", "LCS", "LCP",   # first-seed major regions
    "PCS", "VCS", "CBLOL", "LJL",        # secondary Worlds-qualifying regions
}

# The international events are the only thing chaining the regional pools onto a
# single scale, so take region_strength's authoritative list rather than keeping
# a second copy that can silently fall behind it. It had: MSI and WLDs were both
# missing, so the 71 games of MSI 2026 -- the year's biggest cross-region event,
# and one this module's own docstring names as a linker -- were dropped from the
# ratings entirely, along with every Worlds ever played.
MAJOR_SCOPE = MAJOR_LEAGUES | region_strength.INTL_LEAGUES


def expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def build_matches(tg: pd.DataFrame) -> pd.DataFrame:
    """Collapse 2 team rows/game into one match row: winner vs loser."""
    tg = tg.sort_values(["date", "gameid"])
    matches = []
    for gid, g in tg.groupby("gameid", sort=False):
        if len(g) != 2:
            continue
        won = g[g["won"]]
        lost = g[~g["won"]]
        if len(won) != 1 or len(lost) != 1:
            continue
        w, l = won.iloc[0], lost.iloc[0]
        matches.append({
            "date": w["date"], "gameid": gid, "league": w["league"],
            "playoffs": int(w["playoffs"]),
            "winner": w["team"], "loser": l["team"],
        })
    m = pd.DataFrame(matches).sort_values(["date", "gameid"]).reset_index(drop=True)
    return m


def run_elo(matches: pd.DataFrame, k: float = K_DEFAULT) -> pd.DataFrame:
    rating: dict[str, float] = {}
    peak: dict[str, float] = {}
    wins: dict[str, int] = {}
    games: dict[str, int] = {}
    last_league: dict[str, str] = {}
    last_date: dict[str, object] = {}

    def get(t: str) -> float:
        return rating.get(t, BASE)

    for row in matches.itertuples(index=False):
        w, l = row.winner, row.loser
        rw, rl = get(w), get(l)
        ew = expected(rw, rl)
        rating[w] = rw + k * (1 - ew)
        rating[l] = rl + k * (0 - (1 - ew))
        for t, win in ((w, 1), (l, 0)):
            wins[t] = wins.get(t, 0) + win
            games[t] = games.get(t, 0) + 1
            peak[t] = max(peak.get(t, BASE), rating[t])
            last_league[t] = row.league
            last_date[t] = row.date

    out = pd.DataFrame({
        "team": list(rating.keys()),
        "elo": [round(rating[t], 1) for t in rating],
        "elo_peak": [round(peak[t], 1) for t in rating],
        "n_games": [games[t] for t in rating],
        "wins": [wins[t] for t in rating],
        "losses": [games[t] - wins[t] for t in rating],
        "win_rate": [round(wins[t] / games[t], 3) for t in rating],
        "last_league": [last_league[t] for t in rating],
        "last_game_date": [pd.Timestamp(last_date[t]).date().isoformat() for t in rating],
    })
    return out.sort_values("elo", ascending=False).reset_index(drop=True)


def win_prob(elo: pd.DataFrame, team_a: str, team_b: str) -> float:
    ra = elo.loc[elo["team"] == team_a, "elo"].iloc[0]
    rb = elo.loc[elo["team"] == team_b, "elo"].iloc[0]
    return expected(ra, rb)


def scoped_team_games(year: int = CURRENT_YEAR, all_leagues: bool = False) -> pd.DataFrame:
    """Load the season's team-games table, restricted to the connected pool."""
    tg = pd.read_parquet(paths.processed(year, "team_games"))
    tg["date"] = pd.to_datetime(tg["date"])
    if not all_leagues:
        before = tg["gameid"].nunique()
        tg = tg[tg["league"].isin(MAJOR_SCOPE)]
        print(f"Scope: major regions + international ({before} -> {tg['gameid'].nunique()} games). "
              f"Use --all-leagues to rate everything.")
    return tg


def run(year: int = CURRENT_YEAR, k: float = K_DEFAULT,
        all_leagues: bool = False) -> pd.DataFrame:
    """Rate one season and write the Elo table. Returns it for callers."""
    tg = scoped_team_games(year, all_leagues)

    print("Building match list...")
    matches = build_matches(tg)
    print(f"  {len(matches)} matches, {matches['date'].min().date()} -> {matches['date'].max().date()}")

    print(f"Running Elo (K={k})...")
    elo = run_elo(matches, k)

    paths.ensure_dirs()
    pq = paths.processed(year, "team_elo")
    elo.to_parquet(pq, index=False)
    elo.to_csv(paths.processed(year, "team_elo", "csv"), index=False)
    print(f"\nWrote {len(elo)} team ratings -> {pq}")
    return elo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--k", type=float, default=K_DEFAULT, help=f"Elo K-factor (default {K_DEFAULT})")
    ap.add_argument("--all-leagues", action="store_true",
                    help="rate every league (unanchored minor pools inflate; not Worlds-meaningful)")
    args = ap.parse_args()
    run(args.year, args.k, args.all_leagues)


if __name__ == "__main__":
    main()
