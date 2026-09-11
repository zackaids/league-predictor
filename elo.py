"""Pure Elo rating layer over every 2026 game.

Input : data/processed/2026_team_games.parquet   (from clean.py)
Output: data/processed/2026_team_elo.parquet      (+ .csv mirror)
        data/processed/2026_series.parquet        (+ .csv mirror; one row per
                                                   series with each team's Elo
                                                   change, read by app.py)

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

# A new series starts when a pair's game number stops increasing, or when the
# pair has not played for this long. The gap is the fallback for games with no
# game number.
SERIES_GAP = pd.Timedelta(hours=12)

# A series win counts as an upset when the winner's pre-series chance of taking
# any single game was below this.
UPSET_P = 0.40

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
            # .get: a team-games table written before clean.py kept `game`
            # still rates; group_series then falls back to the time gap.
            "game": w.get("game", float("nan")),
            "winner": w["team"], "loser": l["team"],
        })
    m = pd.DataFrame(matches).sort_values(["date", "gameid"]).reset_index(drop=True)
    return m


def run_elo(matches: pd.DataFrame, k: float = K_DEFAULT,
            game_log: list[dict] | None = None) -> pd.DataFrame:
    """Rate `matches` in order. If `game_log` is given, append one dict per game.

    The log is filled from this loop rather than a second replay, so the rating
    changes the app shows cannot drift from the ratings themselves.
    """
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
        if game_log is not None:
            game_log.append({
                "gameid": row.gameid, "date": row.date, "league": row.league,
                "playoffs": row.playoffs, "game": getattr(row, "game", float("nan")),
                "winner": w, "loser": l,
                "winner_elo_before": rw, "loser_elo_before": rl,
                "delta": k * (1 - ew), "p_winner": ew,
            })
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


def replay(matches: pd.DataFrame, k: float = K_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rate the matches and keep every game's rating change.

    Returns (team table, games): the table is exactly `run_elo`'s; `games` has
    one row per game with both teams' Elo before it and the points exchanged.
    """
    log: list[dict] = []
    table = run_elo(matches, k, game_log=log)
    return table, pd.DataFrame(log)


def group_series(games: pd.DataFrame) -> pd.DataFrame:
    """Collapse `replay`'s per-game log into one row per series."""
    games = games.sort_values(["date", "gameid"]).reset_index(drop=True)
    open_series: dict[tuple, tuple] = {}   # (league, pair) -> (series_id, game, date)
    ids = []
    for row in games.itertuples(index=False):
        key = (row.league, frozenset((row.winner, row.loser)))
        prev = open_series.get(key)
        restarted = (prev is not None and pd.notna(row.game) and pd.notna(prev[1])
                     and row.game <= prev[1])
        if prev is None or restarted or row.date - prev[2] > SERIES_GAP:
            series_id = row.gameid
        else:
            series_id = prev[0]
        open_series[key] = (series_id, row.game, row.date)
        ids.append(series_id)
    games["series_id"] = ids

    out = pd.DataFrame([_series_row(sid, g)
                        for sid, g in games.groupby("series_id", sort=False)])
    out.insert(out.columns.get_loc("upset"), "swing", out["delta_a"].abs())
    return out


def _series_row(series_id: str, g: pd.DataFrame) -> dict:
    first, last = g.iloc[0], g.iloc[-1]
    wins = g["winner"].value_counts()
    t1, t2 = first["winner"], first["loser"]
    if wins.get(t1, 0) == wins.get(t2, 0):
        a, b = sorted((t1, t2))
    elif wins.get(t1, 0) > wins.get(t2, 0):
        a, b = t1, t2
    else:
        a, b = t2, t1

    def before(team: str, game: pd.Series) -> float:
        return game["winner_elo_before"] if game["winner"] == team else game["loser_elo_before"]

    def after(team: str, game: pd.Series) -> float:
        return before(team, game) + (game["delta"] if game["winner"] == team else -game["delta"])

    wins_a, wins_b = int(wins.get(a, 0)), int(wins.get(b, 0))
    a_before, b_before = before(a, first), before(b, first)
    return {
        "series_id": series_id, "date": first["date"], "league": first["league"],
        "playoffs": int(first["playoffs"]), "team_a": a, "team_b": b,
        "wins_a": wins_a, "wins_b": wins_b,
        "result": "draw" if wins_a == wins_b else "win",
        "games": "|".join(g["winner"]),
        "a_elo_before": round(a_before, 1), "a_elo_after": round(after(a, last), 1),
        "b_elo_before": round(b_before, 1), "b_elo_after": round(after(b, last), 1),
        # Each game is zero-sum, so team_b's change is always -delta_a.
        "delta_a": round(float(g["delta"].where(g["winner"] == a, -g["delta"]).sum()), 1),
        "upset": bool(wins_a > wins_b and expected(a_before, b_before) < UPSET_P),
    }


def team_perspective(series: pd.DataFrame, team: str) -> pd.DataFrame:
    """Every series `team` played, from its own side, newest first."""
    keep = ["date", "league", "playoffs", "upset"]
    a = series[series["team_a"] == team]
    b = series[series["team_b"] == team]
    out = pd.concat([
        a[keep].assign(opponent=a["team_b"], won=a["wins_a"], lost=a["wins_b"],
                       elo_change=a["delta_a"], elo_after=a["a_elo_after"]),
        b[keep].assign(opponent=b["team_a"], won=b["wins_b"], lost=b["wins_a"],
                       elo_change=-b["delta_a"], elo_after=b["b_elo_after"]),
    ]).sort_values("date", ascending=False).reset_index(drop=True)
    out["outcome"] = "D"
    out.loc[out["won"] > out["lost"], "outcome"] = "W"
    out.loc[out["won"] < out["lost"], "outcome"] = "L"
    out["score"] = out["won"].astype(str) + "-" + out["lost"].astype(str)
    out["stage"] = out["playoffs"].map({1: "Playoffs", 0: "Regular"})
    return out[["date", "league", "stage", "opponent", "outcome", "score",
                "won", "lost", "elo_change", "elo_after", "upset"]]


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
    elo, games = replay(matches, k)
    series = group_series(games)

    paths.ensure_dirs()
    pq = paths.processed(year, "team_elo")
    elo.to_parquet(pq, index=False)
    elo.to_csv(paths.processed(year, "team_elo", "csv"), index=False)
    print(f"\nWrote {len(elo)} team ratings -> {pq}")

    # team_games is gitignored, so the app cannot rebuild this; commit it.
    series_pq = paths.processed(year, "series")
    series.to_parquet(series_pq, index=False)
    series.to_csv(paths.processed(year, "series", "csv"), index=False)
    print(f"Wrote {len(series)} series ({len(games)} games) -> {series_pq}")
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
