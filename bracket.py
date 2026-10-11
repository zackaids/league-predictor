"""Worlds 2026 Play-In bracket -> exact series odds and each team's path to Swiss.

Four teams, double elimination, every series Bo5, one team advances. With only
six series there are at most 64 ways the bracket can play out, so the odds are
enumerated exactly rather than simulated -- the app can re-solve on every click
of a what-if pick. Teams and pairings are hand-entered in `worlds_bracket.toml`;
results are read from the series log (league `WLDs`) as Oracle's Elixir
publishes them, and a `[results]` entry in the TOML overrides that.

Bracket (a loser's finishing place in brackets):
    O1, O2   opening series                     losers -> LB
    UB       W(O1) v W(O2)                      winner -> QF, loser -> LF
    LB       L(O1) v L(O2)                      loser out (19th)
    LF       L(UB) v W(LB)                      loser out (18th)
    QF       W(UB) v W(LF)                      winner -> Swiss, loser out (17th)

Ratings are the calibrated Elo from calibrate.py. All six Worlds leagues are in
its pool, so every Play-In team has one.

Run:
    python bracket.py        # write {year}_playin_field.csv and print the odds
"""
from __future__ import annotations

import argparse
import tomllib
from collections import defaultdict
from math import comb

import pandas as pd

import calibrate
import paths
from paths import CURRENT_YEAR

CONFIG = paths.ROOT / "worlds_bracket.toml"

WORLDS_LEAGUE = "WLDs"   # Oracle's Elixir's code for the World Championship
NEED = 3                 # every Play-In series is first-to-3

ORDER = ["O1", "O2", "UB", "LB", "LF", "QF"]
# Where each later series' two slots come from: ("W"|"L", earlier series).
FEEDS = {
    "UB": (("W", "O1"), ("W", "O2")),
    "LB": (("L", "O1"), ("L", "O2")),
    "LF": (("L", "UB"), ("W", "LB")),
    "QF": (("W", "UB"), ("W", "LF")),
}
# Finishing place of the team that loses each elimination series.
ELIMINATED = {"LB": 19, "LF": 18, "QF": 17}
SWISS = "swiss"


# ------------------------------------------------------------------ config

def load_config(path=CONFIG) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    names = [t["name"] for t in cfg["teams"]]
    if len(set(names)) != len(names) or len(names) != 4:
        raise ValueError(f"{path.name}: expected 4 distinct teams, got {names}")
    opening = {m: tuple(p) for m, p in cfg["opening"].items()}
    if set(opening) != {"O1", "O2"} or any(len(p) != 2 for p in opening.values()):
        raise ValueError(f"{path.name}: [opening] needs exactly O1 and O2, two teams each")
    if sorted(t for p in opening.values() for t in p) != sorted(names):
        raise ValueError(f"{path.name}: [opening] must use each team in `teams` exactly once")
    unknown = set(cfg.get("results", {})) - set(ORDER)
    if unknown:
        raise ValueError(f"{path.name}: unknown match id(s) in [results]: {sorted(unknown)}")
    cfg["opening"] = opening
    cfg.setdefault("results", {})
    return cfg


# ------------------------------------------------------------------ odds

def p_series(p_game: float, need: int = NEED) -> float:
    """P(win a first-to-`need` series) from a per-game win probability."""
    q = 1.0 - p_game
    return sum(comb(need - 1 + k, k) * p_game ** need * q ** k for k in range(need))


def p_match(ra: float, rb: float) -> float:
    # calibrate.prob_from_gap carries the fitted RATING_SCALE, like
    # monte_carlo.p_game: raw calibrated gaps are too wide to use directly.
    return p_series(calibrate.prob_from_gap(ra - rb))


def _pair(m: str, opening: dict, won: dict, lost: dict) -> tuple[str, str] | None:
    """The two teams in series `m`, or None while an earlier result is missing."""
    if m in opening:
        return opening[m]
    (ka, ma), (kb, mb) = FEEDS[m]
    a = (won if ka == "W" else lost).get(ma)
    b = (won if kb == "W" else lost).get(mb)
    return (a, b) if a and b else None


def determined(opening: dict, fixed: dict[str, str]) -> tuple[dict, dict[str, str]]:
    """Which series have both teams certain given `fixed` winners.

    Returns (pairs, usable): pairs[m] is the (a, b) teams or None, and `usable`
    is the part of `fixed` that is consistent -- a stale what-if pick whose
    team no longer reaches that series is dropped rather than trusted.
    """
    won, lost, pairs, usable = {}, {}, {}, {}
    for m in ORDER:
        pair = _pair(m, opening, won, lost)
        pairs[m] = pair
        if pair and fixed.get(m) in pair:
            usable[m] = fixed[m]
            won[m] = fixed[m]
            lost[m] = pair[1] if fixed[m] == pair[0] else pair[0]
    return pairs, usable


def solve(opening: dict, rating: dict[str, float],
          fixed: dict[str, str] | None = None) -> dict:
    """Exact probabilities over every way the remaining series can finish.

    `fixed` maps series id -> winner for results already decided (real or
    what-if). Returns:
        appear[m][team]  P(team plays in series m)
        wins[m][team]    P(team wins series m)
        finish[team][k]  P(team finishes k), k in {"swiss", 17, 18, 19}
    """
    _, fixed = determined(opening, fixed or {})
    appear = {m: defaultdict(float) for m in ORDER}
    wins = {m: defaultdict(float) for m in ORDER}
    finish: dict = defaultdict(lambda: defaultdict(float))
    won: dict[str, str] = {}
    lost: dict[str, str] = {}

    def walk(i: int, p: float) -> None:
        if i == len(ORDER):
            finish[won["QF"]][SWISS] += p
            for m, place in ELIMINATED.items():
                finish[lost[m]][place] += p
            return
        m = ORDER[i]
        a, b = _pair(m, opening, won, lost)
        appear[m][a] += p
        appear[m][b] += p
        if m in fixed:
            branches = [(fixed[m], 1.0)]
        else:
            pa = p_match(rating[a], rating[b])
            branches = [(a, pa), (b, 1.0 - pa)]
        for winner, q in branches:
            won[m] = winner
            lost[m] = b if winner == a else a
            wins[m][winner] += p * q
            walk(i + 1, p * q)
        del won[m], lost[m]

    walk(0, 1.0)
    return {"appear": appear, "wins": wins, "finish": finish}


# ------------------------------------------------------------------ results

def resolve_results(cfg: dict, series: pd.DataFrame) -> dict[str, dict]:
    """Decided series -> {winner, score, source, series_id}.

    A [results] entry beats auto-detection. Auto-detection matches a finished
    (first-to-3) WLDs series between the two teams inside the Play-In window,
    using each logged series once: the same pair can meet twice in a double
    elimination bracket, and the earlier meeting must not be counted again.
    """
    data = {t["name"]: t.get("data_name", t["name"]) for t in cfg["teams"]}
    display = {v: k for k, v in data.items()}
    lo = pd.Timestamp(cfg["start"])
    hi = pd.Timestamp(cfg["end"]) + pd.Timedelta(days=1)
    if series.empty:   # app.load_series() returns a column-less frame when the file is missing
        series = pd.DataFrame(columns=["series_id", "date", "league",
                                       "team_a", "team_b", "wins_a", "wins_b"])
    pool = series[(series["league"] == WORLDS_LEAGUE)
                  & (series["date"] >= lo) & (series["date"] < hi)].sort_values("date")

    used: set = set()
    won: dict[str, str] = {}
    lost: dict[str, str] = {}
    out: dict[str, dict] = {}
    for m in ORDER:
        manual = cfg["results"].get(m)
        pair = _pair(m, cfg["opening"], won, lost)
        if pair is None:
            if manual:
                raise ValueError(f"[results] {m} is set but the teams in {m} are not "
                                 f"decided yet -- fill in the earlier series first")
            continue

        # Earliest finished, still-unused series between this pair.
        found = None
        for r in pool.itertuples(index=False):
            if r.series_id in used or max(r.wins_a, r.wins_b) < NEED:
                continue
            if {display.get(r.team_a), display.get(r.team_b)} == set(pair):
                found = r
                break
        if found is not None:
            used.add(found.series_id)   # consumed even when overridden below

        if manual:
            winner = manual["winner"]
            if winner not in pair:
                raise ValueError(f"[results] {m}: {winner!r} is not in {pair[0]} v {pair[1]}")
            res = {"winner": winner, "score": manual.get("score"),
                   "source": "manual", "series_id": None}
        elif found is not None:
            a_won = found.wins_a > found.wins_b
            res = {"winner": display[found.team_a if a_won else found.team_b],
                   "score": f"{max(found.wins_a, found.wins_b)}-{min(found.wins_a, found.wins_b)}",
                   "source": "auto", "series_id": found.series_id}
        else:
            continue
        out[m] = res
        won[m] = res["winner"]
        lost[m] = pair[1] if res["winner"] == pair[0] else pair[0]
    return out


# ------------------------------------------------------------------ ratings

def team_rating(data_name: str, cal: pd.DataFrame) -> tuple[float, str]:
    """(calibrated rating, home league) for one team, by Oracle's Elixir name."""
    row = cal[cal["team"] == data_name]
    if row.empty:
        raise ValueError(f"{data_name!r} is not in calibrated_ratings -- check "
                         f"`data_name` in {CONFIG.name}")
    return float(row["calibrated"].iloc[0]), row["home_league"].iloc[0]


def build_field(cfg: dict, year: int = CURRENT_YEAR) -> pd.DataFrame:
    cal = pd.read_parquet(paths.processed(year, "calibrated_ratings"))
    rows = []
    for t in cfg["teams"]:
        data_name = t.get("data_name", t["name"])
        rating, league = team_rating(data_name, cal)
        rows.append({"team": t["name"], "data_name": data_name, "seed": t["seed"],
                     "home_league": league, "rating": rating})
    return pd.DataFrame(rows)


def write_field(year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Write the four teams' ratings and print the pre-result odds."""
    cfg = load_config()
    field = build_field(cfg, year)
    paths.ensure_dirs()
    field.to_csv(paths.processed(year, "playin_field", "csv"), index=False)

    rating = dict(zip(field["team"], field["rating"]))
    solved = solve(cfg["opening"], rating)
    print(f"=== WORLDS {year} PLAY-IN ===")
    print(field.to_string(index=False))
    print()
    for m, (a, b) in cfg["opening"].items():
        print(f"{m}: {a} v {b}  ->  {p_match(rating[a], rating[b]):.1%} / "
              f"{1 - p_match(rating[a], rating[b]):.1%}")
    print()
    for team in sorted(rating, key=lambda t: -solved["finish"][t][SWISS]):
        f = solved["finish"][team]
        print(f"{team:<14} Swiss {f[SWISS]:6.1%}   17th {f[17]:6.1%}   "
              f"18th {f[18]:6.1%}   19th {f[19]:6.1%}")
    return field


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    write_field(ap.parse_args().year)


if __name__ == "__main__":
    main()
