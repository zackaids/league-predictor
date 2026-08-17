"""Monte Carlo simulation of Worlds 2026 -> each team's title odds.

Uses the cross-region calibrated ratings (calibrate.py). The Worlds field is NOT
yet confirmed, so we build a PROJECTED 16-team field from the top calibrated
teams per qualifying league (slot counts are configurable), with the MSI 2026
champion (Hanwha Life) locked in as guaranteed.

Format modelled (current Worlds structure):
  * Swiss stage, 16 teams. Each round pairs teams on the same W-L record
    (rematches avoided). A team that reaches 3 wins advances; 3 losses is out.
    Advancement/elimination matches (a team on 2W or 2L) are Bo3; others Bo1.
    Continues until 8 teams have advanced.
  * Knockout: 8 teams seeded by rating, single-elimination Bo5 to a champion.

Game win prob from calibrated Elo:  p = 1 / (1 + 10 ** ((rb - ra)/400)).
Series are simulated game-by-game (independent games).

Run:
    python monte_carlo.py                 # default field + 20000 sims
    python monte_carlo.py --n 50000
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/processed")

# projected Worlds slots per qualifying league (sum must be 16)
DEFAULT_SLOTS = {"LCK": 4, "LPL": 4, "LEC": 3, "LCP": 3, "LCS": 2}
GUARANTEED = ["Hanwha Life Esports"]  # MSI 2026 champion, locked in


def build_field(df: pd.DataFrame, slots: dict[str, int]) -> pd.DataFrame:
    picks = []
    for league, n in slots.items():
        pool = df[df["home_league"] == league].sort_values("calibrated", ascending=False)
        picks.append(pool.head(n))
    field = pd.concat(picks)
    # make sure guaranteed teams are present (swap in if a league cut them)
    for g in GUARANTEED:
        if g not in field["team"].values and g in df["team"].values:
            field = pd.concat([field, df[df["team"] == g]])
    field = field.drop_duplicates("team").reset_index(drop=True)

    bad = field[field["calibrated"].isna()]
    if len(bad):
        raise ValueError(f"Teams with no calibrated rating in field: "
                         f"{bad['team'].tolist()}")
    return field


def p_game(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def sim_series(ra: float, rb: float, need: int) -> bool:
    """True if team A (rating ra) wins a first-to-`need` series."""
    p = p_game(ra, rb)
    a = b = 0
    while a < need and b < need:
        if random.random() < p:
            a += 1
        else:
            b += 1
    return a == need


def swiss_pairings(teams: list[str], played: set) -> list[tuple[str, str]]:
    """Pair teams sharing a W-L bucket, avoiding rematches where possible."""
    pool = teams[:]
    random.shuffle(pool)
    pairs, used = [], set()
    for i, t in enumerate(pool):
        if t in used:
            continue
        opp = None
        for u in pool[i + 1:]:
            if u not in used and frozenset((t, u)) not in played:
                opp = u
                break
        if opp is None:  # forced rematch
            for u in pool[i + 1:]:
                if u not in used:
                    opp = u
                    break
        if opp is not None:
            pairs.append((t, opp))
            used.add(t); used.add(opp)
    return pairs


def sim_swiss(field: dict[str, float]) -> list[str]:
    """Return the 8 teams that advance from the Swiss stage."""
    wins = defaultdict(int)
    losses = defaultdict(int)
    played = set()
    active = list(field)
    advanced = []

    while len(advanced) < 8:
        buckets = defaultdict(list)
        for t in active:
            buckets[(wins[t], losses[t])].append(t)

        for _, teams in buckets.items():
            for a, b in swiss_pairings(teams, played):
                played.add(frozenset((a, b)))
                # Bo3 if this match can advance or eliminate someone, else Bo1
                decisive = max(wins[a], wins[b]) == 2 or max(losses[a], losses[b]) == 2
                need = 2 if decisive else 1
                a_wins = sim_series(field[a], field[b], need)
                w, l = (a, b) if a_wins else (b, a)
                wins[w] += 1
                losses[l] += 1

        active = [t for t in active if wins[t] < 3 and losses[t] < 3]
        advanced = [t for t in field if wins[t] == 3]

    return sorted(advanced, key=lambda t: field[t], reverse=True)


def sim_knockout(quals: list[str], field: dict[str, float]) -> str:
    """8-team single-elim Bo5, seeded by rating (1v8,4v5,2v7,3v6)."""
    order = [0, 7, 3, 4, 1, 6, 2, 5]
    bracket = [quals[i] for i in order]
    while len(bracket) > 1:
        nxt = []
        for i in range(0, len(bracket), 2):
            a, b = bracket[i], bracket[i + 1]
            nxt.append(a if sim_series(field[a], field[b], 3) else b)
        bracket = nxt
    return bracket[0]


def run(field: dict[str, float], n: int) -> pd.DataFrame:
    titles = defaultdict(int)
    finals = defaultdict(int)
    knockouts = defaultdict(int)
    for _ in range(n):
        quals = sim_swiss(field)
        for t in quals:
            knockouts[t] += 1
        # track finalists by replaying the last round
        order = [0, 7, 3, 4, 1, 6, 2, 5]
        bracket = [quals[i] for i in order]
        while len(bracket) > 2:
            bracket = [bracket[i] if sim_series(field[bracket[i]], field[bracket[i + 1]], 3)
                       else bracket[i + 1] for i in range(0, len(bracket), 2)]
        a, b = bracket
        finals[a] += 1; finals[b] += 1
        champ = a if sim_series(field[a], field[b], 3) else b
        titles[champ] += 1

    rows = [{"team": t, "title_%": round(100 * titles[t] / n, 1),
             "finals_%": round(100 * finals[t] / n, 1),
             "knockout_%": round(100 * knockouts[t] / n, 1)} for t in field]
    return pd.DataFrame(rows).sort_values("title_%", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000, help="number of simulations")
    args = ap.parse_args()

    df = pd.read_parquet(OUT_DIR / "2026_calibrated_ratings.parquet")
    field_df = build_field(df, DEFAULT_SLOTS)
    field = dict(zip(field_df["team"], field_df["calibrated"]))
    print(f"Projected {len(field)}-team field "
          f"({', '.join(f'{k}:{v}' for k, v in DEFAULT_SLOTS.items())}); "
          f"guaranteed: {', '.join(GUARANTEED)}")
    print(f"Running {args.n:,} simulations...\n")

    res = run(field, args.n)
    res = res.merge(field_df[["team", "home_league", "calibrated"]], on="team")
    res = res.sort_values("title_%", ascending=False)[
        ["team", "home_league", "calibrated", "title_%", "finals_%", "knockout_%"]]
    res.to_csv(OUT_DIR / "2026_worlds_title_odds.csv", index=False)
    print("=== WORLDS 2026 — PROJECTED TITLE ODDS ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
