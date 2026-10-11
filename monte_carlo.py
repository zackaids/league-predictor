"""Worlds 2026 odds: a vectorized Swiss + knockout simulator.

Models the real format (lolesports.com/news/worlds-2026-primer), not a
generic one:

  * Swiss, 16 teams, 5 rounds. Three wins advance, three losses are out.
    A match is Bo3 when it can advance or eliminate someone (the record has
    2 wins or 2 losses) and Bo1 otherwise.
  * Round 1: draw tier 1 v tier 4 and tier 2 v tier 3, never a same-league
    pair. The Play-In winner is the 16th team and sits in tier 4. Rounds 2-5:
    a random draw among teams on the same record, no rematches, same-league
    pairs allowed.
  * Quarterfinal draw: each 3-0 team is drawn against one of the three 3-2
    teams, the remaining four teams into two more matches. The 3-0 teams sit
    on opposite halves of the bracket (the 2025 precedent; the 2026 primer is
    silent). Knockout series are Bo5.

Every simulation is vectorized over `n` with numpy. A `Known` state pins
whatever has really happened (or a what-if) -- draws, results, the Play-In
winner -- and everything else is sampled, so one function serves the daily
odds, the live app and a "play out the rest" button.

Draws are modeled as uniform over legal pairings (rejection sampling); Riot
draws sequentially, and the difference is negligible. Games are independent.

Before the field is confirmed the same engine runs on a projected field
(`simulate_and_write`, worlds_field.toml), with pseudo-tiers by rating and no
same-league rule.

Run:
    python monte_carlo.py                 # projected field, 20000 sims
"""
from __future__ import annotations

import argparse
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

import bracket
import calibrate
import paths
from paths import CURRENT_YEAR

FIELD_CONFIG = paths.ROOT / "worlds_field.toml"

SLOTS = 16                 # Swiss field size
DIRECT = 15                # teams that qualify directly; the 16th is the Play-In winner
ROUNDS = 5
RECORDS = [(3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3)]
KO_SERIES = ["QF1", "QF2", "QF3", "QF4", "SF1", "SF2", "F"]
MAX_TRIES = 80             # rejection-sampling retries before a rematch/league clash is allowed


# ------------------------------------------------------------------ data

@dataclass
class Field:
    """The teams in play.

    `teams` holds the 15 direct qualifiers (Swiss slots 0-14, in order) and then
    the Play-In candidates; the Play-In winner takes Swiss slot 15, tier 4.
    """
    teams: list[str]
    rating: np.ndarray
    league: list[str]
    tier: list[int]                 # tier (1-4) of each of the 15 direct teams
    short: list[str]
    playin: dict[str, float]        # candidate -> P(wins the Play-In)
    league_rule: bool = True        # Round 1 forbids same-league pairs
    n_direct: int = DIRECT

    def __post_init__(self) -> None:
        self.rating = np.asarray(self.rating, dtype=float)
        cands = self.teams[self.n_direct:]
        if self.n_direct != DIRECT or len(self.tier) != DIRECT:
            raise ValueError(f"Swiss needs {DIRECT} direct teams plus the Play-In winner")
        if sorted(self.tier) != [1] * 4 + [2] * 4 + [3] * 4 + [4] * 3:
            raise ValueError(f"tiers must be 4/4/4/3 (+ the Play-In winner), got {sorted(self.tier)}")
        if set(self.playin) != set(cands) or abs(sum(self.playin.values()) - 1) > 1e-6:
            raise ValueError("`playin` must give each candidate a probability, summing to 1")
        if not (len(self.teams) == len(self.rating) == len(self.league) == len(self.short)):
            raise ValueError("teams / rating / league / short must be the same length")

    @property
    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}


@dataclass
class Known:
    """What is already decided: real results, or what-if picks."""
    playin_winner: str | None = None
    draws: dict[int, list[tuple[str, str]]] = dc_field(default_factory=dict)
    results: dict[int, dict[tuple[str, str], str]] = dc_field(default_factory=dict)
    qf: list[tuple[str, str]] | None = None
    ko: dict[str, str] = dc_field(default_factory=dict)


@dataclass
class Odds:
    """All arrays are indexed in `Field.teams` order and are probabilities."""
    n: int
    stats: dict
    entered: np.ndarray             # P(plays in the Swiss)
    swiss: np.ndarray               # P(advances to the knockout)
    record: np.ndarray              # [T, 6] final record, columns RECORDS
    semis: np.ndarray               # P(wins a quarterfinal)
    final: np.ndarray
    title: np.ndarray
    occupancy: dict = dc_field(default_factory=dict)   # round -> (w, l) -> [T]
    meet: dict = dc_field(default_factory=dict)        # round -> [T, T] P(a plays b)
    ko_slot: np.ndarray | None = None                  # [T, 7] P(team plays in each KO_SERIES)

    @property
    def fallback(self) -> int:
        return self.stats["fallback"]


@dataclass
class Trace:
    """One fully played-out run (n=1): everything a "play out the rest" needs."""
    playin_winner: str
    draws: dict[int, list[tuple[str, str]]]
    winners: dict[int, list[str]]
    qf: list[tuple[str, str]]
    ko: dict[str, str]


# ------------------------------------------------------------------ probabilities

def p_game(ra: float, rb: float) -> float:
    # Uses the shrinkage fitted in backtest.py rather than a bare Elo curve:
    # calibrated gaps are ~1.32x too wide, and that error compounds across the
    # dozens of games each simulated run plays.
    return calibrate.prob_from_gap(ra - rb)


def _matrices(field: Field) -> dict[int, np.ndarray]:
    """P(row beats column) for a Bo1, Bo3 and Bo5."""
    p1 = calibrate.prob_from_gap(field.rating[:, None] - field.rating[None, :])
    return {1: p1, 3: bracket.p_series(p1, 2), 5: bracket.p_series(p1, 3)}


def match_prob(field: Field, a: str, b: str, best_of: int) -> float:
    """P(team `a` wins a best-of-`best_of` against `b`)."""
    ix = field.index
    return float(_matrices(field)[best_of][ix[a], ix[b]])


def _layout(round_no: int) -> list[tuple[int, int, int]]:
    """(wins, losses, teams) of each record group that plays in `round_no`.

    Group sizes are fixed by the format (8/8, 4/8/4, 6/6, 6), which is what lets
    every simulation be drawn with the same array shapes.
    """
    counts = {(0, 0): SLOTS}
    for _ in range(round_no - 1):
        nxt: dict = defaultdict(int)
        for (w, l), s in counts.items():
            if w < 3 and l < 3:
                nxt[(w + 1, l)] += s // 2
                nxt[(w, l + 1)] += s // 2
            else:
                nxt[(w, l)] += s
        counts = nxt
    return sorted(((w, l, s) for (w, l), s in counts.items() if w < 3 and l < 3),
                  key=lambda g: g[0] * 4 + g[1])


# ------------------------------------------------------------------ draws

def _draw_round1(field: Field, rng, n: int, tiers: dict, lslot: np.ndarray,
                 rcol: np.ndarray, stats: dict):
    """Tier 1 v tier 4 and tier 2 v tier 3, redrawn while a pair shares a league."""
    his, los = [], []
    for hi_t, lo_t in ((1, 4), (2, 3)):
        hi, lo = np.array(tiers[hi_t]), np.array(tiers[lo_t])
        perm = np.argsort(rng.random((n, 4)), axis=1)
        if field.league_rule:
            for _ in range(MAX_TRIES):
                bad = (lslot[:, hi] == lslot[rcol, lo[perm]]).any(axis=1)
                if not bad.any():
                    break
                perm[bad] = np.argsort(rng.random((int(bad.sum()), 4)), axis=1)
            else:
                bad = (lslot[:, hi] == lslot[rcol, lo[perm]]).any(axis=1)
                stats["fallback"] += int(bad.sum())
        his.append(np.broadcast_to(hi, (n, 4)))
        los.append(lo[perm])
    return np.concatenate(his, axis=1), np.concatenate(los, axis=1)


def _draw_by_record(r: int, rng, n: int, wins: np.ndarray, losses: np.ndarray,
                    played: np.ndarray, rcol: np.ndarray, stats: dict):
    """Random pairs inside each record group, redrawn while anyone rematches."""
    active = int(sum(s for _, _, s in _layout(r)))
    code = np.where((wins < 3) & (losses < 3), wins * 4 + losses, 100).astype(float)
    order = np.argsort(code + rng.random((n, SLOTS)), axis=1)[:, :active]
    for _ in range(MAX_TRIES):
        bad = played[rcol, order[:, 0:active:2], order[:, 1:active:2]].any(axis=1)
        if not bad.any():
            break
        nb = int(bad.sum())
        order[bad] = np.argsort(code[bad] + rng.random((nb, SLOTS)), axis=1)[:, :active]
    else:
        bad = played[rcol, order[:, 0:active:2], order[:, 1:active:2]].any(axis=1)
        stats["fallback"] += int(bad.sum())
    return order[:, 0:active:2], order[:, 1:active:2]


def _force(a_wins: np.ndarray, ta: np.ndarray, tb: np.ndarray, winner: int) -> np.ndarray:
    """Pin one column's outcome to `winner` (a team index), wherever it plays."""
    return np.where(ta == winner, True, np.where(tb == winner, False, a_wins))


# ------------------------------------------------------------------ engine

def _execute(field: Field, known: Known, n: int, seed: int | None,
             collect: bool = False) -> tuple[Odds, Trace | None]:
    T = len(field.teams)
    ix = field.index
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(14)]
    rows = np.arange(n)
    rcol = rows[:, None]
    P = _matrices(field)
    stats = {"fallback": 0, "mismatched_records": 0, "bo3_pairs": 0, "bo1_pairs": 0}

    def slot_of(name: str) -> int:
        return min(ix[name], DIRECT)         # every Play-In candidate lands in slot 15

    # Who holds the 16th Swiss slot in each simulation.
    slot_team = np.empty((n, SLOTS), dtype=np.int64)
    slot_team[:, :DIRECT] = np.arange(DIRECT)
    if known.playin_winner is not None:
        slot_team[:, DIRECT] = ix[known.playin_winner]
    else:
        cands = field.teams[DIRECT:]
        cum = np.cumsum([field.playin[c] for c in cands])
        pick = np.searchsorted(cum / cum[-1], streams[0].random(n), side="right")
        slot_team[:, DIRECT] = DIRECT + np.minimum(pick, len(cands) - 1)

    code_of = {lg: i for i, lg in enumerate(sorted(set(field.league)))}
    lslot = np.array([code_of[lg] for lg in field.league])[slot_team]
    tiers = {t: [s for s in range(DIRECT) if field.tier[s] == t] for t in (1, 2, 3, 4)}
    tiers[4].append(DIRECT)

    wins = np.zeros((n, SLOTS), dtype=np.int64)
    losses = np.zeros((n, SLOTS), dtype=np.int64)
    played = np.zeros((n, SLOTS, SLOTS), dtype=bool)
    occupancy = {r: {(w, l): np.zeros(T) for w, l, _ in _layout(r)} for r in range(1, ROUNDS + 1)}
    meet = {r: np.zeros((T, T)) for r in range(1, ROUNDS + 1)}
    tr_draws: dict = {}
    tr_winners: dict = {}

    for r in range(1, ROUNDS + 1):
        for (w, l), acc in occupancy[r].items():
            acc += np.bincount(slot_team[(wins == w) & (losses == l)], minlength=T)

        if r in known.draws:
            fixed = np.array([[slot_of(a), slot_of(b)] for a, b in known.draws[r]])
            pa = np.broadcast_to(fixed[:, 0], (n, len(fixed)))
            pb = np.broadcast_to(fixed[:, 1], (n, len(fixed)))
        elif r == 1:
            pa, pb = _draw_round1(field, streams[1], n, tiers, lslot, rcol, stats)
        else:
            pa, pb = _draw_by_record(r, streams[r], n, wins, losses, played, rcol, stats)
        m = pa.shape[1]

        ta, tb = slot_team[rcol, pa], slot_team[rcol, pb]
        wa, la = wins[rcol, pa], losses[rcol, pa]
        stats["mismatched_records"] += int(((wa != wins[rcol, pb]) | (la != losses[rcol, pb])).sum())
        bo3 = (wa == 2) | (la == 2)
        stats["bo3_pairs"] += int(bo3.sum())
        stats["bo1_pairs"] += int((~bo3).sum())
        a_wins = streams[5 + r].random((n, 8))[:, :m] < np.where(bo3, P[3][ta, tb], P[1][ta, tb])

        if r in known.draws:
            cols = {tuple(p): j for j, p in enumerate(known.draws[r])}
            for (a, b), winner in known.results.get(r, {}).items():
                j = cols.get((a, b), cols.get((b, a)))
                if j is not None:
                    a_wins[:, j] = _force(a_wins[:, j], ta[:, j], tb[:, j], ix[winner])

        win_slot, lose_slot = np.where(a_wins, pa, pb), np.where(a_wins, pb, pa)
        for j in range(m):
            wins[rows, win_slot[:, j]] += 1
            losses[rows, lose_slot[:, j]] += 1
        played[rcol, pa, pb] = True
        played[rcol, pb, pa] = True
        for x, y in ((ta, tb), (tb, ta)):
            meet[r] += np.bincount((x * T + y).ravel(), minlength=T * T).reshape(T, T)
        if collect:
            tr_draws[r] = [(field.teams[ta[0, j]], field.teams[tb[0, j]]) for j in range(m)]
            tr_winners[r] = [field.teams[slot_team[0, win_slot[0, j]]] for j in range(m)]

    # ---- final records
    record = np.zeros((T, 6))
    for i, (w, l) in enumerate(RECORDS):
        record[:, i] = np.bincount(slot_team[(wins == w) & (losses == l)], minlength=T)

    # ---- quarterfinal draw
    if known.qf is not None:
        fixed = np.array([[slot_of(a), slot_of(b)] for a, b in known.qf])
        qa = np.broadcast_to(fixed[:, 0], (n, 4))
        qb = np.broadcast_to(fixed[:, 1], (n, 4))
    else:
        rng = streams[11]
        order = np.argsort(np.where(wins == 3, losses, 9) + rng.random((n, SLOTS)), axis=1)
        g30, g31, g32 = order[:, 0:2], order[:, 2:5], order[:, 5:8]
        rest = np.stack([g31[:, 0], g31[:, 1], g31[:, 2], g32[:, 2]], axis=1)
        rest = np.take_along_axis(rest, np.argsort(rng.random((n, 4)), axis=1), axis=1)
        cA, cB, cR = (rng.random((n, 3)) < 0.5).T
        A = (g30[:, 0], g32[:, 0])                      # each 3-0 v a 3-2
        B = (g30[:, 1], g32[:, 1])
        R0, R1 = (rest[:, 0], rest[:, 1]), (rest[:, 2], rest[:, 3])
        Rx = tuple(np.where(cR, R0[k], R1[k]) for k in (0, 1))   # half 1 with A
        Ry = tuple(np.where(cR, R1[k], R0[k]) for k in (0, 1))   # half 2 with B
        qa = np.stack([np.where(cA, A[0], Rx[0]), np.where(cA, Rx[0], A[0]),
                       np.where(cB, B[0], Ry[0]), np.where(cB, Ry[0], B[0])], axis=1)
        qb = np.stack([np.where(cA, A[1], Rx[1]), np.where(cA, Rx[1], A[1]),
                       np.where(cB, B[1], Ry[1]), np.where(cB, Ry[1], B[1])], axis=1)

    u_ko = streams[12].random((n, 7))
    ko_winner: dict[str, np.ndarray] = {}

    def play(key: str, sa: np.ndarray, sb: np.ndarray, u: np.ndarray) -> np.ndarray:
        ta_, tb_ = slot_team[rows, sa], slot_team[rows, sb]
        a_w = u < P[5][ta_, tb_]
        if key in known.ko:
            a_w = _force(a_w, ta_, tb_, ix[known.ko[key]])
        win = np.where(a_w, sa, sb)
        ko_winner[key] = slot_team[rows, win]
        return win

    qw = [play(f"QF{i + 1}", qa[:, i], qb[:, i], u_ko[:, i]) for i in range(4)]
    sw = [play("SF1", qw[0], qw[1], u_ko[:, 4]), play("SF2", qw[2], qw[3], u_ko[:, 5])]
    champ = play("F", sw[0], sw[1], u_ko[:, 6])

    def bc(slots: np.ndarray) -> np.ndarray:
        return np.bincount(slot_team[rows, slots], minlength=T)

    ko_slot = np.zeros((T, len(KO_SERIES)))
    for j, (sa, sb) in enumerate([(qa[:, 0], qb[:, 0]), (qa[:, 1], qb[:, 1]), (qa[:, 2], qb[:, 2]),
                                  (qa[:, 3], qb[:, 3]), (qw[0], qw[1]), (qw[2], qw[3]), (sw[0], sw[1])]):
        ko_slot[:, j] = bc(sa) + bc(sb)

    odds = Odds(
        n=n, stats=stats,
        entered=np.bincount(slot_team.ravel(), minlength=T) / n,
        swiss=record[:, :3].sum(axis=1) / n,
        record=record / n,
        semis=sum(bc(w) for w in qw) / n,
        final=sum(bc(w) for w in sw) / n,
        title=bc(champ) / n,
        occupancy={r: {k: v / n for k, v in d.items()} for r, d in occupancy.items()},
        meet={r: m_ / n for r, m_ in meet.items()},
        ko_slot=ko_slot / n,
    )
    trace = None
    if collect:
        names = field.teams
        trace = Trace(
            playin_winner=names[slot_team[0, DIRECT]], draws=tr_draws, winners=tr_winners,
            qf=[(names[slot_team[0, qa[0, i]]], names[slot_team[0, qb[0, i]]]) for i in range(4)],
            ko={k: names[v[0]] for k, v in ko_winner.items()},
        )
    return odds, trace


# ------------------------------------------------------------------ public API

def simulate(field: Field, known: Known | None = None, n: int = 20000,
             seed: int | None = 0) -> Odds:
    """Odds for every team, given what is already `known`."""
    return _execute(field, known or Known(), n, seed)[0]


def run_trace(field: Field, known: Known | None = None, seed: int | None = None) -> Trace:
    """Play one whole tournament out, honoring `known`. Drives "play out the rest"."""
    return _execute(field, known or Known(), 1, seed, collect=True)[1]


def _seed_from(rng: np.random.Generator) -> int:
    return int(rng.integers(2**31 - 1))


def next_undrawn(known: Known) -> int | None:
    """The first Swiss round whose draw is not known yet."""
    return next((r for r in range(1, ROUNDS + 1) if r not in known.draws), None)


def sample_round(field: Field, known: Known, rng: np.random.Generator) -> list[tuple[str, str]]:
    """One legal draw for the next undrawn round (earlier unknowns are played out)."""
    r = next_undrawn(known)
    if r is None:
        raise ValueError("every Swiss round is already drawn")
    return run_trace(field, known, _seed_from(rng)).draws[r]


def sample_quarterfinals(field: Field, known: Known, rng: np.random.Generator) -> list[tuple[str, str]]:
    return run_trace(field, known, _seed_from(rng)).qf


# ------------------------------------------------------------------ output table

def odds_table(field: Field, odds: Odds, kind: str, calibrated: dict[str, float] | None = None) -> pd.DataFrame:
    """The title-odds table: one row per team, probabilities in percent."""
    cal = calibrated or dict(zip(field.teams, field.rating))
    df = pd.DataFrame({
        "team": field.teams, "home_league": field.league,
        "calibrated": [round(cal[t], 1) for t in field.teams],
        "swiss_%": 100 * odds.entered, "knockout_%": 100 * odds.swiss,
        "semis_%": 100 * odds.semis, "finals_%": 100 * odds.final, "title_%": 100 * odds.title,
    })
    for i, (w, l) in enumerate(RECORDS):
        df[f"{w}-{l}_%"] = 100 * odds.record[:, i]
    pct = [c for c in df.columns if c.endswith("_%")]
    df[pct] = df[pct].round(2)
    df["field"] = kind
    return df.sort_values(["title_%", "knockout_%"], ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ projected field

def load_field_config(path=FIELD_CONFIG) -> tuple[dict[str, int], list[dict], int]:
    """Read the projected-field assumptions from worlds_field.toml.

    Kept in config rather than code because the qualification picture changes
    throughout the season and these are guesses, not results.
    """
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    slots = cfg["slots"]
    guaranteed = cfg.get("guaranteed", [])
    size = cfg.get("size", sum(slots.values()))
    if sum(slots.values()) != size:
        raise ValueError(f"{path.name}: slots sum to {sum(slots.values())}, expected size={size}")
    return slots, guaranteed, size


def build_field(df: pd.DataFrame, slots: dict[str, int],
                guaranteed: list[dict] | None = None) -> pd.DataFrame:
    picks = []
    for league, n in slots.items():
        pool = df[df["home_league"] == league].sort_values("calibrated", ascending=False)
        picks.append(pool.head(n))
    field = pd.concat(picks)
    # make sure guaranteed teams are present (swap in if a league cut them)
    for g in guaranteed or []:
        name = g["team"]
        if name not in field["team"].values and name in df["team"].values:
            field = pd.concat([field, df[df["team"] == name]])
    field = field.drop_duplicates("team").reset_index(drop=True)

    bad = field[field["calibrated"].isna()]
    if len(bad):
        raise ValueError(f"Teams with no calibrated rating in field: "
                         f"{bad['team'].tolist()}")
    return field


def projected_field(picked: pd.DataFrame) -> Field:
    """Pseudo-tiers by rating; the weakest team stands in for the Play-In winner."""
    if len(picked) != SLOTS:
        raise ValueError(f"projected field has {len(picked)} teams, expected {SLOTS}")
    p = picked.sort_values("calibrated", ascending=False).reset_index(drop=True)
    teams = p["team"].tolist()
    return Field(teams=teams, rating=p["calibrated"].to_numpy(), league=p["home_league"].tolist(),
                 tier=[1] * 4 + [2] * 4 + [3] * 4 + [4] * 3, short=[t[:4] for t in teams],
                 playin={teams[-1]: 1.0}, league_rule=False)


def simulate_and_write(year: int = CURRENT_YEAR, n: int = 20000,
                       seed: int | None = 0) -> pd.DataFrame:
    """Simulate the PROJECTED field and write title odds. Returns the table.

    `seed` is fixed by default: the scheduler reruns this on every data change,
    and unseeded RNG would make odds jitter between runs even when nothing about
    the ratings moved, polluting the history table with noise.
    """
    slots, guaranteed, _ = load_field_config()
    df = pd.read_parquet(paths.processed(year, "calibrated_ratings"))
    picked = build_field(df, slots, guaranteed)
    field = projected_field(picked)

    locked = ", ".join(g["team"] for g in guaranteed) or "none"
    print(f"Projected {len(field.teams)}-team field "
          f"({', '.join(f'{k}:{v}' for k, v in slots.items())}); guaranteed: {locked}")
    print(f"Running {n:,} simulations...\n")

    res = odds_table(field, simulate(field, n=n, seed=seed), "projected")
    paths.ensure_dirs()
    res.to_csv(paths.processed(year, "worlds_title_odds", "csv"), index=False)
    print(f"=== WORLDS {year} — PROJECTED TITLE ODDS ===")
    print(res.to_string(index=False))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--n", type=int, default=20000, help="number of simulations")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducible odds (use -1 for random)")
    args = ap.parse_args()
    simulate_and_write(args.year, args.n, None if args.seed == -1 else args.seed)


if __name__ == "__main__":
    main()
