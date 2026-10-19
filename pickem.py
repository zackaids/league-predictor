"""Worlds Swiss Pick'Em: the picks most likely to be right, from simulated runs.

`monte_carlo.simulate(keep_runs=True)` keeps every run's final record, so a pick
set can be scored against whole simulated Swiss stages, not just per-team odds.
Two objectives:

  * Most expected points. Scoring is additive, so a pick set's expected score is
    the sum of each pick's own probability and only the per-team marginals
    matter. That is an assignment problem, solved exactly (DP over teams).
  * Best chance of at least k points. Picks are correlated -- two favorites who
    meet at 1-0 cannot both go 3-0 -- so this needs the joint outcomes. It is a
    local search over pick sets, scored on the simulated runs.

The search can over-fit noise (a rare 8/8 hit is a few dozen runs), so it runs on
one set of simulations and every reported number comes from an independent
hold-out set (seed + 1).

Ratings can be adjusted to see how much a pick leans on one team's number:
`--rating "Karmine Corp=1708"` pins a rating, `--no-playin` rates every team as if
the Play-In had not been played (see `playin_free_ratings`).

Run:
    python pickem.py                  # 2026: picks, hold-out odds, CSV
    python pickem.py --n 100000 --k 5
    python pickem.py --no-playin      # the picks if Play-In results counted for nothing
"""
from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import bracket
import calibrate
import elo
import monte_carlo as mc
import paths
import swiss
from paths import CURRENT_YEAR

CONFIG = paths.ROOT / "pickem.toml"
LABELS = [f"{w}-{l}" for w, l in mc.RECORDS]
RESTARTS = 6           # random starts for the threshold search, besides the expected-points optimum


# ------------------------------------------------------------------ config

@dataclass
class Config:
    positions: list[str]                 # one entry per pick, slot name each
    counts: dict[str, int]
    points: dict[str, int]
    groups: dict[str, tuple[int, ...]]   # slot -> indices into mc.RECORDS that score it
    distinct: bool

    @property
    def names(self) -> list[str]:
        return list(self.counts)

    @property
    def max_points(self) -> int:
        return sum(self.points[s] for s in self.positions)


def load_config(path=CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    counts = {s: int(c) for s, c in raw["slots"].items()}
    points = {s: int(raw["points"][s]) for s in counts}
    advance = tuple(LABELS.index(r) for r in raw["rules"]["advance"])
    groups = {}
    for s in counts:
        if s == "advance":
            groups[s] = advance
        elif s in LABELS:
            groups[s] = (LABELS.index(s),)
        else:
            raise ValueError(f"slot {s!r} is neither a record ({', '.join(LABELS)}) nor 'advance'")
    positions = [s for s, c in counts.items() for _ in range(c)]
    return Config(positions, counts, points, groups, bool(raw["rules"].get("distinct", True)))


# ------------------------------------------------------------------ scoring

def hits(runs: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    """slot -> bool [T, n]: would team t score a pick in this slot in run i."""
    return {s: np.ascontiguousarray(np.isin(runs, g).T) for s, g in cfg.groups.items()}


def score(h: dict[str, np.ndarray], cfg: Config, teams: list[int]) -> np.ndarray:
    """Points each run pays this pick set (`teams[i]` fills `cfg.positions[i]`)."""
    n = next(iter(h.values())).shape[1]
    out = np.zeros(n, dtype=np.int16)
    for s, t in zip(cfg.positions, teams):
        out += cfg.points[s] * h[s][t]
    return out


def marginals(h: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {s: a.mean(axis=1) for s, a in h.items()}


# ------------------------------------------------------------------ objective 1: expected points

def best_expected(marg: dict[str, np.ndarray], cfg: Config, eligible: list[int]) -> list[int]:
    """The pick set with the highest expected points, exactly.

    With `distinct`, a DP over teams whose state is how many picks each slot still
    needs (48 states here); otherwise each slot takes its own top teams.
    """
    names = cfg.names
    if not cfg.distinct:
        pools = {s: sorted(eligible, key=lambda t: -marg[s][t])[:cfg.counts[s]] for s in names}
    else:
        cur = {tuple(cfg.counts[s] for s in names): 0.0}
        layers = []
        for t in eligible:
            nxt, back = {}, {}
            for st, v in cur.items():
                if nxt.get(st, -1.0) < v:
                    nxt[st], back[st] = v, (st, None)
                for j, s in enumerate(names):
                    if st[j]:
                        st2 = st[:j] + (st[j] - 1,) + st[j + 1:]
                        v2 = v + cfg.points[s] * marg[s][t]
                        if nxt.get(st2, -1.0) < v2:
                            nxt[st2], back[st2] = v2, (st, j)
            layers.append(back)
            cur = nxt
        pools = {s: [] for s in names}
        st = (0,) * len(names)
        for t, back in zip(reversed(eligible), reversed(layers)):
            st, j = back[st]
            if j is not None:
                pools[names[j]].append(t)
        for s in names:
            pools[s].sort(key=lambda t: -marg[s][t])
    return [pools[s].pop(0) for s in cfg.positions]


# ------------------------------------------------------------------ objective 2: P(at least k)

def _valid(cfg: Config, teams: list[int]) -> bool:
    if cfg.distinct:
        return len(set(teams)) == len(teams)
    seen = set()
    for s, t in zip(cfg.positions, teams):
        if (s, t) in seen:
            return False
        seen.add((s, t))
    return True


def _search(h: dict[str, np.ndarray], cfg: Config, k: int, eligible: list[int],
            start: list[int]) -> tuple[list[int], tuple[int, float]]:
    """Steepest-ascent on (#runs with >= k points, expected points)."""
    n = next(iter(h.values())).shape[1]
    pos = cfg.positions
    pts = [cfg.points[s] for s in pos]
    hsum = {s: a.sum(axis=1) for s, a in h.items()}
    teams = list(start)

    def total(ts):
        c = score(h, cfg, ts)
        return int((c >= k).sum()), float(c.sum()) / n

    cur = total(teams)
    while True:
        count = score(h, cfg, teams)
        best, best_val = None, cur
        for i, s in enumerate(pos):                              # replace one pick
            rest = count - pts[i] * h[s][teams[i]]
            base = int((rest >= k).sum())
            gain = h[s][:, (rest >= k - pts[i]) & (rest < k)].sum(axis=1)
            erest = float(rest.sum()) / n
            for u in eligible:
                if u == teams[i]:
                    continue
                cand = teams[:i] + [u] + teams[i + 1:]
                if not _valid(cfg, cand):
                    continue
                val = (base + int(gain[u]), erest + pts[i] * float(hsum[s][u]) / n)
                if val > best_val:
                    best, best_val = cand, val
        for i in range(len(pos)):                                # swap two picks' teams
            for j in range(i + 1, len(pos)):
                if pos[i] == pos[j]:
                    continue
                cand = list(teams)
                cand[i], cand[j] = teams[j], teams[i]
                if _valid(cfg, cand):
                    val = total(cand)
                    if val > best_val:
                        best, best_val = cand, val
        if best is None:
            return teams, cur
        teams, cur = best, best_val


def best_threshold(h: dict[str, np.ndarray], cfg: Config, k: int, eligible: list[int],
                   start: list[int], rng: np.random.Generator, restarts: int = RESTARTS) -> list[int]:
    """The pick set with the best chance of at least `k` points on these runs."""
    starts = [start]
    for _ in range(restarts):
        cand = list(rng.permutation(eligible))
        starts.append(cand[:len(cfg.positions)])
    best = max((_search(h, cfg, k, eligible, s) for s in starts if _valid(cfg, s)),
               key=lambda r: r[1])
    return best[0]


# ------------------------------------------------------------------ analysis

@dataclass
class Picks:
    slots: list[str]
    teams: list[int]            # Field.teams index of each pick
    p_correct: list[float]      # P(each pick scores)
    dist: np.ndarray            # hold-out P(points == 0 .. max)

    @property
    def expected(self) -> float:
        return float((self.dist * np.arange(len(self.dist))).sum())

    def p_at_least(self, k: int) -> float:
        return float(self.dist[k:].sum())


@dataclass
class Analysis:
    cfg: Config
    field: mc.Field
    n: int
    entered: np.ndarray
    record: np.ndarray                  # [T, 6] P(final record), both sets pooled
    marg: dict[str, np.ndarray]         # slot -> [T] P(scores)
    expected: Picks
    best: dict[int, Picks]              # k -> picks maximizing P(>= k)


def simulate(field: mc.Field, known: mc.Known, n: int, seed: int = 0) -> tuple[mc.Odds, mc.Odds]:
    """Search runs and independent hold-out runs."""
    return (mc.simulate(field, known, n, seed, keep_runs=True),
            mc.simulate(field, known, n, seed + 1, keep_runs=True))


def analyze(field: mc.Field, train: mc.Odds, hold: mc.Odds, cfg: Config,
            ks: list[int], seed: int = 0) -> Analysis:
    """Optimal picks for expected points and for each threshold in `ks`.

    The expected-points optimum uses the two sets pooled (it needs only 16 x 4
    averages, which cannot over-fit); the threshold search sees only `train`; the
    reported distributions are `hold`'s alone.
    """
    h_tr, h_ho = hits(train.runs, cfg), hits(hold.runs, cfg)
    marg = {s: (h_tr[s].mean(axis=1) + h_ho[s].mean(axis=1)) / 2 for s in cfg.names}
    entered = (train.entered + hold.entered) / 2
    eligible = [t for t in range(len(field.teams)) if entered[t] > 0]
    if len(eligible) < len(cfg.positions) and cfg.distinct:
        raise ValueError("fewer eligible teams than picks")

    def picks(teams: list[int]) -> Picks:
        c = score(h_ho, cfg, teams)
        dist = np.bincount(c, minlength=cfg.max_points + 1) / c.size
        return Picks(list(cfg.positions), teams, [float(marg[s][t]) for s, t in zip(cfg.positions, teams)], dist)

    exp_teams = best_expected(marg, cfg, eligible)
    rng = np.random.default_rng(seed)
    best = {k: picks(best_threshold(h_tr, cfg, k, eligible, exp_teams, rng))
            for k in ks if 1 <= k <= cfg.max_points}
    return Analysis(cfg, field, train.runs.shape[0], entered,
                    (train.record + hold.record) / 2, marg, picks(exp_teams), best)


# ------------------------------------------------------------------ rating what-ifs

def with_ratings(field: mc.Field, ratings: dict[str, float]) -> mc.Field:
    """A copy of `field` with some teams' ratings replaced."""
    rating = field.rating.copy()
    for team, r in ratings.items():
        rating[field.index[team]] = r
    return replace(field, rating=rating)


def playin_free_ratings(year: int, rows: pd.DataFrame) -> dict[str, float]:
    """Calibrated ratings as if the Play-In had not been played.

    One Elo pass over everything except the Play-In's games (the same pass and
    calibration the pipeline runs), so every team's league mean and the region
    prior move exactly as they would. Needs the local team-games table, which is
    gitignored, so this is a CLI tool and not an app feature.
    """
    tg = pd.read_parquet(paths.processed(year, "team_games"))
    tg["date"] = pd.to_datetime(tg["date"])
    region = calibrate.load_region_prior()
    homes = calibrate.home_league(tg, region, year)
    start = pd.Timestamp(bracket.load_config()["start"])
    keep = tg["league"].isin(elo.MAJOR_SCOPE) & ~((tg["league"] == "WLDs") & (tg["date"] >= start))
    table, _ = elo.replay(elo.build_matches(tg[keep]), elo.K_DEFAULT)
    cal = calibrate.combine(homes, table, region).set_index("team")["calibrated"]
    return {t: float(cal[d]) for t, d in zip(rows["team"], rows["data_name"]) if d in cal.index}


def parse_ratings(items: list[str], field: mc.Field) -> dict[str, float]:
    """`--rating` values like "Karmine Corp=1708" or "KC=1708"."""
    by_name = {**{t: t for t in field.teams}, **dict(zip(field.short, field.teams))}
    out = {}
    for item in items:
        name, _, value = item.rpartition("=")
        if name not in by_name or not value:
            raise SystemExit(f"--rating wants TEAM=VALUE with a team in {sorted(by_name)}, got {item!r}")
        out[by_name[name]] = float(value)
    return out


# ------------------------------------------------------------------ output

def team_table(a: Analysis) -> pd.DataFrame:
    """One row per team in the Swiss: record odds, P(scores) per slot, and its pick."""
    f, cfg = a.field, a.cfg
    chosen = {t: s for s, t in zip(a.expected.slots, a.expected.teams)}
    rows = []
    for t in range(len(f.teams)):
        if a.entered[t] <= 0:
            continue
        row = {"team": f.teams[t], "short": f.short[t]}
        row.update({f"{lab}_%": round(100 * a.record[t, i], 2) for i, lab in enumerate(LABELS)})
        row.update({f"{s}_%": round(100 * a.marg[s][t], 2) for s in cfg.names if s not in LABELS})
        row["expected_pick"] = chosen.get(t, "")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("advance_%" if "advance" in cfg.counts else "3-0_%",
                                          ascending=False, ignore_index=True)


def describe(a: Analysis, p: Picks) -> str:
    sh = a.field.short
    return "  ".join(f"{s}:{sh[t]}" for s, t in zip(p.slots, p.teams))


def run(year: int = CURRENT_YEAR, n: int = 200_000, seed: int = 0,
        ks: list[int] | None = None, ratings: dict[str, float] | None = None,
        no_playin: bool = False) -> Analysis:
    cfg = load_config()
    ev = swiss.resolve(swiss.load_config(), swiss.load_series(year), year)
    k = ev.known
    field = ev.field
    if no_playin:
        ratings = {**playin_free_ratings(year, ev.rows), **(ratings or {})}
    if ratings:
        changed = {t: r for t, r in ratings.items() if abs(r - field.rating[field.index[t]]) >= 0.5}
        print("Rating adjustments: " + ", ".join(
            f"{t} {field.rating[field.index[t]]:.0f} -> {r:.0f}" for t, r in changed.items()))
        field = with_ratings(field, ratings)
    print(f"=== WORLDS {year} -- SWISS PICK'EM ({n:,} search sims + {n:,} hold-out) ===")
    print(f"Play-In winner: {k.playin_winner or 'undecided'}; rounds drawn: {sorted(k.draws) or 'none'}; "
          f"results entered: {sum(len(v) for v in k.results.values())}\n")

    train, hold = simulate(field, k, n, seed)
    ks = ks or list(range(1, cfg.max_points + 1))
    a = analyze(field, train, hold, cfg, ks, seed)

    table = team_table(a)
    print("Final-record odds and pick value (%):")
    print(table.to_string(index=False))

    e = a.expected
    print("\nMost expected points -- exact optimum:")
    for s, t, p in zip(e.slots, e.teams, e.p_correct):
        print(f"  {s:8s} {a.field.teams[t]:22s} {100 * p:5.1f}%")
    print(f"  expected points {e.expected:.2f} of {cfg.max_points}; hold-out distribution: "
          + "  ".join(f"{i}:{100 * x:.1f}%" for i, x in enumerate(e.dist)))

    print("\nBest chance of at least k points (picks found on the search runs, odds from hold-out):")
    print(f"  {'k':>2}  {'P(>=k)':>7}  {'expected set':>12}   picks")
    for kk, p in a.best.items():
        print(f"  {kk:>2}  {100 * p.p_at_least(kk):6.2f}%  {100 * e.p_at_least(kk):11.2f}%   {describe(a, p)}")
    se = np.sqrt(0.25 / n) * 100
    print(f"\n(hold-out standard error is at most {se:.2f} points; small P(>=k) are noisier in relative terms)")

    paths.ensure_dirs()
    out = paths.processed(year, "pickem_adjusted" if ratings else "pickem", "csv")
    table.to_csv(out, index=False)
    print(f"Wrote {out}")
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--n", type=int, default=200_000, help="simulations per set (search and hold-out)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, nargs="*", help="thresholds to optimize (default: every k)")
    ap.add_argument("--rating", nargs="*", default=[], metavar="TEAM=VALUE",
                    help='pin calibrated ratings, e.g. --rating "Karmine Corp=1708" KC=1700')
    ap.add_argument("--no-playin", action="store_true",
                    help="rate every team as if the Play-In results did not count")
    args = ap.parse_args()
    ratings = None
    if args.rating:
        ev = swiss.resolve(swiss.load_config(), swiss.load_series(args.year), args.year)
        ratings = parse_ratings(args.rating, ev.field)
    run(args.year, args.n, args.seed, args.k, ratings, args.no_playin)


if __name__ == "__main__":
    main()
