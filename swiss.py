"""Worlds 2026 Swiss stage + knockout: the confirmed event, as facts and odds.

`bracket.py` does this for the Play-In; this does it for what follows. The
engine (`monte_carlo.py`) knows the format; this module knows the *event*:

  * `worlds_swiss.toml` -- teams and tiers, each round's draw as Riot announces
    it (keyed by record), the quarterfinal draw, optional manual results.
  * results -- read from the series log (league `WLDs`, inside the stage's
    dates) the way `bracket.resolve_results` does, with a [results] entry
    overriding. Results Oracle's Elixir has not published yet are inferred from
    the next round's draw (a team's record there says whether it won), or from
    the quarterfinal draw.
  * the Play-In winner -- from the Play-In page's results; once Round 1 is
    drawn it is read off that draw, because the draw is made after the final.

Ratings are the calibrated Elo from calibrate.py, unchanged.

Run:
    python swiss.py        # write {year}_swiss_field.csv and {year}_worlds_title_odds.csv
"""
from __future__ import annotations

import argparse
import tomllib
from collections import Counter
from dataclasses import dataclass, field as dc_field, replace

import numpy as np
import pandas as pd

import bracket
import monte_carlo as mc
import paths
from paths import CURRENT_YEAR

CONFIG = paths.ROOT / "worlds_swiss.toml"
WORLDS_LEAGUE = bracket.WORLDS_LEAGUE
SERIES_COLS = ["series_id", "date", "league", "team_a", "team_b", "wins_a", "wins_b"]


# ------------------------------------------------------------------ config

def _record(label: str) -> tuple[int, int]:
    try:
        w, l = (int(x) for x in label.split("-"))
    except ValueError:
        raise ValueError(f"bad record label {label!r} (expected like \"1-0\")") from None
    if not (0 <= w <= 2 and 0 <= l <= 2):
        raise ValueError(f"record {label!r} cannot be playing a Swiss round")
    return w, l


def load_config(path=CONFIG) -> dict:
    """Read and structurally validate worlds_swiss.toml.

    Draw rules that need ratings or leagues (tiers, same-league pairs, records
    against results) are checked in `resolve`.
    """
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    teams = cfg["teams"]
    names = [t["name"] for t in teams]
    if len(set(names)) != mc.DIRECT or len(set(t["short"] for t in teams)) != mc.DIRECT:
        raise ValueError(f"{path.name}: expected {mc.DIRECT} distinct teams (names and tricodes)")
    if sorted(t["tier"] for t in teams) != [1] * 4 + [2] * 4 + [3] * 4 + [4] * 3:
        raise ValueError(f"{path.name}: tiers must be 4/4/4/3 -- the Play-In winner is the fourth tier-4 team")

    known = set(names) | {t["name"] for t in bracket.load_config()["teams"]}
    draws: dict[int, list[tuple[str, str]]] = {}
    labels: dict[int, dict[str, tuple[int, int]]] = {}
    for rkey, groups in (cfg.get("draws") or {}).items():
        r = int(rkey)
        if not 1 <= r <= mc.ROUNDS:
            raise ValueError(f"{path.name}: [draws.{rkey}] is not a Swiss round (1-{mc.ROUNDS})")
        draws[r], labels[r] = [], {}
        for label, plist in groups.items():
            rec = _record(label)
            for p in plist:
                if len(p) != 2 or p[0] == p[1]:
                    raise ValueError(f"{path.name}: round {r} {label}: {p!r} is not a pair of two teams")
                for t in p:
                    if t not in known:
                        raise ValueError(f"{path.name}: round {r}: unknown team {t!r}")
                    if t in labels[r]:
                        raise ValueError(f"{path.name}: round {r}: {t!r} is drawn twice")
                    labels[r][t] = rec
                draws[r].append((p[0], p[1]))
    if sorted(draws) != list(range(1, len(draws) + 1)):
        raise ValueError(f"{path.name}: draws must be rounds 1..k with no gaps, got {sorted(draws)}")
    for r, pairs in draws.items():
        if len(pairs) != sum(s for _, _, s in mc._layout(r)) // 2:
            raise ValueError(f"{path.name}: round {r} has {len(pairs)} matches, "
                             f"expected {sum(s for _, _, s in mc._layout(r)) // 2}")

    qf = (cfg.get("knockout") or {}).get("quarterfinals")
    if qf is not None:
        qf = [tuple(p) for p in qf]
        flat = [t for p in qf for t in p]
        if len(qf) != 4 or any(len(p) != 2 for p in qf) or len(set(flat)) != 8 or set(flat) - known:
            raise ValueError(f"{path.name}: quarterfinals must be 4 pairs of 8 distinct known teams")

    res = cfg.get("results") or {}
    results_swiss = {}
    for key, val in (res.get("swiss") or {}).items():
        a, sep, b = key.partition(" v ")
        if not sep or a not in known or b not in known:
            raise ValueError(f"{path.name}: [results.swiss] key {key!r} must be \"Team A v Team B\"")
        results_swiss[frozenset((a, b))] = val
    results_ko = dict(res.get("knockout") or {})
    if set(results_ko) - set(mc.KO_SERIES):
        raise ValueError(f"{path.name}: unknown knockout series in [results.knockout]: "
                         f"{sorted(set(results_ko) - set(mc.KO_SERIES))}")

    cfg["draws"], cfg["labels"], cfg["quarterfinals"] = draws, labels, qf
    cfg["results_swiss"], cfg["results_ko"] = results_swiss, results_ko
    return cfg


def is_current(year: int, path=CONFIG) -> bool:
    """True when the Swiss config is for `year`, so its odds replace the projection."""
    return path.exists() and load_config(path)["start"].year == year


# ------------------------------------------------------------------ field

def field_rows(cfg: dict, bcfg: dict, cal: pd.DataFrame) -> pd.DataFrame:
    """One row per team: the 15 direct qualifiers, then the Play-In candidates."""
    rows = []
    for role, teams in (("swiss", cfg["teams"]), ("playin", bcfg["teams"])):
        for t in teams:
            data_name = t.get("data_name", t["name"])
            rating, league = bracket.team_rating(data_name, cal)
            rows.append({"team": t["name"], "short": t["short"], "data_name": data_name,
                         "tier": t.get("tier", 4), "role": role,
                         "home_league": league, "rating": rating})
    return pd.DataFrame(rows)


def field_from_rows(rows: pd.DataFrame, playin_probs: dict[str, float]) -> mc.Field:
    """The engine's Field from `field_rows` / {year}_swiss_field.csv."""
    return mc.Field(
        teams=rows["team"].tolist(), rating=rows["rating"].to_numpy(),
        league=rows["home_league"].tolist(),
        tier=rows.loc[rows["role"] == "swiss", "tier"].astype(int).tolist(),
        short=rows["short"].tolist(), playin=playin_probs)


def playin_probs(bcfg: dict, rating: dict[str, float], fixed: dict[str, str],
                 winner: str | None = None) -> dict[str, float]:
    """P(each Play-In team takes the 16th Swiss slot)."""
    cands = [t["name"] for t in bcfg["teams"]]
    if winner:
        return {c: float(c == winner) for c in cands}
    finish = bracket.solve(bcfg["opening"], rating, fixed)["finish"]
    probs = {c: finish[c][bracket.SWISS] for c in cands}
    total = sum(probs.values())
    return {c: p / total for c, p in probs.items()}


def build_field(cfg: dict, bcfg: dict, probs: dict[str, float], year: int = CURRENT_YEAR) -> mc.Field:
    cal = pd.read_parquet(paths.processed(year, "calibrated_ratings"))
    return field_from_rows(field_rows(cfg, bcfg, cal), probs)


# ------------------------------------------------------------------ results

@dataclass
class Event:
    """The confirmed facts, as of the series log and the config."""
    cfg: dict
    bcfg: dict
    rows: pd.DataFrame
    field: mc.Field
    known: mc.Known
    info: dict = dc_field(default_factory=dict)    # display detail: winner/score/source per series
    notes: list[str] = dc_field(default_factory=list)


@dataclass
class Live:
    """Facts from the lolesports.com schedule (lolesports.live_facts), merged under the config."""
    draws: dict[int, list[tuple[str, str]]] = dc_field(default_factory=dict)        # complete rounds only
    labels: dict[int, dict[str, tuple[int, int]]] = dc_field(default_factory=dict)  # record going in
    results: dict[frozenset, dict] = dc_field(default_factory=dict)                 # Swiss pair -> winner/score
    qf: list[tuple[str, str]] | None = None                                         # in start-time order
    ko: dict[frozenset, dict] = dc_field(default_factory=dict)                      # knockout pair -> winner/score


def _check_round(r: int, pairs, lab: dict, rec: dict | None, met: set,
                 tier: dict, league: dict, league_rule: bool) -> None:
    sizes = Counter(lab.values())
    for w, l, size in mc._layout(r):
        if sizes.get((w, l), 0) != size:
            raise ValueError(f"round {r}: expected {size} teams on {w}-{l}, "
                             f"the draw has {sizes.get((w, l), 0)}")
    for a, b in pairs:
        if frozenset((a, b)) in met:
            raise ValueError(f"round {r}: {a} v {b} is a rematch")
        if rec is not None and (rec[a] != lab[a] or rec[b] != lab[b]):
            raise ValueError(f"round {r}: {a} v {b} is drawn on {lab[a][0]}-{lab[a][1]}, but their "
                             f"results put them on {rec[a][0]}-{rec[a][1]} and {rec[b][0]}-{rec[b][1]} (record)")
        if r == 1:
            ta, tb = tier[a], tier[b]
            if {ta, tb} not in ({1, 4}, {2, 3}):
                raise ValueError(f"round 1: {a} (tier {ta}) v {b} (tier {tb}) breaks the "
                                 f"tier 1 v 4 / tier 2 v 3 rule")
            if league_rule and league[a] == league[b]:
                raise ValueError(f"round 1: {a} v {b} are both from {league[a]} (same-league rule)")


def _won(team: str, r: int, labels: dict, qf_teams: set | None) -> bool | None:
    """Did `team` win its round-`r` match? None when the draws cannot say."""
    nxt = labels.get(r + 1)
    if nxt is not None and team in nxt:
        return nxt[team][0] == labels[r][team][0] + 1
    if qf_teams is not None:        # a team that finished its Swiss is in the QF iff it won 3
        return team in qf_teams
    return None


def check_quarterfinals(qf: list[tuple[str, str]], rec: dict[str, tuple[int, int]]) -> None:
    """Raise ValueError unless `qf` is a legal quarterfinal draw for the Swiss records `rec`."""
    flat = [t for p in qf for t in p]
    if len(qf) != 4 or any(len(p) != 2 for p in qf) or len(set(flat)) != 8:
        raise ValueError("the quarterfinals must be 4 pairs of 8 distinct teams")
    qf_teams = set(flat)
    wins3 = {t for t, (w, _) in rec.items() if w == 3}
    if wins3 != qf_teams:
        raise ValueError(f"the quarterfinal teams {sorted(qf_teams)} are not the eight teams on 3 wins")
    half = {t: i // 2 for i, p in enumerate(qf) for t in p}
    zero = [t for t in wins3 if rec[t][1] == 0]
    if len({half[t] for t in zero}) != 2:
        raise ValueError("the 3-0 teams must be on opposite halves of the bracket")
    for t in zero:
        opp = next(set(p) - {t} for p in qf if t in p).pop()
        if rec[opp][1] != 2:
            raise ValueError(f"3-0 team {t} must face a 3-2 team, not {opp}")


def resolve(cfg: dict, series: pd.DataFrame, year: int = CURRENT_YEAR,
            rows: pd.DataFrame | None = None) -> Event:
    """Everything that has really happened, from the series log plus the config.

    `rows` is {year}_swiss_field.csv. The app passes it so the page renders from
    committed output, as the Play-In page always did, rather than from live
    ratings; the pipeline builds it fresh.
    """
    bcfg = bracket.load_config()
    if rows is None:
        rows = field_rows(cfg, bcfg, pd.read_parquet(paths.processed(year, "calibrated_ratings")))
    rating = dict(zip(rows["team"], rows["rating"]))
    notes: list[str] = []

    if series.empty:
        series = pd.DataFrame(columns=SERIES_COLS)
    real = bracket.resolve_results(bcfg, series)
    cands = [t["name"] for t in bcfg["teams"]]
    winner = real["QF"]["winner"] if "QF" in real else None
    draws, labels = cfg["draws"], cfg["labels"]
    if 1 in draws:
        in_r1 = [t for p in draws[1] for t in p if t in cands]
        if len(in_r1) != 1:
            raise ValueError(f"round 1 must include exactly one Play-In team, found {in_r1}")
        if winner is not None and winner != in_r1[0]:
            raise ValueError(f"the Play-In result says {winner} advanced but the Round 1 draw has {in_r1[0]}")
        if winner is None:
            notes.append(f"Play-In winner taken from the Round 1 draw: {in_r1[0]}.")
        winner = in_r1[0]
    probs = playin_probs(bcfg, rating, {m: r["winner"] for m, r in real.items()}, winner)
    field = field_from_rows(rows, probs)

    tier = dict(zip(rows["team"], rows["tier"]))
    league = dict(zip(rows["team"], rows["home_league"]))
    display = dict(zip(rows["data_name"], rows["team"]))
    qf = cfg["quarterfinals"]
    qf_teams = {t for p in qf for t in p} if qf else None

    def window(start, end):
        lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
        return (series[(series["league"] == WORLDS_LEAGUE) & (series["date"] >= lo)
                       & (series["date"] < hi)].sort_values("date"))

    pool = window(cfg["start"], cfg["end"])
    used: set = set()
    known = mc.Known(playin_winner=winner)
    info: dict = {"swiss": {}, "ko": {}, "playin": real}
    met: set = set()
    rec: dict | None = {t: (0, 0) for t in rows.loc[rows["role"] == "swiss", "team"]}
    if winner:
        rec[winner] = (0, 0)
    else:
        rec = None

    for r in sorted(draws):
        pairs, lab = draws[r], labels[r]
        _check_round(r, pairs, lab, rec, met, tier, league, True)
        known.draws[r], known.results[r], info["swiss"][r] = list(pairs), {}, {}
        for a, b in pairs:
            need = 2 if (lab[a][0] == 2 or lab[a][1] == 2) else 1
            found = bracket.find_series(pool, used, display, (a, b), need)
            if found is not None:
                used.add(found.series_id)
            manual = cfg["results_swiss"].get(frozenset((a, b)))
            sched = cfg.get("results_live_swiss", {}).get(frozenset((a, b)))
            if manual:
                if manual["winner"] not in (a, b):
                    raise ValueError(f"[results.swiss] {a} v {b}: {manual['winner']!r} is not in the match")
                res = {"winner": manual["winner"], "score": manual.get("score"), "source": "manual"}
            elif found is not None:
                a_won = found.wins_a > found.wins_b
                res = {"winner": display[found.team_a if a_won else found.team_b],
                       "score": f"{max(found.wins_a, found.wins_b)}-{min(found.wins_a, found.wins_b)}",
                       "source": "auto"}
                if sched and sched["winner"] != res["winner"]:
                    notes.append(f"Round {r}: {a} v {b} -- Oracle's Elixir says {res['winner']} won, "
                                 f"lolesports.com says {sched['winner']}; using Oracle's Elixir.")
            elif sched:
                res = {**sched, "source": "live"}
            else:
                res = None
                for x in (a, b):
                    v = _won(x, r, labels, qf_teams)
                    if v is not None:
                        res = {"winner": x if v == (True) else (b if x == a else a),
                               "score": None, "source": "draw"}
                        break
            if res:
                known.results[r][(a, b)] = res["winner"]
                info["swiss"][r][(a, b)] = res
        for a, b in pairs:
            met.add(frozenset((a, b)))
        if rec is not None and len(known.results[r]) == len(pairs):
            for (a, b), w in known.results[r].items():
                lo_ = b if w == a else a
                rec[w], rec[lo_] = (rec[w][0] + 1, rec[w][1]), (rec[lo_][0], rec[lo_][1] + 1)
        else:
            rec = None

    # ---- quarterfinal draw
    if qf:
        if winner is None or len(draws) < mc.ROUNDS:
            raise ValueError("quarterfinals are drawn but the Swiss rounds are not all in the config")
        if rec is not None:
            check_quarterfinals(qf, rec)
        known.qf = list(qf)

        # ---- knockout results
        kpool = window(cfg["knockout_start"], cfg["knockout_end"])
        won: dict[str, str] = {}
        feeds = {"SF1": ("QF1", "QF2"), "SF2": ("QF3", "QF4"), "F": ("SF1", "SF2")}
        for key in mc.KO_SERIES:
            if key.startswith("QF"):
                pair = qf[int(key[2]) - 1]
            else:
                a, b = (won.get(k) for k in feeds[key])
                pair = (a, b) if a and b else None
            manual = cfg["results_ko"].get(key)
            sched = cfg.get("results_live_ko", {}).get(frozenset(pair)) if pair else None
            if pair is None:
                if manual:
                    raise ValueError(f"[results.knockout] {key} is set but its teams are not decided yet")
                continue
            found = bracket.find_series(kpool, used, display, pair, 3)
            if found is not None:
                used.add(found.series_id)
            if manual:
                if manual["winner"] not in pair:
                    raise ValueError(f"[results.knockout] {key}: {manual['winner']!r} is not in {pair[0]} v {pair[1]}")
                res = {"winner": manual["winner"], "score": manual.get("score"), "source": "manual"}
            elif found is not None:
                a_won = found.wins_a > found.wins_b
                res = {"winner": display[found.team_a if a_won else found.team_b],
                       "score": f"{max(found.wins_a, found.wins_b)}-{min(found.wins_a, found.wins_b)}",
                       "source": "auto"}
                if sched and sched["winner"] != res["winner"]:
                    notes.append(f"{key}: {pair[0]} v {pair[1]} -- Oracle's Elixir says {res['winner']} won, "
                                 f"lolesports.com says {sched['winner']}; using Oracle's Elixir.")
            elif sched:
                res = {**sched, "source": "live"}
            else:
                info["ko"][key] = {"pair": pair}
                continue
            won[key] = res["winner"]
            known.ko[key] = res["winner"]
            info["ko"][key] = {"pair": pair, **res}

    # ---- series between two Swiss teams that no draw explains
    drawn = {frozenset(p) for ps in draws.values() for p in ps}
    for r_ in pool.itertuples(index=False):
        a, b = display.get(r_.team_a), display.get(r_.team_b)
        if a and b and frozenset((a, b)) not in drawn:
            notes.append(f"{r_.date:%b %d}: {a} v {b} ({r_.wins_a}-{r_.wins_b}) matches no drawn pair -- "
                         f"add that round's draw to {CONFIG.name}.")
    return Event(cfg=cfg, bcfg=bcfg, rows=rows, field=field, known=known, info=info, notes=notes)


# ------------------------------------------------------------------ live schedule

def merge_live(cfg: dict, live: Live, rounds: int, qf: bool) -> dict:
    """A copy of `cfg` with the schedule's facts layered underneath it.

    The config's own draws, quarterfinals and results are never replaced: the
    schedule only extends the draws past the config's last round (up to `rounds`),
    supplies the quarterfinal draw when the config has none (and `qf`), and adds
    results that `resolve` consults after the series log.
    """
    draws = {r: list(p) for r, p in cfg["draws"].items()}
    labels = {r: dict(l) for r, l in cfg["labels"].items()}
    r = len(draws) + 1
    while r <= rounds and r in live.draws:
        draws[r], labels[r] = list(live.draws[r]), dict(live.labels[r])
        r += 1
    out = {**cfg, "draws": draws, "labels": labels,
           "results_live_swiss": dict(live.results), "results_live_ko": dict(live.ko)}
    if qf and cfg["quarterfinals"] is None and live.qf is not None and len(draws) >= mc.ROUNDS:
        out["quarterfinals"] = list(live.qf)
    return out


def live_conflicts(cfg: dict, live: Live) -> list[str]:
    """Where the config and the schedule disagree about a draw; the config wins."""
    notes = [f"Round {r}: the draw in {CONFIG.name} differs from lolesports.com; using {CONFIG.name}."
             for r in sorted(cfg["draws"]) if r in live.draws
             and {frozenset(p) for p in cfg["draws"][r]} != {frozenset(p) for p in live.draws[r]}]
    if cfg["quarterfinals"] is not None and live.qf is not None \
            and {frozenset(p) for p in cfg["quarterfinals"]} != {frozenset(p) for p in live.qf}:
        notes.append(f"Quarterfinals: the draw in {CONFIG.name} differs from lolesports.com; "
                     f"using {CONFIG.name}.")
    return notes


def resolve_live(cfg: dict, series: pd.DataFrame, year: int = CURRENT_YEAR,
                 rows: pd.DataFrame | None = None, live: Live | None = None) -> Event:
    """`resolve`, with the schedule's draws and results filled in under the config.

    A schedule fact that does not validate (a rematch, a record that contradicts the
    results) is dropped with a note rather than raised: the config alone must still
    resolve, and the schedule only ever adds to it. The draws are tried a round at a
    time, so one bad round costs only itself and what follows.
    """
    if live is None:
        return resolve(cfg, series, year, rows)
    if rows is None:
        rows = field_rows(cfg, bracket.load_config(),
                          pd.read_parquet(paths.processed(year, "calibrated_ratings")))
    conflicts = live_conflicts(cfg, live)
    try:
        ev = resolve(merge_live(cfg, live, mc.ROUNDS, True), series, year, rows)
        ev.notes = conflicts + ev.notes
        return ev
    except (ValueError, KeyError) as exc:
        full_error = exc
    good = resolve(cfg, series, year, rows)    # a broken config fails here, as it does without a schedule
    ignored: list[str] = []
    first = len(cfg["draws"])
    top = max(max(live.draws, default=0), first)
    for k in range(first, top + 1):
        try:
            good = resolve(merge_live(cfg, live, k, False), series, year, rows)
        except (ValueError, KeyError) as exc:
            ignored.append(f"Ignored the lolesports.com results: {exc}" if k == first else
                           f"Ignored the lolesports.com draw for round {k} and later: {exc}")
            break
    else:
        ignored.append(f"Ignored the lolesports.com quarterfinal draw: {full_error}")
    good.notes = conflicts + ignored + good.notes
    return good


# ------------------------------------------------------------------ what-ifs
#
# A what-if is plain data, so it can live in session state:
#   playin  {series: winner}                     Play-In picks
#   draws   {round: {"pairs", "premise"}}        hypothetical Swiss draws
#   picks   {"r:A v B": winner}                  Swiss picks
#   qf      {"pairs", "premise"} | None          hypothetical quarterfinal draw
#   ko      {series: winner}                     knockout picks
# A hypothetical draw records its `premise` -- who it was made from -- so a
# change upstream is detected rather than silently reused.

def new_whatif() -> dict:
    return {"playin": {}, "draws": {}, "picks": {}, "qf": None, "ko": {}}


def pick_key(r: int, a: str, b: str) -> str:
    return f"{r}:{a} v {b}"


def toggle_pick(wi: dict, match_id: str, team: str) -> None:
    """Pick `team` in the series with this board id, or clear it if already picked."""
    if match_id in bracket.ORDER:
        d, key = wi["playin"], match_id
    elif match_id in mc.KO_SERIES:
        d, key = wi["ko"], match_id
    elif match_id.startswith("S"):
        d, key = wi["picks"], match_id[1:]
    else:
        raise ValueError(f"unknown series {match_id!r}")
    if d.get(key) == team:
        del d[key]
    else:
        d[key] = team


@dataclass
class State:
    """Facts plus what-if picks, resolved into what the odds are computed from."""
    field: mc.Field
    known: mc.Known
    whatif: dict            # the picks that still apply (stale ones are dropped)
    pairs: dict             # Play-In series -> (a, b) or None
    solved: dict            # bracket.solve output for the Play-In
    hypo_draws: set = dc_field(default_factory=set)   # rounds whose draw is hypothetical
    hypo_qf: bool = False
    notes: list[str] = dc_field(default_factory=list)


def _decided(known: mc.Known, r: int) -> bool:
    return r in known.draws and len(known.results.get(r, {})) == len(known.draws[r])


def records(known: mc.Known, upto: int = mc.ROUNDS) -> dict[str, tuple[int, int]]:
    """(wins, losses) of every team that has played, from the decided results."""
    rec: dict[str, list[int]] = {}
    for r in range(1, upto + 1):
        for (a, b), w in known.results.get(r, {}).items():
            lo = b if w == a else a
            rec.setdefault(w, [0, 0])[0] += 1
            rec.setdefault(lo, [0, 0])[1] += 1
    return {t: (w, l) for t, (w, l) in rec.items()}


def _premise(known: mc.Known, r: int, winner: str | None) -> list | None:
    """What round `r`'s draw is made from; None while that is not decided."""
    if r == 1:
        return [winner] if winner else None
    return sorted(known.results[r - 1].values()) if _decided(known, r - 1) else None


def _qf_premise(known: mc.Known) -> list | None:
    if not _decided(known, mc.ROUNDS):
        return None
    return sorted((t, l) for t, (w, l) in records(known).items() if w == 3)


KO_FEEDS = {"SF1": ("QF1", "QF2"), "SF2": ("QF3", "QF4"), "F": ("SF1", "SF2")}


def ko_pair(key: str, qf: list, won: dict[str, str]) -> tuple[str, str] | None:
    if key.startswith("QF"):
        return tuple(qf[int(key[2]) - 1])
    a, b = (won.get(k) for k in KO_FEEDS[key])
    return (a, b) if a and b else None


def state(event: Event, whatif: dict | None = None) -> State:
    """Layer what-if picks over the real facts.

    A real result or draw always beats a pick. A pick or hypothetical draw whose
    premise has changed (an earlier series went the other way, the Play-In winner
    changed, the stage is decided) is dropped rather than trusted, along with
    everything that was built on it.
    """
    wi_in = {**new_whatif(), **(whatif or {})}
    wi = new_whatif()

    # ---- Play-In
    picks = dict(wi_in["playin"])
    real_win = {m: r["winner"] for m, r in event.info["playin"].items()}
    opening = event.bcfg["opening"]
    if event.known.playin_winner is not None:
        picks = {}                                   # the Play-In is over
    pairs, fixed = bracket.determined(opening, {**picks, **real_win})
    wi["playin"] = {m: w for m, w in picks.items() if m not in real_win and fixed.get(m) == w}
    rating = dict(zip(event.rows["team"], event.rows["rating"]))
    solved = bracket.solve(opening, rating, fixed)
    winner = event.known.playin_winner or fixed.get("QF")
    field = field_from_rows(event.rows, playin_probs(event.bcfg, rating, fixed, winner))

    known = replace(event.known, playin_winner=winner,
                    draws={r: list(p) for r, p in event.known.draws.items()},
                    results={r: dict(d) for r, d in event.known.results.items()},
                    ko=dict(event.known.ko))

    # ---- Swiss: a hypothetical draw needs everything before it decided
    hypo: set[int] = set()
    for r in range(1, mc.ROUNDS + 1):
        if r not in event.known.draws:
            entry = wi_in["draws"].get(r)
            if not entry or entry["premise"] != _premise(known, r, winner):
                break
            known.draws[r] = [tuple(p) for p in entry["pairs"]]
            known.results.setdefault(r, {})
            wi["draws"][r] = {"pairs": [list(p) for p in entry["pairs"]], "premise": list(entry["premise"])}
            hypo.add(r)
        for pair in known.draws[r]:
            if pair not in known.results[r]:
                w = wi_in["picks"].get(pick_key(r, *pair))
                if w in pair:
                    known.results[r][pair] = w
                    wi["picks"][pick_key(r, *pair)] = w
        if not _decided(known, r):
            break

    # ---- quarterfinal draw
    hypo_qf = False
    entry = wi_in["qf"]
    if known.qf is None and entry and entry["premise"] == _qf_premise(known):
        known.qf = [tuple(p) for p in entry["pairs"]]
        wi["qf"] = {"pairs": [list(p) for p in entry["pairs"]], "premise": list(entry["premise"])}
        hypo_qf = True

    # ---- knockout picks
    if known.qf is not None:
        won = dict(known.ko)
        for key in mc.KO_SERIES:
            pair = ko_pair(key, known.qf, won)
            if pair is None or key in known.ko:
                continue
            w = wi_in["ko"].get(key)
            if w in pair:
                known.ko[key] = won[key] = wi["ko"][key] = w
    return State(field=field, known=known, whatif=wi, pairs=pairs, solved=solved,
                 hypo_draws=hypo, hypo_qf=hypo_qf, notes=list(event.notes))


def _known_before(known: mc.Known, r: int) -> mc.Known:
    """`known` with round `r` and everything after it forgotten."""
    return mc.Known(playin_winner=known.playin_winner,
                    draws={k: v for k, v in known.draws.items() if k < r},
                    results={k: v for k, v in known.results.items() if k < r})


def draw_round(stt: State, wi: dict, r: int, rng: np.random.Generator) -> None:
    """Make a hypothetical, legal draw for Swiss round `r` (a redraw, if it exists)."""
    before = _known_before(stt.known, r)
    premise = _premise(before, r, before.playin_winner)
    if premise is None:
        raise ValueError(f"round {r} cannot be drawn until the round before it is decided")
    pairs = mc.sample_round(stt.field, before, rng)
    wi["draws"][r] = {"pairs": [list(p) for p in pairs], "premise": premise}


def draw_quarterfinals(stt: State, wi: dict, rng: np.random.Generator) -> None:
    before = replace(stt.known, qf=None, ko={})
    premise = _qf_premise(before)
    if premise is None:
        raise ValueError("the quarterfinals cannot be drawn until the Swiss stage is decided")
    wi["qf"] = {"pairs": [list(p) for p in mc.sample_quarterfinals(stt.field, before, rng)],
                "premise": premise}


def met_before(known: mc.Known, r: int) -> set[frozenset]:
    """The pairs already drawn in rounds before `r`."""
    return {frozenset(p) for k, ps in known.draws.items() if k < r for p in ps}


def _swiss_teams(stt: State) -> list[str]:
    return stt.field.teams[:mc.DIRECT] + ([stt.known.playin_winner] if stt.known.playin_winner else [])


def draw_groups(stt: State, r: int) -> list[tuple[tuple[int, int], list[str]]]:
    """Round `r`'s record groups, each with its teams from the highest rated down."""
    rec = records(stt.known, r - 1)
    rating = dict(zip(stt.field.teams, stt.field.rating))
    teams = sorted(_swiss_teams(stt), key=lambda t: -rating[t])
    return [((w, l), [t for t in teams if rec.get(t, (0, 0)) == (w, l)])
            for w, l, _ in mc._layout(r)]


def set_draw(stt: State, wi: dict, r: int, pairs: list[tuple[str, str]]) -> None:
    """Enter Swiss round `r`'s draw by hand, under the same rules as a real one."""
    if r in stt.known.draws and r not in stt.hypo_draws:
        raise ValueError(f"round {r} is already drawn")
    before = _known_before(stt.known, r)
    premise = _premise(before, r, before.playin_winner)
    if premise is None:
        raise ValueError(f"round {r} cannot be drawn until the round before it is decided")
    rec = {t: records(before).get(t, (0, 0)) for t in _swiss_teams(stt)}
    seen: set[str] = set()
    for a, b in pairs:
        for t in (a, b):
            if t not in rec:
                raise ValueError(f"{t!r} is not in the Swiss stage")
            if 3 in rec[t]:
                raise ValueError(f"{t} has finished the Swiss stage")
            if t in seen:
                raise ValueError(f"{t} is drawn twice")
            seen.add(t)
        if rec[a] != rec[b]:
            raise ValueError(f"{a} ({rec[a][0]}-{rec[a][1]}) and {b} ({rec[b][0]}-{rec[b][1]}) "
                             f"are not on the same record")
    field = stt.field
    tier = dict(zip(field.teams[:mc.DIRECT], field.tier)) | {before.playin_winner: 4}
    league = dict(zip(field.teams, field.league))
    _check_round(r, pairs, {t: rec[t] for t in seen}, rec, met_before(before, r), tier, league,
                 field.league_rule)
    wi["draws"][r] = {"pairs": [list(p) for p in pairs], "premise": premise}


def set_quarterfinals(stt: State, wi: dict, pairs: list[tuple[str, str]]) -> None:
    """Enter the quarterfinal draw by hand, under the same rules as a real one."""
    if stt.known.qf is not None and not stt.hypo_qf:
        raise ValueError("the quarterfinals are already drawn")
    before = replace(stt.known, qf=None, ko={})
    premise = _qf_premise(before)
    if premise is None:
        raise ValueError("the quarterfinals cannot be drawn until the Swiss stage is decided")
    check_quarterfinals(pairs, records(before))
    wi["qf"] = {"pairs": [list(p) for p in pairs], "premise": premise}


def play_out(event: Event, whatif: dict | None, rng: np.random.Generator) -> dict:
    """Fill every open decision once, weighted by the model: Play-In series, draws,
    Swiss results, the quarterfinal draw and the knockout. Real facts stay as they
    are, and so do the picks already made."""
    wi = {**new_whatif(), **(whatif or {})}
    wi = {"playin": dict(wi["playin"]), "draws": dict(wi["draws"]), "picks": dict(wi["picks"]),
          "qf": wi["qf"], "ko": dict(wi["ko"])}

    if event.known.playin_winner is None:
        rating = dict(zip(event.rows["team"], event.rows["rating"]))
        real_win = {m: r["winner"] for m, r in event.info["playin"].items()}
        opening = event.bcfg["opening"]
        for m in bracket.ORDER:
            pairs, fixed = bracket.determined(opening, {**wi["playin"], **real_win})
            if m in fixed:
                continue
            a, b = pairs[m]
            wi["playin"][m] = a if rng.random() < bracket.p_match(rating[a], rating[b]) else b

    stt = state(event, wi)
    tr = mc.run_trace(stt.field, stt.known, int(rng.integers(2**31 - 1)))
    for r in range(1, mc.ROUNDS + 1):
        if r not in stt.known.draws:
            wi["draws"][r] = {"pairs": [list(p) for p in tr.draws[r]],
                              "premise": [tr.playin_winner] if r == 1 else sorted(tr.winners[r - 1])}
        for pair, w in zip(tr.draws[r], tr.winners[r]):
            if pair not in stt.known.results.get(r, {}):
                wi["picks"][pick_key(r, *pair)] = w
    if stt.known.qf is None:
        rec: dict[str, list[int]] = {}
        for r in range(1, mc.ROUNDS + 1):
            for (a, b), w in zip(tr.draws[r], tr.winners[r]):
                lo = b if w == a else a
                rec.setdefault(w, [0, 0])[0] += 1
                rec.setdefault(lo, [0, 0])[1] += 1
        wi["qf"] = {"pairs": [list(p) for p in tr.qf],
                    "premise": sorted((t, l) for t, (w, l) in rec.items() if w == 3)}
    for key, w in tr.ko.items():
        if key not in stt.known.ko:
            wi["ko"][key] = w
    return wi


# ------------------------------------------------------------------ outputs

def load_series(year: int) -> pd.DataFrame:
    path = paths.processed(year, "series")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=SERIES_COLS)


def write_outputs(year: int = CURRENT_YEAR, n: int = 20000, seed: int | None = 0) -> pd.DataFrame:
    """Write the Swiss field and the title-odds table. Returns the table."""
    import lolesports      # here, not at the top: lolesports imports this module
    cfg = load_config()
    bcfg = bracket.load_config()
    paths.ensure_dirs()
    events, src = lolesports.schedule(year, str(bcfg["start"]))
    live = None
    if events:
        try:
            live = lolesports.live_facts(cfg, bcfg, events)
        except Exception as exc:     # a malformed event is a KeyError/TypeError, not a ValueError
            src += f"; could not be read: {exc}"
    ev = resolve_live(cfg, load_series(year), year, live=live)
    ev.rows.to_csv(paths.processed(year, "swiss_field", "csv"), index=False)

    odds = mc.simulate(ev.field, ev.known, n, seed)
    table = mc.odds_table(ev.field, odds, "confirmed")
    table.to_csv(paths.processed(year, "worlds_title_odds", "csv"), index=False)

    k = ev.known
    print(f"=== WORLDS {year} -- SWISS + KNOCKOUT ODDS ({n:,} sims) ===")
    print(f"Schedule: {src}")
    print(f"Play-In winner: {k.playin_winner or 'undecided (sampled from the Play-In odds)'}; "
          f"rounds drawn: {sorted(k.draws) or 'none'}; quarterfinals drawn: {bool(k.qf)}; "
          f"knockout results: {len(k.ko)}")
    for note in ev.notes:
        print(f"NOTE: {note}")
    if odds.fallback:
        print(f"WARNING: {odds.fallback} draws could not avoid a rematch / same-league pair")
    print(table.to_string(index=False))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0, help="-1 for random")
    args = ap.parse_args()
    write_outputs(args.year, args.n, None if args.seed == -1 else args.seed)


if __name__ == "__main__":
    main()
