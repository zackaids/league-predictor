"""Worlds Crystal Ball: a probability for every answer of the 18 Pick'Em questions.

The tournament comes from the existing engine -- the Play-In walked with
`bracket`, the Swiss + knockout played out by `monte_carlo.run_trace` -- and each
series is then expanded into games (length drawn from the same per-game
probability the engine uses). Each game's stats are bootstrapped from the two
teams' own recent games with the SAME RESULT, so winners play shorter, more
snowballing games than losers; Fearless draft is enforced inside a series.

Questions are answered from the simulated tournaments: champion and player
leaders, thresholds, regions, placement. The Teemo and Finals-MVP-role questions
are not simulable and use base rates (crystal_ball.toml).

Run:
    python crystal_ball.py                    # 2026 predictions + expected points
    python crystal_ball.py --backtest 2025    # replay a past Worlds; checks the model
"""
from __future__ import annotations

import argparse
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import comb

import numpy as np
import pandas as pd

import bracket
import calibrate
import monte_carlo as mc
import paths
import swiss
from paths import CURRENT_YEAR

CONFIG = paths.ROOT / "crystal_ball.toml"
ROLES = ["top", "jng", "mid", "bot", "sup"]

POOL_DAYS = 135        # a team's games this long before the event feed its pool
STARTER_DAYS = 75      # who counts as a role's starter
HALF_LIFE = 45.0       # days; recent games weigh more
BANDWIDTH = 0.15       # a pool game is matched to the simulated matchup by pre-game win probability
P_BINS = np.array([0.12, 0.25, 0.37, 0.5, 0.63, 0.75, 0.88])
MAX_REDRAW = 25        # Fearless redraws before a clash is allowed
MIN_STARTERS = 4       # a pool game needs this many of the five starters
MIN_POOL = 8           # fewer games than this: use all of the team's games

RAW_COLS = ["gameid", "league", "date", "game", "position", "playername", "teamname",
            "champion", "ban1", "ban2", "ban3", "ban4", "ban5", "gamelength", "result",
            "kills", "deaths", "assists", "teamkills", "damagetochampions", "total cs", "patch"]

QUESTIONS = {
    "presence": "Highest presence (picked or banned) champion",
    "champ_wr": "Highest win rate champion (min 5 games)",
    "champ_deaths": "Most deaths per game champion (min 5 games)",
    "kda": "Highest average KDA",
    "damage": "Highest average damage per game",
    "kp": "Highest average kill participation",
    "peyz_guma": "Peyz vs Gumayusi: more total kills",
    "faker_700": "Faker reaches 700 career Worlds kills",
    "teemo": "Teemo picked at least once",
    "unique_champs": "Unique champions picked",
    "length_32": "More games finish before or after 32 minutes",
    "kills_15": "Any pro with >= 15 kills in a game",
    "cs_450": "Any pro with >= 450 CS in a game",
    "mvp_role": "Role of the Finals MVP",
    "regions_ko": "Different regions in Knockouts",
    "lcs_ko": "An LCS team reaches Knockouts",
    "hle_t1": "Which places higher: HLE or T1",
    "shortest_game": "Team that wins the shortest game",
}


def load_config(path=CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ------------------------------------------------------------------ data

@dataclass(slots=True)
class Rec:
    """One team's side of one game: its five players' lines, picks and bans."""
    gameid: str
    team: str
    date: pd.Timestamp
    game: float
    result: int
    length: float                       # minutes
    champs: tuple
    players: tuple
    k: tuple
    d: tuple
    a: tuple
    dmg: tuple
    cs: tuple
    tk: int
    bans: tuple
    p: float = 0.5                      # P(this team won), from Elo before the game


def load_raw(year: int) -> pd.DataFrame:
    df = pd.read_csv(paths.raw_csv(year), usecols=RAW_COLS, dtype={"patch": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


def worlds_window(df: pd.DataFrame) -> pd.Timestamp:
    """First day of the Worlds main event.

    League code WLDs also holds the LCK/LPL regional qualifiers weeks earlier
    (they are not Worlds games: unfiltered, 2024's top KDA was a KT qualifier
    player), so the event is what follows the last gap of more than a week.
    """
    days = np.sort(df.loc[df["league"] == "WLDs", "date"].dt.normalize().unique())
    gaps = np.where(np.diff(days) > np.timedelta64(7, "D"))[0]
    return pd.Timestamp(days[gaps[-1] + 1] if len(gaps) else days[0])


def elo_replay(df: pd.DataFrame, start: pd.Timestamp, k: float = 30.0):
    """A plain chronological Elo over every game in `df`.

    Returns ({(gameid, team): P(team wins) before the game}, ratings at `start`).
    It only has to say how lopsided a pool game was, so leagues staying
    disconnected does no harm; per-game probabilities use the fitted shrinkage.
    """
    t = df[df["position"] == "team"].drop_duplicates(["gameid", "teamname"]).sort_values(["date", "gameid"])
    rating: dict[str, float] = {}
    at_start: dict[str, float] = {}
    pmap: dict[tuple, float] = {}
    for gid, g in t.groupby("gameid", sort=False):
        if len(g) != 2:
            continue
        date = g["date"].iloc[0]
        if not at_start and date >= start:
            at_start = dict(rating)
        a, b = g["teamname"].tolist()
        ra, rb = rating.get(a, 1500.0), rating.get(b, 1500.0)
        pa = float(calibrate.prob_from_gap(ra - rb))
        pmap[(gid, a)], pmap[(gid, b)] = pa, 1 - pa
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        won_a = int(g["result"].iloc[0])
        rating[a] = ra + k * (won_a - ea)
        rating[b] = rb + k * ((1 - won_a) - (1 - ea))
    return pmap, at_start or dict(rating)


def build_records(df: pd.DataFrame, pmap: dict | None = None) -> list[Rec]:
    team = df[df["position"] == "team"].drop_duplicates(["gameid", "teamname"])
    tinfo = team.set_index(["gameid", "teamname"])
    need = ["champion", "kills", "deaths", "assists", "damagetochampions", "total cs"]
    out = []
    for (gid, tn), g in df[df["position"] != "team"].groupby(["gameid", "teamname"], sort=False):
        if len(g) != 5 or set(g["position"]) != set(ROLES) or (gid, tn) not in tinfo.index:
            continue
        g = g.set_index("position").loc[ROLES]
        if g[need].isna().any().any():
            continue
        t = tinfo.loc[(gid, tn)]
        out.append(Rec(
            gameid=gid, team=tn, date=t["date"], game=pd.to_numeric(t["game"], errors="coerce"),
            result=int(t["result"]), length=float(t["gamelength"]) / 60,
            champs=tuple(g["champion"]), players=tuple(g["playername"].fillna("?")),
            k=tuple(int(x) for x in g["kills"]), d=tuple(int(x) for x in g["deaths"]),
            a=tuple(int(x) for x in g["assists"]),
            dmg=tuple(float(x) for x in g["damagetochampions"]),
            cs=tuple(float(x) for x in g["total cs"]), tk=int(t["teamkills"]),
            bans=tuple(b for b in (t[f"ban{i}"] for i in range(1, 6)) if isinstance(b, str)),
            p=(pmap or {}).get((gid, tn), 0.5)))
    return out


class Pools:
    """Each team's recent games, ready to be sampled by result."""

    def __init__(self, recs: list[Rec], start: pd.Timestamp):
        self.start = start
        by: dict[str, list[Rec]] = defaultdict(list)
        for r in recs:
            by[r.team].append(r)
        cut = start - pd.Timedelta(days=STARTER_DAYS)
        self.starters: dict[str, list[str]] = {}
        self.by_team: dict[str, list[Rec]] = {}
        for t, rs in by.items():
            recent = [r for r in rs if r.date >= cut] or rs
            st = [Counter(r.players[i] for r in recent).most_common(1)[0][0] for i in range(5)]
            ok = [r for r in rs if sum(p == s for p, s in zip(r.players, st)) >= MIN_STARTERS]
            use = ok if len(ok) >= MIN_POOL else rs
            self.starters[t] = st
            # A substitute's line is credited to the role's starter.
            self.by_team[t] = [replace(r, players=tuple(st)) for r in use]
        self._global = [r for rs in self.by_team.values() for r in rs]
        self._arr: dict[tuple, tuple] = {}
        self.fallbacks = Counter()
        self.clashes = 0

    def size(self, team: str) -> int:
        return len(self.by_team.get(team, []))

    def _get(self, team: str, result: int, p: float = 0.5):
        b = int(np.abs(P_BINS - p).argmin()) if BANDWIDTH else 0
        key = (team, result, b)
        if key not in self._arr:
            mine = self.by_team.get(team, [])
            rs = [r for r in mine if r.result == result]
            if not rs:
                self.fallbacks[team] += 1
                rs = mine or [r for r in self._global if r.result == result]
            age = np.array([(self.start - r.date).total_seconds() / 86400 for r in rs])
            w = 0.5 ** (age / HALF_LIFE)
            if BANDWIDTH:
                w = w * np.exp(-0.5 * ((np.array([r.p if result else 1 - r.p for r in rs]) - P_BINS[b])
                                       / BANDWIDTH) ** 2) + 1e-9
            self._arr[key] = (rs, np.cumsum(w))
        return self._arr[key]

    def draw(self, team: str, result: int, rng: np.random.Generator, locked: set, p: float = 0.5) -> Rec:
        rs, cum = self._get(team, result, p)
        for _ in range(MAX_REDRAW):
            i = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), len(rs) - 1)
            r = rs[i]
            if locked.isdisjoint(r.champs):
                return r
        self.clashes += 1
        return r

    def role_of(self, team: str, player: str) -> str | None:
        st = self.starters.get(team, [])
        return ROLES[st.index(player)] if player in st else None


# ------------------------------------------------------------------ one tournament's statistics

def leaders(d: dict) -> list:
    if not d:
        return []
    m = max(d.values())
    return [k for k, v in d.items() if v == m]


class Acc:
    """Everything the questions need, accumulated over one tournament's games."""

    def __init__(self):
        self.pres, self.picks, self.wins, self.deaths = Counter(), Counter(), Counter(), Counter()
        self.pg, self.K, self.D, self.A = Counter(), Counter(), Counter(), Counter()
        self.TK, self.dmg = Counter(), Counter()
        self.lens: list[float] = []
        self.under = self.over = self.g15 = self.g450 = 0
        self.shortest = (float("inf"), None)
        self.picked: set = set()
        self.seen: set = set()          # picked or banned
        self.team_wins: Counter = Counter()

    @property
    def n(self) -> int:
        return len(self.lens)

    def game(self, wt: str, wr: Rec, lr: Rec, eff: float) -> None:
        self.lens.append(eff)
        self.team_wins[wt] += 1
        if eff < 32:
            self.under += 1
        elif eff > 32:
            self.over += 1
        if eff < self.shortest[0]:
            self.shortest = (eff, wt)
        for c in set(wr.champs) | set(lr.champs) | set(wr.bans) | set(lr.bans):
            self.pres[c] += 1
            self.seen.add(c)
        gk, gcs = 0, 0.0
        for rec, res in ((wr, 1), (lr, 0)):
            f = eff / rec.length          # per-minute stats are rescaled to the game's length
            self.picked.update(rec.champs)
            for i in range(5):
                c, p = rec.champs[i], rec.players[i]
                self.picks[c] += 1
                self.wins[c] += res
                self.deaths[c] += rec.d[i]
                self.pg[p] += 1
                self.K[p] += rec.k[i]
                self.D[p] += rec.d[i]
                self.A[p] += rec.a[i]
                self.TK[p] += rec.tk
                self.dmg[p] += rec.dmg[i] * f
                gk = max(gk, rec.k[i])
                gcs = max(gcs, rec.cs[i] * f)
        self.g15 += gk >= 15
        self.g450 += gcs >= 450

    def kda(self, p: str) -> float:
        return (self.K[p] + self.A[p]) / max(self.D[p], 1)

    def answers(self) -> dict:
        elig = [c for c, n in self.picks.items() if n >= 5]
        return {
            "presence": leaders(self.pres),
            "champ_wr": leaders({c: self.wins[c] / self.picks[c] for c in elig}),
            "champ_deaths": leaders({c: self.deaths[c] / self.picks[c] for c in elig}),
            "kda": leaders({p: self.kda(p) for p in self.pg}),
            "damage": leaders({p: self.dmg[p] / self.pg[p] for p in self.pg}),
            "kp": leaders({p: (self.K[p] + self.A[p]) / self.TK[p] for p in self.pg if self.TK[p]}),
            "unique_champs": len(self.picked),
            "unique_seen": len(self.seen),
            "length_32": "Before 32:00" if self.under > self.over else
                         "After 32:00" if self.over > self.under else "Equal",
            "kills_15": "Yes" if self.g15 else "No",
            "cs_450": "Yes" if self.g450 else "No",
            "shortest_game": [self.shortest[1]],
        }


# ------------------------------------------------------------------ presence: meta model

def _presence(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Share of games each champion is picked or banned in (either side).

    `weights` (gameid -> weight) makes it a weighted share."""
    team = df[df["position"] == "team"].drop_duplicates(["gameid", "teamname"])
    ply = df[df["position"] != "team"]
    seen = pd.concat([ply[["gameid", "champion"]].rename(columns={"champion": "c"}),
                      *[team[["gameid", f"ban{i}"]].rename(columns={f"ban{i}": "c"}) for i in range(1, 6)]])
    seen = seen.dropna().drop_duplicates()
    if weights is None:
        return seen["c"].value_counts() / max(team["gameid"].nunique(), 1)
    total = sum(weights.get(g, 1.0) for g in team["gameid"].unique())
    return seen.assign(w=seen["gameid"].map(weights).fillna(1.0)).groupby("c")["w"].sum() / max(total, 1e-9)


def meta_signals(df: pd.DataFrame, start: pd.Timestamp, teams: set[str],
                 newest_weight: float = 1.0) -> pd.DataFrame:
    """Two pre-event presence signals per champion.

    A: every league, on the latest patch with 150+ games and any newer one --
       Worlds is played on a fresh patch, so the newest data is the best guide --
       with the newest patch that has 40+ games counted `newest_weight` times.
    C: the event teams' own games over the pool window.
    """
    pre = df[(df["date"] < start) & (df["league"] != "WLDs")]
    key = lambda s_: tuple(int(x) for x in s_.split("."))
    counts = pre.groupby("patch")["gameid"].nunique()
    big = sorted((p_ for p_ in counts.index if counts[p_] >= 150), key=key)
    newest = big[-1]
    latest = sorted((p_ for p_ in counts.index if counts[p_] >= 40), key=key)[-1]
    on_patch = pre[pre["patch"].map(lambda s_: isinstance(s_, str) and key(s_) >= key(newest))]
    per_game = on_patch.drop_duplicates("gameid").set_index("gameid")["patch"]
    weights = {g: (newest_weight if p_ == latest else 1.0) for g, p_ in per_game.items()}
    sig = pd.DataFrame({
        "A": _presence(on_patch, weights),
        "C": _presence(pre[(pre["date"] >= start - pd.Timedelta(days=POOL_DAYS)) & pre["teamname"].isin(teams)]),
    }).fillna(0.0)
    sig.attrs["patch"] = newest
    sig.attrs["latest"] = latest
    return sig


def presence_leaders(sig: pd.DataFrame, cfg: dict, n: int = 40000, seed: int = 0) -> dict:
    """P(each champion has the highest presence).

    Expected Worlds presence is a regression on the signals (fitted on Worlds
    2021-25); the residual is large (sd ~0.2), so the leader is mostly luck of
    which champions the event's patch and teams happen to favour.
    """
    m = cfg["model"]["meta_presence"]
    sig = sig[sig[["A", "C"]].max(axis=1) >= 0.10]
    mu = m["intercept"] + m["a"] * sig["A"].to_numpy() + m["c"] * sig["C"].to_numpy()
    rng = np.random.default_rng(seed)
    draw = np.clip(mu[None, :] + m["sigma"] * rng.standard_normal((n, len(mu))), 0, 1)
    lead = pd.Series(sig.index[draw.argmax(axis=1)]).value_counts(normalize=True)
    return lead.to_dict()


def wr_lottery(df: pd.DataFrame, start: pd.Timestamp, sig: pd.DataFrame, cfg: dict) -> dict:
    """Highest win rate (5+ games) is a lottery: flat odds over plausible champions.

    Backtests found no signal. The simulation gave the real leader 0-4% in 2020-25
    (its own top picks, at 10-36%, never won), and neither a champion's pre-event
    win rate nor its win rate on the newest patch beat a uniform guess: the leader
    is usually a 5-8 game champion on a lucky streak, and in 6 of 9 years was not
    even a high-volume pre-event champion. So every champion expected to reach 5
    games gets the same chance; the order within that (the tie-break for the single
    pick) is newest-patch win rate, which is a preference, not evidence.
    """
    m = cfg["model"]["meta_presence"]
    mu = m["intercept"] + m["a"] * sig["A"] + m["c"] * sig["C"]
    elig = mu[mu >= cfg["model"]["wr_min_presence"]].index
    key = lambda s_: tuple(int(x) for x in s_.split("."))
    pre = df[(df["date"] < start) & (df["league"] != "WLDs") & (df["position"] != "team")]
    on = pre[pre["patch"].map(lambda s_: isinstance(s_, str) and key(s_) >= key(sig.attrs["patch"]))]
    wr = on.groupby("champion")["result"].agg(["size", "mean"])
    shrunk = ((wr["size"] * wr["mean"] + 40 * 0.5) / (wr["size"] + 40)).reindex(elig).fillna(0.5)
    order = shrunk.sort_values(ascending=False).index
    return {c: 1 / len(order) for c in order}


# ------------------------------------------------------------------ deaths per game: high-volume champions

MAJOR_LEAGUES = ["LCK", "LPL", "LEC", "LCS", "LTA N", "LTA S", "LCP", "PCS", "VCS", "CBLOL",
                 "MSI", "EWC", "FST", "LLA", "LJL", "LCO"]


def deaths_leaders(df: pd.DataFrame, start: pd.Timestamp, cfg: dict) -> dict:
    """P(each champion has the most deaths per game among those with 5+ games).

    The pool simulation lets any champion win by a lucky 5-game average (Lissandra),
    but the real leader is a high-volume engage champion: Nautilus or Leona at each
    of the last 6 Worlds, and in the pre-event major-league data they sat top-5
    every year. So rank champions with 30+ pre-event games by shrunk deaths per
    game (recency-weighted, blended with the newest patch across all leagues) and
    turn the gaps into odds. Fit on Worlds 2020-25: 1st-5th place in 6 of 6.
    """
    m = cfg["model"]["deaths"]
    pre = df[(df["date"] < start) & (df["date"] >= start - pd.Timedelta(days=POOL_DAYS))
             & (df["league"] != "WLDs") & (df["position"] != "team")]
    d = pre[pre["league"].isin(MAJOR_LEAGUES)].copy()
    d["w"] = 0.5 ** ((start - d["date"]).dt.total_seconds() / 86400 / m["half_life"])
    g = d.assign(wd=d["w"] * d["deaths"]).groupby("champion").agg(n=("w", "size"), wn=("w", "sum"), wd=("wd", "sum"))
    mu, k = g["wd"].sum() / g["wn"].sum(), m["shrink"]
    g["score"] = (g["wd"] + k * mu) / (g["wn"] + k)
    key = lambda s_: tuple(int(x) for x in s_.split("."))
    counts = pre.groupby("patch")["gameid"].nunique()
    big = sorted((p_ for p_ in counts.index if counts[p_] >= 150), key=key)
    if big:
        on = pre[pre["patch"].map(lambda s_: isinstance(s_, str) and key(s_) >= key(big[-1]))]
        on = on.groupby("champion")["deaths"].agg(["size", "mean"])
        newest = ((on["size"] * on["mean"] + k * mu) / (on["size"] + k)).reindex(g.index)
        g["score"] = (1 - m["newest_weight"]) * g["score"] + m["newest_weight"] * newest.fillna(g["score"])
    g = g[g["n"] >= m["min_games"]]
    gap = (g["score"] - g["score"].max()).to_numpy()
    tau = m["tau"]
    # The fit is for typical gaps; when one champion stands far clear (this year's
    # newest-patch signal comes from minor leagues, where everyone dies more) the
    # softmax saturates. Its top pick won 2-3 of 6 backtest years, so claim no more.
    while (np.exp(gap / tau) / np.exp(gap / tau).sum()).max() > m["max_top"]:
        tau *= 1.05
    p = np.exp(gap / tau)
    return pd.Series(p / p.sum(), index=g.index).sort_values(ascending=False).to_dict()


# ------------------------------------------------------------------ playing games

def play_game(pools: Pools, rng, accs: list, ta: str, tb: str, a_won: bool,
              locked: set, off: float, p_a: float = 0.5,
              name_a: str | None = None, name_b: str | None = None) -> None:
    """`p_a` is the pre-game chance that `ta` wins; it picks how lopsided a game to draw."""
    ra = pools.draw(ta, int(a_won), rng, locked, p_a)
    rb = pools.draw(tb, int(not a_won), rng, locked | set(ra.champs), 1 - p_a)
    wr, lr, wt = (ra, rb, name_a or ta) if a_won else (rb, ra, name_b or tb)
    eff = wr.length + off
    for acc in accs:
        acc.game(wt, wr, lr, eff)
    locked.update(ra.champs)
    locked.update(rb.champs)


def play_series(pools, rng, accs, oe, rating, a, b, w, need, off) -> None:
    """Expand a decided series into games: the loser's wins are placed at random."""
    p = float(calibrate.prob_from_gap(rating[w] - rating[b if w == a else a]))
    q = 1 - p
    pr = np.array([comb(need - 1 + k, k) * p ** need * q ** k for k in range(need)])
    k = int(rng.choice(need, p=pr / pr.sum()))
    seq = [True] * (need - 1) + [False] * k
    rng.shuffle(seq)
    seq.append(True)
    locked: set = set()
    p_a = float(calibrate.prob_from_gap(rating[a] - rating[b]))
    for took in seq:
        play_game(pools, rng, accs, oe[a], oe[b], (a == w) == took, locked, off, p_a, a, b)


def placement(t: str, tr, rec: dict, ko: dict) -> int:
    if t == tr.ko["F"]:
        return 1
    if t in ko["F"]:
        return 2
    if t in ko["SF1"] + ko["SF2"]:
        return 3
    if any(t in p for p in tr.qf):
        return 5
    if t not in rec:
        return 17
    return {2: 9, 1: 12, 0: 15}[rec[t][0]]


def simulate_event(ev, pools: Pools, rng, off: float, include_playin: bool, cfg: dict):
    """One whole Worlds. Returns (stats for every game, stats for the final, facts)."""
    f = ev.field
    rating, league = dict(zip(f.teams, f.rating)), dict(zip(f.teams, f.league))
    oe = dict(zip(ev.rows["team"], ev.rows["data_name"]))
    acc, fin = Acc(), Acc()

    known = ev.known
    if known.playin_winner is None:
        real = {m: r["winner"] for m, r in ev.info.get("playin", {}).items()}
        won, lost = {}, {}
        for m in bracket.ORDER:
            a, b = bracket._pair(m, ev.bcfg["opening"], won, lost)
            w = real.get(m) or (a if rng.random() < bracket.p_match(rating[a], rating[b]) else b)
            won[m], lost[m] = w, (b if w == a else a)
            if include_playin:
                play_series(pools, rng, [acc], oe, rating, a, b, w, bracket.NEED, off)
        known = replace(known, playin_winner=won["QF"])

    tr = mc.run_trace(f, known, int(rng.integers(2 ** 31 - 1)))
    rec: dict[str, tuple] = {}
    for r in range(1, mc.ROUNDS + 1):
        for (a, b), w in zip(tr.draws[r], tr.winners[r]):
            wa, la = rec.get(a, (0, 0))
            play_series(pools, rng, [acc], oe, rating, a, b, w, 2 if 2 in (wa, la) else 1, off)
            l = b if w == a else a
            rw, rl = rec.get(w, (0, 0)), rec.get(l, (0, 0))
            rec[w], rec[l] = (rw[0] + 1, rw[1]), (rl[0], rl[1] + 1)

    ko = {f"QF{i + 1}": tr.qf[i] for i in range(4)}
    ko["SF1"], ko["SF2"] = (tr.ko["QF1"], tr.ko["QF2"]), (tr.ko["QF3"], tr.ko["QF4"])
    ko["F"] = (tr.ko["SF1"], tr.ko["SF2"])
    for key, (a, b) in ko.items():
        play_series(pools, rng, [acc, fin] if key == "F" else [acc], oe, rating, a, b, tr.ko[key], 3, off)

    kot = {t for p in tr.qf for t in p}
    champ = tr.ko["F"]
    facts = {
        "regions": len({league[t] for t in kot}),
        "lcs": any(league[t] == cfg["teams"]["lcs"] for t in kot),
        "hle": placement(cfg["teams"]["hle"], tr, rec, ko),
        "t1": placement(cfg["teams"]["t1"], tr, rec, ko),
        "champion": champ,
        "mvp": mvp_role(fin, pools, oe[champ]),
    }
    return acc, facts


def mvp_role(fin: Acc, pools: Pools, team: str) -> str | None:
    """Role of the champion's best player over the final (heuristic: highest KDA)."""
    best = max(pools.starters.get(team, []), key=lambda p: (fin.kda(p), fin.dmg[p]), default=None)
    return pools.role_of(team, best) if best else None


# ------------------------------------------------------------------ tallying

class Tally:
    def __init__(self):
        self.cat: dict[str, Counter] = defaultdict(Counter)
        self.num: dict[str, list] = defaultdict(list)
        self.n = 0

    def add(self, q: str, answers) -> None:
        answers = answers if isinstance(answers, list) else [answers]
        for a in answers:
            self.cat[q][a] += 1 / len(answers)

    def dist(self, q: str) -> dict:
        return {a: v / self.n for a, v in self.cat[q].items()}


def tally_sim(t: Tally, acc: Acc, facts: dict | None, cfg: dict | None) -> None:
    t.n += 1
    for q, ans in acc.answers().items():
        if q in ("unique_champs", "unique_seen"):
            t.num[q].append(ans)
        else:
            t.add(q, ans)
    for team, w in acc.team_wins.items():
        t.cat["wins_share"][team] += w / acc.n
    t.num["games"].append(acc.n)
    t.num["median_len"].append(float(np.median(acc.lens)))
    t.num["under_share"].append(acc.under / max(acc.n, 1))
    t.num["g15"].append(acc.g15)
    t.num["g450"].append(acc.g450)
    t.num["shortest_len"].append(acc.shortest[0])
    if facts is None:
        return
    p = cfg["players"]
    k = acc.K
    t.add("peyz_guma", p["a"] if k[p["a"]] > k[p["b"]] else p["b"] if k[p["b"]] > k[p["a"]] else "Tie")
    t.add("faker_700", "Yes" if k[p["faker"]] >= p["faker_target"] - p["faker_career_kills"] else "No")
    t.num["faker_kills"].append(k[p["faker"]])
    t.add("regions_ko", facts["regions"])
    t.add("lcs_ko", "Yes" if facts["lcs"] else "No")
    t.add("hle_t1", "HLE" if facts["hle"] < facts["t1"] else "T1" if facts["t1"] < facts["hle"] else "Same placement")
    t.add("mvp_sim", facts["mvp"] or "unknown")
    t.add("champion", facts["champion"])


# ------------------------------------------------------------------ backtest

def actual_games(recs: list[Rec]):
    """The event's real games in order, with a series id for Fearless locking."""
    by = defaultdict(list)
    for r in recs:
        by[r.gameid].append(r)
    games = sorted(((rs[0].date, gid, rs[0], rs[1]) for gid, rs in by.items()
                    if len(rs) == 2 and rs[0].result != rs[1].result), key=lambda g: (g[0], g[1]))
    open_, sids = {}, []
    for date, gid, r0, r1 in games:
        key = frozenset((r0.team, r1.team))
        prev = open_.get(key)
        restart = prev is None or date - prev[2] > pd.Timedelta(hours=12) or (
            pd.notna(r0.game) and pd.notna(prev[1]) and r0.game <= prev[1])
        sid = gid if restart else prev[0]
        open_[key] = (sid, r0.game, date)
        sids.append(sid)
    return games, sids


def backtest(year: int, reps: int, seed: int, off: float) -> None:
    """Replay a past Worlds' real matchups and results with stats drawn from the
    teams' pre-Worlds games, and see where the real numbers fall."""
    df = load_raw(year)
    start = worlds_window(df)
    pmap, elo0 = elo_replay(df, start)
    real_recs = build_records(df[(df["league"] == "WLDs") & (df["date"] >= start)])
    games, sids = actual_games(real_recs)
    teams = sorted({r.team for r in real_recs})
    lo = start - pd.Timedelta(days=POOL_DAYS)
    pool_recs = build_records(df[(df["league"] != "WLDs") & (df["date"] >= lo) & (df["date"] < start)
                                 & df["teamname"].isin(teams)], pmap)
    pools = Pools(pool_recs, start)
    print(f"Backtest Worlds {year}: main event from {start.date()}, {len(games)} games, "
          f"{len(set(sids))} series, {len(teams)} teams; pool = {len(pool_recs)} games "
          f"from {lo.date()} (starter games kept: {sum(pools.size(t) for t in teams)})")
    short = [t for t in teams if pools.size(t) < MIN_POOL]
    if short:
        print(f"  thin pools: {', '.join(f'{t} ({pools.size(t)})' for t in short)}")

    real = Acc()
    for _, _, r0, r1 in games:
        wr, lr = (r0, r1) if r0.result == 1 else (r1, r0)
        real.game(wr.team, wr, lr, wr.length)
    truth = real.answers()

    rng = np.random.default_rng(seed)
    t = Tally()
    for _ in range(reps):
        locks: dict = defaultdict(set)
        acc = Acc()
        for (_, _, r0, r1), s in zip(games, sids):
            p_a = float(calibrate.prob_from_gap(elo0.get(r0.team, 1500.0) - elo0.get(r1.team, 1500.0)))
            play_game(pools, rng, [acc], r0.team, r1.team, r0.result == 1, locks[s], off, p_a)
        tally_sim(t, acc, None, None)

    print(f"\n{'quantity':<28}{'actual':>8}    model p10 / p50 / p90     in band?")
    flags = []
    nums = [("unique champions", truth["unique_champs"], "unique_champs"),
            ("share of games < 32:00", real.under / real.n, "under_share"),
            ("median length (min)", float(np.median(real.lens)), "median_len"),
            ("shortest game (min)", real.shortest[0], "shortest_len"),
            ("games with a 15+ kill pro", real.g15, "g15"),
            ("games with a 450+ CS pro", real.g450, "g450")]
    for label, actual, key in nums:
        lo_, md_, hi_ = np.percentile(t.num[key], [10, 50, 90])
        ok = lo_ <= actual <= hi_
        flags.append(ok)
        print(f"{label:<28}{actual:>8.2f}    {lo_:>7.2f} / {md_:>6.2f} / {hi_:>6.2f}     {'yes' if ok else 'NO'}")
    print(f"\n{'leader question':<16}{'actual':<32}{'model P':>8}{'rank':>6}   model's top 3")
    for q in ("presence", "champ_wr", "champ_deaths", "kda", "damage", "kp", "shortest_game"):
        d = t.dist(q)
        ranked = sorted(d.items(), key=lambda kv: -kv[1])
        p, rank = 0.0, "-"
        for a in truth[q]:
            if d.get(a, 0) > p:
                p, rank = d[a], 1 + [k for k, _ in ranked].index(a)
        top = ", ".join(f"{k} {v:.0%}" for k, v in ranked[:3])
        print(f"{q:<16}{', '.join(map(str, truth[q]))[:30]:<32}{p:>8.1%}{str(rank):>6}   {top}")
    print(f"\nactual: {real.under} games under 32:00 vs {real.over} over -> {truth['length_32']}; "
          f"model P(before)={t.dist('length_32').get('Before 32:00', 0):.0%}")
    gap = float(np.median(real.lens)) - float(np.mean(t.num["median_len"]))
    print(f"suggested [model] length_offset_min for this year: {gap:+.2f} "
          f"(pool drew {pools.clashes} Fearless clashes, {sum(pools.fallbacks.values())} result fallbacks)")
    print(f"numeric quantities inside the 10-90% band: {sum(flags)}/{len(flags)}")


# ------------------------------------------------------------------ report

def smooth_mvp(cfg: dict, sim: dict) -> dict:
    """Add-one-smoothed history, nudged by who the simulated champion's star is."""
    counts = cfg["model"]["mvp_role"]
    tot = sum(counts.values()) + len(ROLES)
    hist = {r: (counts.get(r, 0) + 1) / tot for r in ROLES}
    return {r: 0.75 * hist[r] + 0.25 * sim.get(r, 0.0) for r in ROLES}


def unique_counts(t: Tally, cfg: dict, key: str = "unique_champs") -> np.ndarray:
    """Simulated distinct-champion counts, scaled and with the year-level error the
    backtests show (the pool misses by about +-4 either way)."""
    m = cfg["model"]
    noise = np.random.default_rng(1).normal(0, m["unique_sd"], t.n)
    return np.round(np.array(t.num[key]) * m["unique_scale"] + noise).astype(int)


def build_answers(t: Tally, cfg: dict) -> dict[str, dict]:
    """question -> {answer: probability}."""
    out = {q: t.dist(q) for q in ("presence", "champ_wr", "champ_deaths", "peyz_guma", "faker_700",
                                  "length_32", "kills_15", "cs_450", "regions_ko", "lcs_ko",
                                  "hle_t1", "shortest_game")}
    # The shortest game goes to whoever wins it, so deep runs matter: in all of Worlds
    # 2017-25 its winner won 10% of the event's games on average (6 of 9 years a top-3
    # team by wins), while the pool simulation gave the actual winner 0% in 2022-25.
    out["shortest_game"] = t.dist("wins_share")
    for q in ("kda", "damage", "kp"):
        m = cfg["model"]["other_mass"][q]
        out[q] = {a: p * (1 - m) for a, p in t.dist(q).items()}
        if m:
            out[q]["Other (one-game sub)"] = m
    k15 = cfg["model"]["kills_15"]       # the simulation overshoots Worlds' rate several-fold
    r15 = (k15["worlds_hits"] + 0.5) / (k15["worlds_games"] + 1)
    p15 = float(np.mean([1 - (1 - r15) ** g for g in t.num["games"]]))
    out["kills_15"] = {"Yes": p15, "No": 1 - p15}
    p_no = max(out["cs_450"].get("No", 0.0), cfg["model"]["cs_450_no_floor"])   # Worlds 2020 had none
    out["cs_450"] = {"Yes": 1 - p_no, "No": p_no}
    tm = cfg["model"]["teemo"]
    rate = (tm["worlds_picks"] + tm["top_picks"] + 0.5) / (tm["worlds_games"] + tm["top_games"])
    p_teemo = float(np.mean([1 - (1 - rate) ** g for g in t.num["games"]]))
    out["teemo"] = {"Yes": p_teemo, "No": 1 - p_teemo}
    pmf = Counter(unique_counts(t, cfg).tolist())
    out["unique_champs"] = {k: v / t.n for k, v in sorted(pmf.items())}
    out["mvp_role"] = smooth_mvp(cfg, t.dist("mvp_sim"))
    return out


def report(t: Tally, cfg: dict, year: int, overrides: dict | None = None) -> pd.DataFrame:
    answers = build_answers(t, cfg)
    overrides = overrides or {}
    pool_view = {q: answers[q] for q in overrides}
    answers.update(overrides)
    pts = cfg["points"]
    rows = []
    print(f"\n=== WORLDS {year} CRYSTAL BALL ({t.n:,} simulated tournaments) ===")
    print(f"games per tournament: mean {np.mean(t.num['games']):.0f}, "
          f"p10-p90 {np.percentile(t.num['games'], 10):.0f}-{np.percentile(t.num['games'], 90):.0f}"
          f"{'' if cfg['rules']['include_playin'] else ' (Play-In excluded)'}")
    for q, text in QUESTIONS.items():
        d = answers[q]
        ranked = sorted(d.items(), key=lambda kv: -kv[1])
        best = next(a for a, _ in ranked if not str(a).startswith("Other"))   # a named pro must be picked
        eff = d
        if q == "hle_t1":      # a shared placement is unresolved: split it between the two
            eff = {"HLE": d.get("HLE", 0) + d.get("Same placement", 0) / 2,
                   "T1": d.get("T1", 0) + d.get("Same placement", 0) / 2}
            best = max(eff, key=eff.get)
        print(f"\n[{pts[q]:>3} pts] {text}")
        if q == "unique_champs":
            vals = unique_counts(t, cfg)
            seen = unique_counts(t, cfg, "unique_seen")
            print(f"   picked only; counting bans too it would be ~{np.median(seen):.0f} "
                  f"(sources disagree on which one counts)")
            print(f"   mean {vals.mean():.1f}, median {np.median(vals):.0f}, "
                  f"p10-p90 {np.percentile(vals, 10):.0f}-{np.percentile(vals, 90):.0f}")
            bins = Counter(int(v // 5) * 5 for v in vals)
            print("   " + "  ".join(f"{b}-{b + 4}: {c / len(vals):.0%}"
                                    for b, c in sorted(bins.items()) if c / len(vals) >= 0.02))
            print("   most likely exact counts: " + ", ".join(f"{k} ({v:.0%})" for k, v in ranked[:5]))
        else:
            for a, p in ranked[:6]:
                print(f"   {str(a):<28}{p:>7.1%}   EV {p * pts[q]:5.1f}")
            if q == "shortest_game":
                pool = sorted(t.dist("shortest_game").items(), key=lambda kv: -kv[1])[:3]
                print("   (share of the event's wins; the pool simulation alone, 0 for 4 in backtests: "
                      + ", ".join(f"{a} {p:.0%}" for a, p in pool) + ")")
            if q in overrides:
                top_pool = sorted(pool_view[q].items(), key=lambda kv: -kv[1])[:3]
                print("   (team-pool simulation alone, unreliable here: "
                      + ", ".join(f"{a} {p:.0%}" for a, p in top_pool) + ")")
            if q == "hle_t1":
                print(f"   (shared placement split: HLE {eff['HLE']:.1%} / T1 {eff['T1']:.1%})")
        print(f"   -> pick: {best}")
        for a, p in d.items():
            if q == "unique_champs" or p >= 0.002:
                rows.append({"question": q, "answer": str(a), "probability": round(p, 4),
                             "points": pts[q], "expected_points": round(eff.get(a, p) * pts[q], 2),
                             "pick": str(a) == str(best)})
    df = pd.DataFrame(rows)
    ev_total = df[df["pick"]].groupby("question")["expected_points"].max().sum()
    print(f"\nExpected points if every recommended pick is entered: {ev_total:.0f} of {sum(pts.values())}")
    return df


def run_live(year: int, n: int, seed: int, include_playin: bool, off: float) -> pd.DataFrame:
    cfg = load_config()
    cfg["rules"]["include_playin"] = include_playin
    ev = swiss.resolve(swiss.load_config(), swiss.load_series(year), year)
    start = pd.Timestamp(ev.bcfg["start"])
    lo = start - pd.Timedelta(days=POOL_DAYS)
    df = load_raw(year)
    teams = set(ev.rows["data_name"])
    pmap, _ = elo_replay(df, start)
    recs = build_records(df[(df["league"] != "WLDs") & (df["date"] >= lo) & (df["date"] < start)
                            & df["teamname"].isin(teams)], pmap)
    pools = Pools(recs, start)
    sig = meta_signals(df, start, teams, cfg["model"]["meta_presence"]["newest_weight"])
    overrides = {"presence": presence_leaders(sig, cfg), "champ_deaths": deaths_leaders(df, start, cfg),
                 "champ_wr": wr_lottery(df, start, sig, cfg)}
    print(f"Presence signal: all leagues on patch >= {sig.attrs['patch']} (newest, {sig.attrs['latest']}, "
          f"counted x{cfg['model']['meta_presence']['newest_weight']:g}), plus the event teams' own games")

    print(f"Pool: {len(recs)} games since {lo.date()} for {len(teams)} teams (event starts {start.date()})")
    print(f"\n{'team':<22}{'pool':>5}   top / jng / mid / bot / sup")
    for _, r in ev.rows.iterrows():
        st = pools.starters.get(r["data_name"], [])
        print(f"{r['team']:<22}{pools.size(r['data_name']):>5}   {' / '.join(st) or 'NO DATA'}")

    rng = np.random.default_rng(seed)
    t = Tally()
    for i in range(n):
        acc, facts = simulate_event(ev, pools, rng, off + rng.normal(0, cfg["model"]["length_offset_sd_min"]),
                                    include_playin, cfg)
        tally_sim(t, acc, facts, cfg)
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1:,} / {n:,}")
    print(f"Fearless clashes accepted: {pools.clashes}; result fallbacks: {dict(pools.fallbacks) or 'none'}")
    champ = sorted(t.dist("champion").items(), key=lambda kv: -kv[1])[:4]
    print("title odds from this run: " + ", ".join(f"{k} {v:.1%}" for k, v in champ))
    return report(t, cfg, year, overrides)


def main() -> None:
    global HALF_LIFE, BANDWIDTH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--n", type=int, default=4000, help="simulated tournaments")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backtest", type=int, metavar="YEAR", help="replay a past Worlds instead")
    ap.add_argument("--reps", type=int, default=1000, help="backtest replays")
    ap.add_argument("--no-playin", action="store_true", help="Play-In games do not count")
    ap.add_argument("--half-life", type=float, default=None, help="days (default %g)" % HALF_LIFE)
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="matchup-closeness kernel width; 0 turns closeness matching off (default %g)" % BANDWIDTH)
    ap.add_argument("--length-offset", type=float, default=None,
                    help="minutes added to every game (default: [model] length_offset_min)")
    args = ap.parse_args()
    if args.half_life is not None:
        HALF_LIFE = args.half_life
    if args.bandwidth is not None:
        BANDWIDTH = args.bandwidth
    cfg = load_config()
    off = cfg["model"]["length_offset_min"] if args.length_offset is None else args.length_offset
    if args.backtest:
        backtest(args.backtest, args.reps, args.seed, off)
        return
    df = run_live(args.year, args.n, args.seed,
                  cfg["rules"]["include_playin"] and not args.no_playin, off)
    paths.ensure_dirs()
    out = paths.processed(args.year, "crystal_ball", "csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
