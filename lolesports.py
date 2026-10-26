"""Worlds schedule: what lolesports.com knows that the configs do not.

`worlds_bracket.toml` and `worlds_swiss.toml` are hand-entered, and Oracle's
Elixir lags the real event by hours. The schedule behind lolesports.com has each
Play-In and Swiss series the moment it is drawn, with scores once it is played.

As a CLI this reads it, compares it with the configs and the series log, and
prints the TOML to paste for anything new. It never writes a file.

    python lolesports.py            # print what to add, and any conflicts
    python lolesports.py --json     # dump the raw schedule events instead
    python lolesports.py --file F   # read a saved schedule (--json output) instead of the network

Exit status: 0 in sync or only additions, 1 a conflict with the configs,
2 the schedule could not be fetched.

As a library it feeds the Worlds odds and the app: `live_facts` turns the events
into a `swiss.Live` that `swiss.resolve_live` merges under the config, and
`schedule` / `save_snapshot` / `load_snapshot` keep the pipeline's copy of the
schedule (`{year}_worlds_schedule.json`), the app's fallback when the API is down.
Only the pipeline (`swiss.write_outputs`) writes that snapshot; the CLI does not.

The quarterfinal draw comes back in start-time order. Riot does not say which
slot is QF1 (QF1+QF2 feed SF1), so confirm that order against the official bracket.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import bracket
import monte_carlo as mc
import paths
import swiss
from paths import CURRENT_YEAR

API = "https://esports-api.lolesports.com/persisted/gw/getSchedule"
# The key lolesports.com's own front end sends; it is public, not a secret.
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
WORLDS_LEAGUE_ID = "98767975604431411"
MAX_PAGES = 6

BLOCKS = {"Play-Ins", "Swiss", "Quarterfinals", "Semifinals", "Finals"}
KO_BLOCKS = BLOCKS - {"Play-Ins", "Swiss"}
FEEDS = {"SF1": ("QF1", "QF2"), "SF2": ("QF3", "QF4"), "F": ("SF1", "SF2")}
# The API's tricode where it differs from the config's `short`.
CODE_ALIASES = {"TLAW": "TL"}


# ------------------------------------------------------------------ fetch

def _page(token: str | None, timeout: float = 30) -> dict:
    query = {"hl": "en-US", "leagueId": WORLDS_LEAGUE_ID}
    if token:
        query["pageToken"] = token
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(query)}",
                                 headers={"x-api-key": API_KEY})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["data"]["schedule"]


def fetch_events(since: str, timeout: float = 30) -> list[dict]:
    """Every schedule event from the newest page back to `since` (an ISO date)."""
    events: dict[str, dict] = {}
    token = None
    for _ in range(MAX_PAGES):
        page = _page(token, timeout)
        for e in page["events"]:
            if e.get("match"):
                events[e["match"]["id"]] = e
        token = (page.get("pages") or {}).get("older")
        if not token or min((e["startTime"] for e in page["events"]), default=since) < since:
            break
    return sorted(events.values(), key=lambda e: e["startTime"])


# ------------------------------------------------------------------ parse

@dataclass
class Match:
    start: str
    block: str
    teams: tuple          # config names; None for an undrawn slot
    wins: tuple
    done: bool

    @property
    def winner(self) -> str | None:
        return self.teams[0] if self.wins[0] > self.wins[1] else self.teams[1] if self.done else None

    @property
    def score(self) -> str:
        return f"{max(self.wins)}-{min(self.wins)}"


def parse(events: list[dict], since: str, names: dict[str, str]) -> list[Match]:
    out = []
    for e in events:
        if e.get("blockName") not in BLOCKS or e["startTime"] < since:
            continue
        teams, wins, outcomes = [], [], []
        for t in e["match"]["teams"]:
            code = CODE_ALIASES.get(t["code"], t["code"])
            if code == "TBD":
                teams.append(None)
            elif code in names:
                teams.append(names[code])
            else:
                raise ValueError(f"unknown team code {t['code']!r} ({t['name']!r}) in the schedule -- "
                                 f"add it to the configs or to CODE_ALIASES")
            res = t.get("result") or {}
            wins.append(res.get("gameWins") or 0)
            outcomes.append(res.get("outcome"))
        done = e["state"] == "completed" and None not in teams and "win" in outcomes
        out.append(Match(e["startTime"], e["blockName"], tuple(teams), tuple(wins), done))
    return out


# ------------------------------------------------------------------ compare

class Report:
    def __init__(self):
        self.adds: list[tuple[str, list[str]]] = []
        self.conflicts: list[str] = []
        self.notes: list[str] = []

    def add(self, target: str, lines: list[str]) -> None:
        self.adds.append((target, lines))

    def result(self, a: str, b: str, w: str, score: str) -> str:
        return f'"{a} v {b}" = {{ winner = "{w}", score = "{score}" }}'


def check_playin(rep: Report, bcfg: dict, matches: list[Match], series) -> None:
    pool = [m for m in matches if m.block == "Play-Ins" and m.done]
    known = bracket.resolve_results(bcfg, series)
    won: dict[str, str] = {}
    lost: dict[str, str] = {}
    lines = []
    for mid in bracket.ORDER:
        pair = bracket._pair(mid, bcfg["opening"], won, lost)
        hit = next((m for m in pool if set(m.teams) == set(pair)), None) if pair else None
        if hit is None:
            break
        pool.remove(hit)
        won[mid], lost[mid] = hit.winner, pair[0] if hit.winner == pair[1] else pair[1]
        if mid not in known:
            lines.append(f'{mid} = {{ winner = "{hit.winner}", score = "{hit.score}" }}')
        elif known[mid]["winner"] != hit.winner:
            rep.conflicts.append(f"Play-In {mid} ({pair[0]} v {pair[1]}): the config/log says "
                                 f"{known[mid]['winner']} won, lolesports says {hit.winner}")
    if lines:
        rep.add("worlds_bracket.toml -> under [results]", lines)


def swiss_rounds(matches: list[Match]) -> tuple[dict[int, list[Match]], int]:
    """The Swiss series split into rounds by the format's sizes, and how many the format has."""
    ms = [m for m in matches if m.block == "Swiss"]
    rounds, i = {}, 0
    for r in range(1, mc.ROUNDS + 1):
        n = sum(s for _, _, s in mc._layout(r)) // 2
        rounds[r] = ms[i:i + n]
        i += n
    return rounds, i


def walk_swiss(matches: list[Match]) -> list[tuple[int, list[Match], dict[str, tuple[int, int]]]]:
    """(round, its series, every team's record going in) for each round the schedule has drawn.

    Stops before a round with an undrawn slot or a missing series, and after the first
    round that is not finished -- later records would not be known. Records come only
    from the schedule's own results.
    """
    rounds, _ = swiss_rounds(matches)
    rec: dict[str, tuple[int, int]] = {}
    out = []
    for r, ms in rounds.items():
        if len(ms) < sum(s for _, _, s in mc._layout(r)) // 2 or any(None in m.teams for m in ms):
            break
        out.append((r, ms, {t: rec.get(t, (0, 0)) for m in ms for t in m.teams}))
        if not all(m.done for m in ms):
            break
        for m in ms:
            w = m.winner
            lo = m.teams[1] if w == m.teams[0] else m.teams[0]
            rec[w] = (rec.get(w, (0, 0))[0] + 1, rec.get(w, (0, 0))[1])
            rec[lo] = (rec.get(lo, (0, 0))[0], rec.get(lo, (0, 0))[1] + 1)
    return out


def check_swiss(rep: Report, cfg: dict, ev: swiss.Event | None, matches: list[Match]) -> None:
    _, expected = swiss_rounds(matches)
    total = sum(m.block == "Swiss" for m in matches)
    if total != expected:
        rep.notes.append(f"the schedule has {total} Swiss series, the format has {expected}")

    for r, ms, rec_in in walk_swiss(matches):
        groups: dict[tuple, list] = {}
        for m in ms:
            a, b = m.teams
            ra, rb = rec_in[a], rec_in[b]
            if ra != rb:
                rep.conflicts.append(f"Swiss round {r}: {a} ({ra[0]}-{ra[1]}) v {b} ({rb[0]}-{rb[1]}) "
                                     f"are not on the same record")
            groups.setdefault(ra, []).append(m)

        if r in cfg["draws"]:
            have = {frozenset(p) for p in cfg["draws"][r]}
            api = {frozenset(m.teams) for m in ms}
            if have != api:
                rep.conflicts.append(f"Swiss round {r}: the config draw differs from lolesports -- "
                                     f"only in config: {sorted(sorted(p) for p in have - api)}, "
                                     f"only in lolesports: {sorted(sorted(p) for p in api - have)}")
            else:
                for g, gm in groups.items():
                    for m in gm:
                        for t in m.teams:
                            if cfg["labels"][r][t] != g:
                                rep.conflicts.append(f"Swiss round {r}: {t} is on {cfg['labels'][r][t][0]}-"
                                                     f"{cfg['labels'][r][t][1]} in the config, "
                                                     f"{g[0]}-{g[1]} by lolesports")
        else:
            lines = [f"[draws.{r}]"]
            for g in sorted(groups, key=lambda g: (-g[0], g[1])):
                lines.append(f'"{g[0]}-{g[1]}" = [')
                lines += [f'  ["{m.teams[0]}", "{m.teams[1]}"],' for m in groups[g]]
                lines.append("]")
            rep.add("worlds_swiss.toml -> new table", lines)

        res_lines = []
        info = ev.info["swiss"].get(r, {}) if ev else {}
        by_pair = {frozenset(p): v for p, v in info.items()}
        for m in ms:
            if not m.done:
                continue
            got = by_pair.get(frozenset(m.teams))
            if got is None:
                res_lines.append(rep.result(*m.teams, m.winner, m.score))
            elif got["winner"] != m.winner:
                rep.conflicts.append(f"Swiss round {r}: {m.teams[0]} v {m.teams[1]} -- the config/log says "
                                     f"{got['winner']} won, lolesports says {m.winner}")
        if res_lines:
            rep.add("worlds_swiss.toml -> under [results.swiss]", res_lines)


def api_quarterfinals(matches: list[Match]) -> list[tuple] | None:
    """QF1..QF4 in start-time order once all four are drawn, else None."""
    qm = sorted((m for m in matches if m.block == "Quarterfinals"), key=lambda m: m.start)
    return [m.teams for m in qm] if len(qm) == 4 and all(None not in m.teams for m in qm) else None


def check_knockout(rep: Report, cfg: dict, ev: swiss.Event | None, matches: list[Match]) -> None:
    api_qf = api_quarterfinals(matches)
    qf = cfg["quarterfinals"]
    if api_qf and qf is None:
        rep.add("worlds_swiss.toml -> under [knockout]", [
            "quarterfinals = [" + ", ".join(f'["{a}", "{b}"]' for a, b in api_qf) + "]",
            "# order = start time; confirm which is QF1..QF4 (QF1+QF2 feed SF1) against the official bracket"])
    elif api_qf and {frozenset(p) for p in qf} != {frozenset(p) for p in api_qf}:
        rep.conflicts.append("quarterfinals: the config pairs differ from lolesports")
    elif api_qf and [frozenset(p) for p in qf] != [frozenset(p) for p in api_qf]:
        rep.notes.append("quarterfinals: same pairs as lolesports but in a different slot order "
                         "(the schedule orders by start time; the config's order decides who feeds which semifinal)")

    pairs = qf or api_qf
    done = [m for m in matches if m.block in KO_BLOCKS and m.done]
    won: dict[str, str] = {}
    lines = []
    for key in mc.KO_SERIES:
        if key.startswith("QF"):
            pair = pairs[int(key[2]) - 1] if pairs else None
        else:
            a, b = (won.get(k) for k in FEEDS[key])
            pair = (a, b) if a and b else None
        hit = next((m for m in done if pair and set(m.teams) == set(pair)), None)
        if hit is None:
            continue
        won[key] = hit.winner
        have = ev.known.ko.get(key) if ev else None
        if have is None:
            lines.append(f'{key} = {{ winner = "{hit.winner}", score = "{hit.score}" }}')
        elif have != hit.winner:
            rep.conflicts.append(f"Knockout {key}: the config/log says {have} won, lolesports says {hit.winner}")
    if lines:
        rep.add("worlds_swiss.toml -> under [results.knockout]", lines)


# ------------------------------------------------------------------ facts

def live_facts(cfg: dict, bcfg: dict, events: list[dict]) -> swiss.Live:
    """What the schedule says, as the facts `swiss.merge_live` layers under the config.

    A Swiss round counts only when every series is on a single record, and nothing
    after a round that fails that. An unknown team code raises (from `parse`).
    """
    names = {t["short"]: t["name"] for t in cfg["teams"] + bcfg["teams"]}
    matches = parse(events, str(bcfg["start"]), names)
    live = swiss.Live()
    for r, ms, rec_in in walk_swiss(matches):
        if any(rec_in[a] != rec_in[b] for a, b in (m.teams for m in ms)):
            break
        live.draws[r] = [m.teams for m in ms]
        live.labels[r] = rec_in
        for m in ms:
            if m.done:
                live.results[frozenset(m.teams)] = {"winner": m.winner, "score": m.score}
    live.qf = api_quarterfinals(matches)
    for m in matches:
        if m.block in KO_BLOCKS and m.done:
            live.ko[frozenset(m.teams)] = {"winner": m.winner, "score": m.score}
    return live


# ------------------------------------------------------------------ snapshot

def snapshot_path(year: int) -> Path:
    return paths.processed(year, "worlds_schedule", "json")


def save_snapshot(year: int, events: list[dict], since: str) -> None:
    """Keep the stage events from `since` on, so the app has a schedule when the API is down."""
    kept = sorted((e for e in events if e.get("blockName") in BLOCKS and e["startTime"] >= since),
                  key=lambda e: e["startTime"])
    path = snapshot_path(year)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(kept, indent=1, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)


def load_snapshot(year: int) -> list[dict] | None:
    path = snapshot_path(year)
    return json.loads(path.read_text()) if path.exists() else None


def schedule(year: int, since: str) -> tuple[list[dict] | None, str]:
    """The schedule and where it came from: the API (saved as the snapshot), else the snapshot, else None."""
    try:
        events = fetch_events(since)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        events = load_snapshot(year)
        if events is None:
            return None, f"lolesports.com unreachable ({exc!r}); no snapshot, config only"
        return events, f"lolesports.com unreachable ({exc!r}); using the last snapshot"
    save_snapshot(year, events, since)
    return events, "live from lolesports.com (snapshot saved)"


# ------------------------------------------------------------------ main

def run(year: int, events: list[dict] | None = None) -> Report:
    cfg, bcfg = swiss.load_config(), bracket.load_config()
    names = {t["short"]: t["name"] for t in cfg["teams"] + bcfg["teams"]}
    since = str(bcfg["start"])
    if events is None:
        events = fetch_events(since)
    matches = parse(events, since, names)
    series = swiss.load_series(year)
    rep = Report()

    check_playin(rep, bcfg, matches, series)
    ev = None
    try:
        ev = swiss.resolve(cfg, series, year)
    except Exception as exc:     # a bad hand edit must be reported, not hide the rest
        rep.conflicts.append(f"{swiss.CONFIG.name} does not resolve: {exc}")
    check_swiss(rep, cfg, ev, matches)
    check_knockout(rep, cfg, ev, matches)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--json", action="store_true", help="dump the raw schedule events and exit")
    ap.add_argument("--file", help="read a saved schedule instead of the network")
    args = ap.parse_args()

    try:
        if args.file:
            with open(args.file) as f:
                events = json.load(f)
        else:
            events = fetch_events(str(bracket.load_config()["start"]))
        if args.json:
            print(json.dumps(events, indent=2))
            return 0
        rep = run(args.year, events)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Could not read the schedule: {exc!r}", file=sys.stderr)
        return 2

    for target, lines in rep.adds:
        print(f"# {target}")
        print("\n".join(lines))
        print()
    for n in rep.notes:
        print(f"NOTE: {n}")
    for c in rep.conflicts:
        print(f"CONFLICT: {c}")
    if not (rep.adds or rep.conflicts or rep.notes):
        print("In sync: nothing to add.")
    return 1 if rep.conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
