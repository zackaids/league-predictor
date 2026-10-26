"""Board data for the Worlds page.

Turns facts and odds into the dict `board.render` draws. Pure functions with no
Streamlit import, so the shapes can be checked without running the app.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import bracket
import monte_carlo as mc

SOURCES = {"auto": "from Oracle's Elixir", "manual": "entered by hand in the config",
           "draw": "inferred from the next draw", "live": "from lolesports.com"}


@dataclass
class Teams:
    """Per-team display facts, keyed by the name shown in the app."""
    short: dict[str, str]
    league: dict[str, str]
    link: set[str]          # teams that have a Team page

    @classmethod
    def from_rows(cls, rows: pd.DataFrame, rated: set[str] | None = None) -> "Teams":
        link = set(rows["team"]) if rated is None else {
            t for t, d in zip(rows["team"], rows["data_name"]) if d in rated}
        return cls(dict(zip(rows["team"], rows["short"])),
                   dict(zip(rows["team"], rows["home_league"])), link)

    def row(self, team: str, p: float | None = None, state: str = "open",
            score: str | None = None) -> dict:
        return {"team": team, "short": self.short[team], "league": self.league[team],
                "p": p, "state": state, "score": score, "link": team in self.link}


# ------------------------------------------------------------------ Play-In

PLAYIN_COLUMNS = [
    ("Opening", ["O1", "O2"], False),
    ("Upper & lower", ["UB", "LB"], True),
    ("Lower final", ["LF"], True),
    ("Qualifier", ["QF"], True),
]
PLAYIN_NAMES = {"O1": "Opening 1", "O2": "Opening 2", "UB": "Upper bracket",
                "LB": "Lower bracket", "LF": "Lower final", "QF": "Qualifying match"}
PLAYIN_STAKES = {
    "O1": "loser drops to the lower bracket",
    "O2": "loser drops to the lower bracket",
    "UB": "winner to the qualifier, loser to the lower final",
    "LB": "loser out (19th)",
    "LF": "loser out (18th), winner to the qualifier",
    "QF": "winner advances to Swiss, loser 17th",
}
# (from, to, which finisher moves on): winners solid, losers dashed.
PLAYIN_EDGES = [("O1", "UB", "win"), ("O2", "UB", "win"), ("O1", "LB", "lose"), ("O2", "LB", "lose"),
                ("UB", "QF", "win"), ("UB", "LF", "lose"), ("LB", "LF", "win"), ("LF", "QF", "win")]


def playin_board(teams: Teams, pairs: dict, real: dict, solved: dict,
                 rating: dict[str, float], picks: dict[str, str]) -> dict:
    """The Play-In bracket: results, open series with odds and picks, and for a
    series whose teams are not decided yet, who could end up there."""
    columns = []
    for title, ids, center in PLAYIN_COLUMNS:
        boxes = []
        for m in ids:
            pair = pairs[m]
            if pair is None:
                likely = sorted(solved["appear"][m].items(), key=lambda kv: -kv[1])
                boxes.append({"title": PLAYIN_NAMES[m], "subtitle": "Bo5", "matches": [],
                              "likely": [teams.row(t, p) for t, p in likely if p > 0.0005],
                              "likely_caption": "Chance each team plays here"})
                continue
            match = {"id": m, "tag": PLAYIN_NAMES[m], "stake": "Bo5", "pickable": False}
            if m in real:
                win = real[m]["winner"]
                score = real[m]["score"].split("-") if real[m]["score"] else ["W", "L"]
                match["rows"] = [teams.row(t, state="won" if t == win else "lost",
                                           score=score[0] if t == win else score[1])
                                 for t in pair]
                match["footer"] = f"Final · {SOURCES[real[m]['source']]}"
                match["decided"] = True
            else:
                p_a = bracket.p_match(rating[pair[0]], rating[pair[1]])
                match["rows"] = [teams.row(t, p, state="picked" if picks.get(m) == t else "open")
                                 for t, p in zip(pair, (p_a, 1 - p_a))]
                match["footer"] = PLAYIN_STAKES[m]
                match["pickable"] = True
            boxes.append({"matches": [match]})
        columns.append({"title": title, "center": center, "boxes": boxes})
    return {"columns": columns,
            "edges": [{"from": a, "to": b, "kind": k} for a, b, k in PLAYIN_EDGES]}


# ------------------------------------------------------------------ Swiss

MIN_SHOWN = 0.005       # a team less likely than this to be somewhere is left off the list
FINAL_BOXES = [("Advance", [(3, 0), (3, 1), (3, 2)]), ("Eliminated", [(2, 3), (1, 3), (0, 3)])]


def prob_fn(field: mc.Field):
    """prob(a, b, best_of) -> P(a wins a best-of-`best_of` against b)."""
    m, ix = mc._matrices(field), field.index
    return lambda a, b, bo: float(m[bo][ix[a], ix[b]])


def _likely(teams: Teams, names: list[str], probs, floor: float = MIN_SHOWN) -> list[dict]:
    order = sorted(range(len(names)), key=lambda i: -probs[i])
    return [teams.row(names[i], float(probs[i])) for i in order if probs[i] > floor]


def _stake(w: int, l: int) -> str:
    if w == 2 and l == 2:
        return "win to advance, lose and you're out"
    return "win to advance" if w == 2 else "lose and you're out" if l == 2 else ""


def _result_rows(teams: Teams, pair, winner: str, score: str | None) -> list[dict]:
    parts = score.split("-") if score else ["W", "L"]
    return [teams.row(t, state="won" if t == winner else "lost",
                      score=parts[0] if t == winner else parts[1]) for t in pair]


def swiss_board(teams: Teams, event, stt, odds: mc.Odds) -> dict:
    """Round 1-5 columns by record, then the final records.

    A drawn round shows its matches (odds, picks, results); a round not drawn yet
    shows who is likely to be on each record, and an action to draw it once the
    round before is decided.
    """
    field, known, prob = stt.field, stt.known, prob_fn(stt.field)
    names = field.teams
    nxt = mc.next_undrawn(known)
    last_hypo = max(stt.hypo_draws) if stt.hypo_draws else None
    columns = []

    for r in range(1, mc.ROUNDS + 1):
        rec_in = {t: (0, 0) for t in names}
        rec_in.update(swiss_records(known, r - 1))
        boxes = []

        if r == nxt or (r == last_hypo and r not in event.known.draws):
            ready = swiss_premise_ok(known, r)
            label = f"Draw round {r}" if r == nxt else f"Redraw round {r}"
            hint = None if ready else ("Decide the Play-In first" if r == 1 else f"Decide round {r - 1} first")
            boxes.append({"action": {"id": f"draw:{r}", "label": label, "disabled": not ready, "hint": hint}})
        if r in stt.hypo_draws:
            boxes.append({"note": "Hypothetical draw — yours alone, not saved."})

        if r in known.draws:
            real_info = event.info["swiss"].get(r, {})
            for w, l, _ in mc._layout(r):
                pairs = [p for p in known.draws[r] if rec_in[p[0]] == (w, l)]
                cards = []
                for a, b in pairs:
                    bo = 3 if (w == 2 or l == 2) else 1
                    match = {"id": f"S{r}:{a} v {b}", "tag": f"Bo{bo}", "stake": _stake(w, l), "pickable": False}
                    if (a, b) in event.known.results.get(r, {}):
                        info = real_info.get((a, b), {})
                        match["rows"] = _result_rows(teams, (a, b), event.known.results[r][(a, b)], info.get("score"))
                        match["footer"] = f"Final · {SOURCES.get(info.get('source'), 'result')}"
                        match["decided"] = True
                    else:
                        p_a = prob(a, b, bo)
                        picked = known.results[r].get((a, b))
                        match["rows"] = [teams.row(t, p, state="picked" if picked == t else "open")
                                         for t, p in ((a, p_a), (b, 1 - p_a))]
                        match["pickable"] = True
                    cards.append(match)
                boxes.append({"title": f"{w}-{l}", "subtitle": f"Bo{3 if (w == 2 or l == 2) else 1}",
                              "matches": cards})
        elif r == 1:
            boxes += _round1_boxes(teams, field, odds)
        else:
            for w, l, size in mc._layout(r):
                boxes.append({"title": f"{w}-{l}", "subtitle": f"Bo{3 if (w == 2 or l == 2) else 1}",
                              "matches": [], "likely": _likely(teams, names, odds.occupancy[r][(w, l)]),
                              "likely_caption": "Chance each team is on this record"})
        columns.append({"title": f"Round {r}", "boxes": boxes})

    done = _decided_all(known)
    final_boxes = []
    rec = swiss_records(known)
    for title, recs in FINAL_BOXES:
        for w, l in recs:
            i = mc.RECORDS.index((w, l))
            if done:
                rows = [teams.row(t) for t in sorted(names, key=lambda t: -field.rating[field.index[t]])
                        if rec.get(t) == (w, l)]
                final_boxes.append({"title": f"{w}-{l}", "subtitle": title.lower(), "matches": [], "likely": rows})
            else:
                final_boxes.append({"title": f"{w}-{l}", "subtitle": title.lower(), "matches": [],
                                    "likely": _likely(teams, names, odds.record[:, i]),
                                    "likely_caption": "Chance of finishing here"})
    columns.append({"title": "Final record", "boxes": final_boxes})
    return {"columns": columns, "edges": []}


def swiss_records(known: mc.Known, upto: int = mc.ROUNDS) -> dict[str, tuple[int, int]]:
    rec: dict[str, list[int]] = {}
    for r in range(1, upto + 1):
        for (a, b), w in known.results.get(r, {}).items():
            lo = b if w == a else a
            rec.setdefault(w, [0, 0])[0] += 1
            rec.setdefault(lo, [0, 0])[1] += 1
    return {t: (w, l) for t, (w, l) in rec.items()}


def _decided_all(known: mc.Known) -> bool:
    return all(r in known.draws and len(known.results.get(r, {})) == len(known.draws[r])
               for r in range(1, mc.ROUNDS + 1))


def swiss_premise_ok(known: mc.Known, r: int) -> bool:
    """Can round `r` be drawn: is everything before it decided?"""
    if r == 1:
        return known.playin_winner is not None
    return r - 1 in known.draws and len(known.results.get(r - 1, {})) == len(known.draws[r - 1])


def _round1_boxes(teams: Teams, field: mc.Field, odds: mc.Odds) -> list[dict]:
    """Before the draw: tier 1 plays tier 4 and tier 2 plays tier 3. For each team in
    the higher tier, who it could draw."""
    names, meet = field.teams, odds.meet[1]
    tier = {t: field.tier[i] for i, t in enumerate(names[:mc.DIRECT])}
    for t in names[mc.DIRECT:]:
        tier[t] = 4
    boxes = []
    for hi, lo in ((1, 4), (2, 3)):
        cards = []
        for a in (t for t in names[:mc.DIRECT] if tier[t] == hi):
            i = field.index[a]
            opp = [(names[j], meet[i, j]) for j in range(len(names)) if tier[names[j]] == lo and meet[i, j] > MIN_SHOWN]
            opp.sort(key=lambda kv: -kv[1])
            cards.append({"id": f"R1:{a}", "tag": teams.short[a], "stake": "draws one of", "pickable": False,
                          "rows": [teams.row(t, float(p)) for t, p in opp]})
        boxes.append({"title": f"Tier {hi} v tier {lo}", "subtitle": "Bo1", "matches": cards})
    return boxes


# ------------------------------------------------------------------ knockout

KO_NAMES = {"QF1": "Quarterfinal 1", "QF2": "Quarterfinal 2", "QF3": "Quarterfinal 3",
            "QF4": "Quarterfinal 4", "SF1": "Semifinal 1", "SF2": "Semifinal 2", "F": "Final"}
KO_EDGES = [("QF1", "SF1", "win"), ("QF2", "SF1", "win"), ("QF3", "SF2", "win"),
            ("QF4", "SF2", "win"), ("SF1", "F", "win"), ("SF2", "F", "win")]


def knockout_board(teams: Teams, event, stt, odds: mc.Odds) -> dict:
    """Quarterfinals -> semifinals -> final, Bo5. Before the quarterfinal draw, who
    could be in each series; after it, odds, picks and results."""
    field, known, prob = stt.field, stt.known, prob_fn(stt.field)
    qf_ready = known.qf is not None or swiss_premise_ok(known, mc.ROUNDS)
    won: dict[str, str] = dict(known.ko)

    def card(key: str) -> dict:
        j = mc.KO_SERIES.index(key)
        pair = None
        if known.qf is not None:
            if key.startswith("QF"):
                pair = tuple(known.qf[int(key[2]) - 1])
            else:
                feeds = {"SF1": ("QF1", "QF2"), "SF2": ("QF3", "QF4"), "F": ("SF1", "SF2")}[key]
                pair = tuple(won.get(f) for f in feeds)
        if pair is None or None in pair:
            return {"title": KO_NAMES[key], "subtitle": "Bo5", "matches": [],
                    "likely": _likely(teams, field.teams, odds.ko_slot[:, j]),
                    "likely_caption": "Chance each team plays here"}
        match = {"id": key, "tag": KO_NAMES[key], "stake": "Bo5", "pickable": False}
        if key in event.known.ko:
            info = event.info["ko"].get(key, {})
            match["rows"] = _result_rows(teams, pair, event.known.ko[key], info.get("score"))
            match["footer"] = f"Final · {SOURCES.get(info.get('source'), 'result')}"
            match["decided"] = True
        else:
            p_a = prob(pair[0], pair[1], 5)
            match["rows"] = [teams.row(t, p, state="picked" if known.ko.get(key) == t else "open")
                             for t, p in zip(pair, (p_a, 1 - p_a))]
            match["pickable"] = True
        return {"matches": [match]}

    qf_boxes = []
    if known.qf is None or stt.hypo_qf:
        action = {"id": "draw:qf", "label": "Redraw quarterfinals" if stt.hypo_qf else "Draw quarterfinals",
                  "disabled": not qf_ready, "hint": None if qf_ready else "Decide the Swiss stage first"}
        qf_boxes.append({"action": action})
    if stt.hypo_qf:
        qf_boxes.append({"note": "Hypothetical draw — yours alone, not saved."})
    columns = [
        {"title": "Quarterfinals", "boxes": qf_boxes + [card(k) for k in mc.KO_SERIES[:4]]},
        {"title": "Semifinals", "center": True, "boxes": [card("SF1"), card("SF2")]},
        {"title": "Final", "center": True, "boxes": [card("F")]},
    ]
    return {"columns": columns, "edges": [{"from": a, "to": b, "kind": k} for a, b, k in KO_EDGES]}


def live_stage(known: mc.Known) -> str:
    """The stage worth opening on, from the real facts: where the tournament is."""
    if known.playin_winner is None:
        return "Play-In"
    return "Swiss" if not _decided_all(known) else "Knockout"
