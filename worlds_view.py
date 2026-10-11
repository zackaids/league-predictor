"""Board data for the Worlds page.

Turns facts and odds into the dict `board.render` draws. Pure functions with no
Streamlit import, so the shapes can be checked without running the app.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import bracket

SOURCES = {"auto": "from Oracle's Elixir", "manual": "entered by hand in the config",
           "draw": "inferred from the next draw"}


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
