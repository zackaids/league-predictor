# Match Feed + Team Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show which series moved team ratings the most, as a match feed beside the power ranking and a per-team breakdown page.

**Architecture:** `elo.run_elo` logs every game's rating change from its existing loop; `elo.group_series` collapses that log into one row per series and `elo.run()` writes it to a committed `{year}_series.parquet`. `app.py` reads that file: a feed column on Power ranking and a new Team page.

**Tech Stack:** Python 3.12, pandas 3.0.3, Streamlit 1.61.1 (`streamlit.testing.v1.AppTest` for checks).

**Spec:** `docs/superpowers/specs/2026-09-11-match-elo-feed-design.md`

## Global Constraints

- Run everything with `.venv/bin/python` from `/Users/zack/projects/league-predictor`.
- No new dependencies; `requirements.txt` does not change.
- Exactly one copy of the Elo update formula (the loop in `elo.run_elo`).
- `2026_team_elo.parquet` must be identical before and after these changes.
- Series boundary: `game` number does not increase, or more than **12h** since the pair's previous game.
- Upset: series winner's `expected(a_elo_before, b_elo_before) < 0.40`.
- Feed: top **15** series; order `Biggest swings` (default) | `Latest`; window `7d / 30d / Season` (default `30d`), measured back from the **latest game in the data**.
- UI uses native Streamlit elements only, no custom HTML.
- Team-page caption, verbatim: `Elo change, not calibrated. League games move calibrated by the same amount; international games by ~90%.`
- The repo has no test suite and this plan does not add one. Check scripts live in the session scratchpad (`$SP` below) and are not committed.
- Work on branch `match-elo-feed`. Commit per task; never push. Commit messages are plain sentences (repo style), ending with the attribution trailer shown in each commit step.

`$SP` = `/private/tmp/claude-501/-Users-zack-projects-league-predictor/7a97f3e3-e286-4c93-ade6-a91821e1c16d/scratchpad`

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `clean.py` | modify | keep raw `game` column in the team-games table |
| `elo.py` | modify | per-game log (`run_elo(game_log=)`, `replay`), `group_series`, `team_perspective`, write `{year}_series` |
| `app.py` | modify | `load_series`, `open_team` navigation, Team page, feed column, ranking row click |
| `CLAUDE.md` | modify | document the series artifact |
| `README.md` | modify | page list on line 45 |
| `data/processed/2026_series.{parquet,csv}` | create (generated) | committed artifact the app reads |

---

### Task 1: Per-game Elo log without changing ratings (~30 min)

**Files:**
- Modify: `clean.py:97-98` (`CONTEXT`), `clean.py:142-143` (`build`)
- Modify: `elo.py:52-70` (`build_matches`), `elo.py:73-108` (`run_elo`), add `replay` after `run_elo`
- Check: `$SP/check_replay.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - team-games table column `game` (float, game number within series, NaN if unknown)
  - `elo.build_matches(tg)` rows gain `game`
  - `elo.run_elo(matches, k=K_DEFAULT, game_log: list[dict] | None = None) -> pd.DataFrame` (unchanged output)
  - `elo.replay(matches, k=K_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(team_table, games)`; `games` columns in order: `gameid, date, league, playoffs, game, winner, loser, winner_elo_before, loser_elo_before, delta, p_winner`

- [ ] **Step 1: Branch and commit the approved docs**

```bash
git switch -c match-elo-feed
git add docs/superpowers/specs/2026-09-11-match-elo-feed-design.md docs/superpowers/plans/2026-09-11-match-elo-feed.md
git commit -m "$(cat <<'EOF'
Add design spec and plan for the match feed and Team page

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UHhXUjjvGmWntso2jA3Fw6
EOF
)"
```

- [ ] **Step 2: Capture baselines with the unmodified code**

```bash
SP=/private/tmp/claude-501/-Users-zack-projects-league-predictor/7a97f3e3-e286-4c93-ade6-a91821e1c16d/scratchpad
.venv/bin/python clean.py > /dev/null
.venv/bin/python elo.py > /dev/null
cp data/processed/2026_team_elo.parquet "$SP/baseline_team_elo.parquet"
.venv/bin/python backtest.py --quick > "$SP/backtest_before.txt" 2>&1
git status --short data/processed
```

Expected: `backtest_before.txt` ends with metrics, no traceback. Note whether `git status` lists any tracked file under `data/processed` as modified. If it does, the local raw CSV is newer than the last committed run: do **not** commit those files in any task; mention it in the final report.

- [ ] **Step 3: Write the check script**

Create `$SP/check_replay.py`:

```python
"""Task 1 check: the replay refactor keeps ratings identical and logs every game."""
import sys

REPO = "/Users/zack/projects/league-predictor"
SP = "/private/tmp/claude-501/-Users-zack-projects-league-predictor/7a97f3e3-e286-4c93-ade6-a91821e1c16d/scratchpad"
sys.path.insert(0, REPO)

import pandas as pd

import elo
import paths

GAME_COLS = ["gameid", "date", "league", "playoffs", "game", "winner", "loser",
             "winner_elo_before", "loser_elo_before", "delta", "p_winner"]

baseline = pd.read_parquet(f"{SP}/baseline_team_elo.parquet")
current = pd.read_parquet(paths.processed(2026, "team_elo"))
assert current.equals(baseline), "2026_team_elo.parquet changed"

matches = elo.build_matches(elo.scoped_team_games())
assert "game" in matches.columns, "build_matches does not carry `game`"
assert matches["game"].notna().all(), "game numbers missing -- did clean.py run?"

table, games = elo.replay(matches)
assert table.equals(baseline), "replay's team table differs from run_elo's"
assert list(games.columns) == GAME_COLS, list(games.columns)
assert len(games) == len(matches)
assert (games["delta"] > 0).all()

gain = (pd.concat([games.groupby("winner")["delta"].sum(),
                   -games.groupby("loser")["delta"].sum()])
        .groupby(level=0).sum())
drift = (table.set_index("team")["elo"] - (elo.BASE + gain).round(1)).abs().max()
# 0.0 on real data when prototyped; the tolerance only absorbs a float landing
# exactly on a rounding boundary.
assert drift <= 0.1, f"per-game deltas do not sum to final Elo (max diff {drift})"

print(f"OK: ratings identical, {len(games)} games logged, sum drift {drift}")
```

- [ ] **Step 4: Run it to verify it fails**

Run: `.venv/bin/python $SP/check_replay.py`
Expected: `AssertionError: build_matches does not carry `game``

- [ ] **Step 5: Keep `game` in `clean.py`**

In `clean.py`, replace:

```python
CONTEXT = ["gameid", "date", "league", "split", "playoffs", "patch",
           "side", "team", "teamid", "datacompleteness"]
```

with:

```python
# `game` is the game number within a series (G1, G2, ...); elo.group_series
# uses it to tell a Bo3 from three separate Bo1s.
CONTEXT = ["gameid", "date", "league", "split", "playoffs", "game", "patch",
           "side", "team", "teamid", "datacompleteness"]
```

and in `build()`, replace:

```python
    out["date"] = pd.to_datetime(out["date"])
    out["playoffs"] = team["playoffs"].astype("int8")
```

with:

```python
    out["date"] = pd.to_datetime(out["date"])
    out["playoffs"] = team["playoffs"].astype("int8")
    out["game"] = pd.to_numeric(team["game"], errors="coerce")
```

- [ ] **Step 6: Carry `game` through `build_matches` and log from `run_elo`**

In `elo.py` `build_matches`, replace:

```python
            "playoffs": int(w["playoffs"]),
            "winner": w["team"], "loser": l["team"],
```

with:

```python
            "playoffs": int(w["playoffs"]),
            # .get: a team-games table written before clean.py kept `game`
            # still rates; group_series then falls back to the time gap.
            "game": w.get("game", float("nan")),
            "winner": w["team"], "loser": l["team"],
```

Replace the `run_elo` signature and loop head:

```python
def run_elo(matches: pd.DataFrame, k: float = K_DEFAULT) -> pd.DataFrame:
```

with:

```python
def run_elo(matches: pd.DataFrame, k: float = K_DEFAULT,
            game_log: list[dict] | None = None) -> pd.DataFrame:
    """Rate `matches` in order. If `game_log` is given, append one dict per game.

    The log is filled from this loop rather than a second replay, so the rating
    changes the app shows cannot drift from the ratings themselves.
    """
```

and replace:

```python
        ew = expected(rw, rl)
        rating[w] = rw + k * (1 - ew)
        rating[l] = rl + k * (0 - (1 - ew))
```

with:

```python
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
```

Add directly after `run_elo`:

```python
def replay(matches: pd.DataFrame, k: float = K_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rate the matches and keep every game's rating change.

    Returns (team table, games): the table is exactly `run_elo`'s; `games` has
    one row per game with both teams' Elo before it and the points exchanged.
    """
    log: list[dict] = []
    table = run_elo(matches, k, game_log=log)
    return table, pd.DataFrame(log)
```

- [ ] **Step 7: Regenerate and run the check**

```bash
.venv/bin/python clean.py > /dev/null && .venv/bin/python elo.py > /dev/null
.venv/bin/python $SP/check_replay.py
```

Expected: `OK: ratings identical, 2856 games logged, sum drift 0.0` (game count may differ if the raw CSV changed).

- [ ] **Step 8: Confirm the backtest did not move**

```bash
.venv/bin/python backtest.py --quick > "$SP/backtest_after.txt" 2>&1
diff "$SP/backtest_before.txt" "$SP/backtest_after.txt"
```

Expected: no output (or differences only in elapsed-time lines; any metric difference is a failure).

- [ ] **Step 9: Commit**

```bash
git add clean.py elo.py
git commit -m "$(cat <<'EOF'
Log each game's Elo change from the rating loop

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UHhXUjjvGmWntso2jA3Fw6
EOF
)"
```

---

### Task 2: Series table artifact (~60 min)

**Files:**
- Modify: `elo.py` (docstring, constants, add `group_series`, `_series_row`, `team_perspective`, update `run`)
- Modify: `CLAUDE.md` (Architecture section)
- Create (generated): `data/processed/2026_series.parquet`, `data/processed/2026_series.csv`
- Check: `$SP/check_series.py`

**Interfaces:**
- Consumes: `elo.replay(matches, k) -> (table, games)` from Task 1.
- Produces:
  - `elo.SERIES_GAP = pd.Timedelta(hours=12)`, `elo.UPSET_P = 0.40`
  - `elo.group_series(games: pd.DataFrame) -> pd.DataFrame` with columns in order: `series_id, date, league, playoffs, team_a, team_b, wins_a, wins_b, result, games, a_elo_before, a_elo_after, b_elo_before, b_elo_after, delta_a, swing, upset`
  - `elo.team_perspective(series: pd.DataFrame, team: str) -> pd.DataFrame`, newest first, columns in order: `date, league, stage, opponent, outcome, score, won, lost, elo_change, elo_after, upset` (`outcome` in `W/L/D`, `stage` in `Playoffs/Regular`)
  - file `paths.processed(year, "series")`

- [ ] **Step 1: Write the check script**

Create `$SP/check_series.py`:

```python
"""Task 2 check: series grouping on synthetic edge cases, then on real data."""
import sys

REPO = "/Users/zack/projects/league-predictor"
sys.path.insert(0, REPO)

import pandas as pd

import elo
import paths

SERIES_COLS = ["series_id", "date", "league", "playoffs", "team_a", "team_b",
               "wins_a", "wins_b", "result", "games", "a_elo_before", "a_elo_after",
               "b_elo_before", "b_elo_after", "delta_a", "swing", "upset"]
PERSPECTIVE_COLS = ["date", "league", "stage", "opponent", "outcome", "score",
                    "won", "lost", "elo_change", "elo_after", "upset"]


def m(day, hour, game, winner, loser, league="LCK", playoffs=0):
    return {"date": pd.Timestamp("2026-03-01") + pd.Timedelta(days=day, hours=hour),
            "gameid": f"g{day:02d}{hour:02d}-{winner}-{loser}", "league": league,
            "playoffs": playoffs, "game": game, "winner": winner, "loser": loser}


def series_for(rows):
    matches = pd.DataFrame(rows).sort_values(["date", "gameid"]).reset_index(drop=True)
    _, games = elo.replay(matches)
    return elo.group_series(games)


# 1. A Bo3, then a rematch a week later: two series.
s = series_for([m(0, 0, 1, "A", "B"), m(0, 1, 2, "B", "A"), m(0, 2, 3, "A", "B"),
                m(7, 0, 1, "B", "A")])
assert list(s.columns) == SERIES_COLS, list(s.columns)
assert len(s) == 2, len(s)
r = s.iloc[0]
assert (r.team_a, r.team_b, r.wins_a, r.wins_b, r.result) == ("A", "B", 2, 1, "win")
assert r.games == "A|B|A", r.games
assert r.a_elo_before == r.b_elo_before == 1500.0
assert abs(r.a_elo_after - r.a_elo_before - r.delta_a) < 0.15
tv = elo.team_perspective(s, "A")
assert list(tv.columns) == PERSPECTIVE_COLS, list(tv.columns)
assert list(tv.outcome) == ["L", "W"] and list(tv.score) == ["0-1", "2-1"]

# 2. A 1-1 Bo2: draw, alphabetical order, never an upset.
r = series_for([m(0, 0, 1, "Zed", "Ace"), m(0, 1, 2, "Ace", "Zed")]).iloc[0]
assert (r.result, r.team_a, r.team_b, r.wins_a, r.wins_b, bool(r.upset)) == \
       ("draw", "Ace", "Zed", 1, 1, False)
assert list(elo.team_perspective(series_for([m(0, 0, 1, "Zed", "Ace"), m(0, 1, 2, "Ace", "Zed")]), "Zed").outcome) == ["D"]

# 3. A heavy favourite (1643.8 vs 1500) wins 3-2 and still loses Elo (net -8.2).
fillers = [m(d, 0, 1, "Fav", f"Filler{d}") for d in range(12)]
s = series_for(fillers + [m(20, 0, 1, "Dog", "Fav"), m(20, 1, 2, "Fav", "Dog"),
                          m(20, 2, 3, "Dog", "Fav"), m(20, 3, 4, "Fav", "Dog"),
                          m(20, 4, 5, "Fav", "Dog")])
r = s[s.team_b == "Dog"].iloc[0]
assert (r.team_a, r.wins_a, r.wins_b) == ("Fav", 3, 2)
assert r.delta_a < 0, r.delta_a
assert not r.upset

# 4. The underdog (p = 0.304) sweeps: an upset with a positive swing.
s = series_for(fillers + [m(20, 0, 1, "Dog", "Fav"), m(20, 1, 2, "Dog", "Fav")])
r = s[s.team_a == "Dog"].iloc[0]
assert r.upset and r.delta_a > 0 and r.swing == abs(r.delta_a)

# 5. Missing game numbers fall back to the 12h gap.
nan = float("nan")
s = series_for([m(0, 0, nan, "A", "B"), m(0, 1, nan, "A", "B"), m(0, 14, nan, "A", "B")])
assert list(s.wins_a + s.wins_b) == [2, 1], list(s.wins_a + s.wins_b)

# 6. Same-day rematch (double elimination): the game number resets inside 12h.
s = series_for([m(0, 0, 1, "A", "B"), m(0, 1, 2, "A", "B"),
                m(0, 5, 1, "B", "A"), m(0, 6, 2, "B", "A")])
assert len(s) == 2, len(s)

print("OK: synthetic cases")

# ---- real data
series = pd.read_parquet(paths.processed(2026, "series"))
team_elo = pd.read_parquet(paths.processed(2026, "team_elo"))
matches = elo.build_matches(elo.scoped_team_games())

assert list(series.columns) == SERIES_COLS
assert series["series_id"].is_unique
n_games = series.wins_a + series.wins_b
assert int(n_games.sum()) == len(matches)
assert n_games.max() <= 5
assert ((series.a_elo_after - series.a_elo_before - series.delta_a).abs() < 0.15).all()
lck = n_games[series.league == "LCK"]
assert lck.isin([2, 3]).mean() > 0.8, lck.value_counts()

# Every team's latest series must end on its final rating.
long = pd.concat([
    series[["date", "team_a", "a_elo_after"]].set_axis(["date", "team", "after"], axis=1),
    series[["date", "team_b", "b_elo_after"]].set_axis(["date", "team", "after"], axis=1),
])
last = long.sort_values("date").groupby("team")["after"].last()
gap = (team_elo.set_index("team")["elo"] - last).abs().max()
assert gap <= 0.1, gap

top = team_elo.iloc[0]["team"]
tv = elo.team_perspective(series, top)
assert len(tv) == int(((series.team_a == top) | (series.team_b == top)).sum())
assert tv["date"].is_monotonic_decreasing
assert abs(tv.iloc[0].elo_after - team_elo.iloc[0]["elo"]) <= 0.1

print(f"OK: {len(series)} series from {len(matches)} games; "
      f"{int(series.upset.sum())} upsets, {int((series.result == 'draw').sum())} draws")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python $SP/check_series.py`
Expected: `AttributeError: module 'elo' has no attribute 'group_series'`

- [ ] **Step 3: Implement `group_series` and `team_perspective`**

In `elo.py`, replace the docstring's output line:

```python
Output: data/processed/2026_team_elo.parquet      (+ .csv mirror)
```

with:

```python
Output: data/processed/2026_team_elo.parquet      (+ .csv mirror)
        data/processed/2026_series.parquet        (+ .csv mirror; one row per
                                                   series with each team's Elo
                                                   change, read by app.py)
```

Below `BASE = 1500.0`, add:

```python
# A new series starts when a pair's game number stops increasing, or when the
# pair has not played for this long. The gap is the fallback for games with no
# game number.
SERIES_GAP = pd.Timedelta(hours=12)

# A series win counts as an upset when the winner's pre-series chance of taking
# any single game was below this.
UPSET_P = 0.40
```

After `replay`, add:

```python
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
```

In `run()`, replace:

```python
    print(f"Running Elo (K={k})...")
    elo = run_elo(matches, k)

    paths.ensure_dirs()
    pq = paths.processed(year, "team_elo")
    elo.to_parquet(pq, index=False)
    elo.to_csv(paths.processed(year, "team_elo", "csv"), index=False)
    print(f"\nWrote {len(elo)} team ratings -> {pq}")
    return elo
```

with:

```python
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
```

- [ ] **Step 4: Regenerate and run both checks**

```bash
.venv/bin/python elo.py | tail -3
.venv/bin/python $SP/check_series.py
.venv/bin/python $SP/check_replay.py
```

Expected: `OK: synthetic cases`, then `OK: 1276 series from 2856 games; 125 upsets, 4 draws` (counts from the prototype; they may differ if the raw CSV changed), then Task 1's `OK:` line.

- [ ] **Step 5: Document the artifact in `CLAUDE.md`**

In `CLAUDE.md`, directly after the paragraph that starts `**`monte_carlo.py`** builds a projected field`, add:

```markdown
**Series log.** `elo.run()` also writes `{year}_series.parquet`: one row per
series with both teams' Elo before/after and the net change, which the app's
match feed and Team page read. It comes from the rating loop itself
(`run_elo(game_log=...)`), not a second replay, so it cannot drift from the
ratings. It must be committed: `team_games` is gitignored, so Streamlit Cloud
cannot rebuild it. Series are split when a pair's `game` number stops
increasing, or after a 12h gap when the number is missing. The feed shows raw
Elo deltas; for league games that equals the change in `calibrated`, for
internationals it is ~10% larger.
```

- [ ] **Step 6: Commit**

```bash
git add elo.py CLAUDE.md data/processed/2026_series.parquet data/processed/2026_series.csv
git commit -m "$(cat <<'EOF'
Write a per-series Elo change table for the app

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UHhXUjjvGmWntso2jA3Fw6
EOF
)"
```

---

### Task 3: Team page (~60 min)

**Files:**
- Modify: `app.py` (imports, loaders, new navigation section, `page_team`, `PAGES`, sidebar radio)
- Check: `$SP/check_team_page.py`

**Interfaces:**
- Consumes: `elo.team_perspective(series, team)` and `paths.processed(year, "series")` from Task 2.
- Produces:
  - `load_series(year=YEAR) -> pd.DataFrame` (empty frame if the file is missing)
  - `open_team(team: str) -> None` widget callback: sets `st.session_state["page"] = "Team"` and `st.session_state["team"] = team`
  - sidebar radio key `"page"`, Team page selectbox key `"team"`, page name `"Team"`

- [ ] **Step 1: Write the check script**

Create `$SP/check_team_page.py`:

```python
"""Task 3 check: the Team page renders for every ranked team."""
import sys

REPO = "/Users/zack/projects/league-predictor"
sys.path.insert(0, REPO)

import pandas as pd
from streamlit.testing.v1 import AppTest

import paths

ratings = pd.read_parquet(paths.processed(2026, "calibrated_ratings"))


def ok(at, context=""):
    assert not at.exception, (context, [e.value for e in at.exception])


at = AppTest.from_file(f"{REPO}/app.py", default_timeout=120).run()
ok(at, "first load")

at.radio(key="page").set_value("Team").run()
ok(at, "open Team")
assert at.selectbox(key="team").value == ratings.iloc[0]["team"]
labels = [m.label for m in at.metric]
assert labels[0].startswith("Calibrated · #1"), labels
assert labels[1:] == ["Elo", "Series", "Games", "Elo, last 30 days"], labels
assert len(at.dataframe) == 1
assert any("Elo change, not calibrated" in c.value for c in at.caption)

for team in ratings["team"]:
    at.selectbox(key="team").set_value(team).run()
    ok(at, team)

print(f"OK: Team page renders for all {len(ratings)} teams")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python $SP/check_team_page.py`
Expected: `KeyError` / `ValueError` on `at.radio(key="page")` (the radio has no key yet).

- [ ] **Step 3: Add the loader, navigation helper, and Team page**

In `app.py`, replace:

```python
import backtest
import history
```

with:

```python
import backtest
import elo
import history
```

After `load_report`, add:

```python
@st.cache_data
def load_series(year: int = YEAR) -> pd.DataFrame:
    path = paths.processed(year, "series")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
```

After `data_age_banner`, add:

```python
# ------------------------------------------------------------------ navigation

def open_team(team: str) -> None:
    """Jump to the Team page. Must run as a widget callback: callbacks fire
    before the rerun renders the sidebar radio, the only moment its state may
    still be changed."""
    st.session_state["page"] = "Team"
    st.session_state["team"] = team
```

After `page_ranking`, add:

```python
def page_team() -> None:
    st.header("Team breakdown")
    ratings = load_ratings()
    series = load_series()
    if series.empty:
        st.info("No match data yet. Run `python pipeline.py`.")
        return

    teams = ratings["team"].tolist()  # calibrated_ratings is already rank order
    if st.session_state.get("team") not in teams:
        st.session_state["team"] = teams[0]
    team = st.selectbox("Team", teams, key="team")

    r = ratings.set_index("team").loc[team]
    view = elo.team_perspective(series, team)
    # Measured from the data, not the clock -- see match_feed.
    recent = view[view["date"] > series["date"].max() - pd.Timedelta(days=30)]
    outcomes = view["outcome"].value_counts()
    series_record = f"{outcomes.get('W', 0)}-{outcomes.get('L', 0)}"
    if outcomes.get("D", 0):
        series_record += f"-{outcomes['D']}"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Calibrated · #{teams.index(team) + 1}", f"{r['calibrated']:.1f}")
    c2.metric("Elo", f"{r['elo']:.1f}")
    c3.metric("Series", series_record)
    c4.metric("Games", f"{int(view['won'].sum())}-{int(view['lost'].sum())}")
    c5.metric("Elo, last 30 days", f"{recent['elo_change'].sum():+.1f}")
    if r["low_sample"]:
        st.caption("Fewer than 20 rated games: this rating is mostly noise.")

    if view.empty:
        st.info(f"No rated series for {team}.")
        return

    st.line_chart(view.set_index("date")["elo_after"].sort_index(), height=300,
                  y_label="Elo")
    st.dataframe(
        view.drop(columns=["won", "lost"]), width="stretch", hide_index=True,
        column_config={
            "date": st.column_config.DatetimeColumn(format="MMM D, YYYY"),
            "outcome": st.column_config.TextColumn("result"),
            "elo_change": st.column_config.NumberColumn("Elo change", format="%+.1f"),
            "elo_after": st.column_config.NumberColumn("Elo after", format="%.1f"),
            "upset": st.column_config.CheckboxColumn(),
        },
    )
    st.caption("Elo change, not calibrated. League games move calibrated by the "
               "same amount; international games by ~90%.")
```

Replace:

```python
PAGES = {
    "Power ranking": page_ranking,
    "Rating history": page_history,
```

with:

```python
PAGES = {
    "Power ranking": page_ranking,
    "Team": page_team,
    "Rating history": page_history,
```

Replace:

```python
choice = st.sidebar.radio("Page", list(PAGES))
```

with:

```python
choice = st.sidebar.radio("Page", list(PAGES), key="page")
```

- [ ] **Step 4: Run the check**

Run: `.venv/bin/python $SP/check_team_page.py`
Expected: `OK: Team page renders for all 51 teams`

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "$(cat <<'EOF'
Add a Team page showing each series' Elo change

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UHhXUjjvGmWntso2jA3Fw6
EOF
)"
```

---

### Task 4: Match feed beside the power ranking (~60 min)

**Files:**
- Modify: `app.py` (`page_ranking`; add feed constants and helpers above `page_ranking`)
- Check: `$SP/check_feed.py`

**Interfaces:**
- Consumes: `load_series()`, `open_team(team)`, radio key `"page"`, selectbox key `"team"` from Task 3.
- Produces: radio keys `"feed_order"`, `"feed_window"`; feed team buttons keyed `feed_{series_id}_a` / `feed_{series_id}_b`; ranking dataframe key `"ranking_table"`.

- [ ] **Step 1: Write the check script**

Create `$SP/check_feed.py`:

```python
"""Task 4 check: the feed renders in every mode and a team click opens its page."""
import sys

REPO = "/Users/zack/projects/league-predictor"
sys.path.insert(0, REPO)

import pandas as pd
from streamlit.testing.v1 import AppTest

import paths

ratings = pd.read_parquet(paths.processed(2026, "calibrated_ratings"))


def ok(at, context=""):
    assert not at.exception, (context, [e.value for e in at.exception])


def feed_buttons(at):
    return [b for b in at.button if str(b.key).startswith("feed_")]


at = AppTest.from_file(f"{REPO}/app.py", default_timeout=120).run()
ok(at, "first load")
assert any(h.value == "Matches" for h in at.subheader)

for order in ["Biggest swings", "Latest"]:
    for window in ["7d", "30d", "Season"]:
        at.radio(key="feed_order").set_value(order).run()
        at.radio(key="feed_window").set_value(window).run()
        ok(at, (order, window))

at.radio(key="feed_window").set_value("Season").run()
buttons = feed_buttons(at)
assert buttons, "no clickable teams in the season feed"
assert len(buttons) <= 2 * 15, len(buttons)
assert {b.label for b in buttons} <= set(ratings["team"])

lec = set(ratings.loc[ratings["home_league"] == "LEC", "team"])
at.multiselect[0].set_value(["LEC"]).run()
ok(at, "LEC filter")
assert {b.label for b in feed_buttons(at)} & lec, "LEC filter shows no LEC team"

name = feed_buttons(at)[0].label
feed_buttons(at)[0].click().run()
ok(at, "click")
assert at.radio(key="page").value == "Team"
assert at.selectbox(key="team").value == name

print(f"OK: feed renders in all modes; clicking {name!r} opens its Team page")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python $SP/check_feed.py`
Expected: `AssertionError` on `any(h.value == "Matches" ...)`

- [ ] **Step 3: Add the feed helpers**

In `app.py`, directly above `def page_ranking`, add:

```python
FEED_WINDOWS = {"7d": 7, "30d": 30, "Season": None}
FEED_SIZE = 15


def feed_series(series: pd.DataFrame, teams: set[str], window: str,
                order: str) -> pd.DataFrame:
    shown = series[series["team_a"].isin(teams) | series["team_b"].isin(teams)]
    days = FEED_WINDOWS[window]
    if days is not None:
        # Measured back from the latest game, not from today: the data can
        # trail the clock by weeks, and a clock window would silently be empty.
        shown = shown[shown["date"] > series["date"].max() - pd.Timedelta(days=days)]
    keys = ["swing", "date"] if order == "Biggest swings" else ["date"]
    return shown.sort_values(keys, ascending=False).head(FEED_SIZE)


def team_row(team: str, wins: int, delta: float, clickable: set[str], key: str) -> None:
    name, score, change = st.columns([5, 1, 2], vertical_alignment="center")
    if team in clickable:
        name.button(team, key=key, type="tertiary", on_click=open_team, args=(team,))
    else:
        name.markdown(team)  # outside the ranking (e.g. LJL), so no Team page
    score.markdown(f"**{wins}**")
    change.markdown(f":{'green' if delta >= 0 else 'red'}[{delta:+.1f}]")


def match_card(s, clickable: set[str]) -> None:
    with st.container(border=True):
        head = f"{s.date:%b %d} · {s.league}" + (" · Playoffs" if s.playoffs else "")
        if s.upset:
            head += " · :orange-badge[UPSET]"
        st.caption(head)
        team_row(s.team_a, s.wins_a, s.delta_a, clickable, f"feed_{s.series_id}_a")
        team_row(s.team_b, s.wins_b, -s.delta_a, clickable, f"feed_{s.series_id}_b")
        st.caption(" · ".join(f"G{i} {w}" for i, w in enumerate(s.games.split("|"), 1)))


def match_feed(teams: set[str], clickable: set[str]) -> None:
    st.subheader("Matches")
    series = load_series()
    if series.empty:
        st.info("No match data yet. Run `python pipeline.py`.")
        return
    order = st.radio("Order", ["Biggest swings", "Latest"], key="feed_order",
                     horizontal=True, label_visibility="collapsed")
    window = st.radio("Window", list(FEED_WINDOWS), index=1, key="feed_window",
                      horizontal=True, label_visibility="collapsed")
    shown = feed_series(series, teams, window, order)
    if shown.empty:
        st.caption("No series in this window.")
        return
    with st.container(height=640):
        for s in shown.itertuples(index=False):
            match_card(s, clickable)
    st.caption("Raw Elo points exchanged over the series.")


def select_ranking_row() -> None:
    rows = st.session_state["ranking_table"].selection.rows
    if rows:
        open_team(st.session_state["ranking_teams"][rows[0]])
```

- [ ] **Step 4: Put the table and feed side by side**

In `page_ranking`, replace everything from `    view["win_rate"] = view["win_rate"] * 100` to the end of the function:

```python
    view["win_rate"] = view["win_rate"] * 100

    # ProgressColumn rather than a pandas background_gradient: the latter needs
    # matplotlib, which is not worth a dependency for one column of shading.
    st.dataframe(
        view, width="stretch", hide_index=True,
        column_config={
            "calibrated": st.column_config.ProgressColumn(
                "calibrated", format="%.1f",
                min_value=float(view["calibrated"].min()),
                max_value=float(view["calibrated"].max())),
            "win_rate": st.column_config.NumberColumn("win rate", format="%.1f%%"),
            "within_league": st.column_config.NumberColumn("within league", format="%+.1f"),
            "elo": st.column_config.NumberColumn(format="%.1f"),
            "region_elo": st.column_config.NumberColumn("region elo", format="%.1f"),
            "low_sample": st.column_config.CheckboxColumn("low sample"),
        },
    )

    n_low = int(df["low_sample"].sum())
    if n_low and not show_low:
        st.caption(f"{n_low} low-sample team(s) hidden.")
```

with:

```python
    view["win_rate"] = view["win_rate"] * 100
    # The row-click callback only receives row positions.
    st.session_state["ranking_teams"] = view["team"].tolist()

    table, feed = st.columns([3, 2], gap="large")
    with table:
        # ProgressColumn rather than a pandas background_gradient: the latter needs
        # matplotlib, which is not worth a dependency for one column of shading.
        st.dataframe(
            view, width="stretch", hide_index=True,
            key="ranking_table", on_select=select_ranking_row,
            selection_mode="single-row",
            column_config={
                "calibrated": st.column_config.ProgressColumn(
                    "calibrated", format="%.1f",
                    min_value=float(view["calibrated"].min()),
                    max_value=float(view["calibrated"].max())),
                "win_rate": st.column_config.NumberColumn("win rate", format="%.1f%%"),
                "within_league": st.column_config.NumberColumn("within league", format="%+.1f"),
                "elo": st.column_config.NumberColumn(format="%.1f"),
                "region_elo": st.column_config.NumberColumn("region elo", format="%.1f"),
                "low_sample": st.column_config.CheckboxColumn("low sample"),
            },
        )

        n_low = int(df["low_sample"].sum())
        if n_low and not show_low:
            st.caption(f"{n_low} low-sample team(s) hidden.")
        st.caption("Click a row to open that team's page.")

    with feed:
        match_feed(set(view["team"]), set(df["team"]))
```

- [ ] **Step 5: Run both UI checks**

```bash
.venv/bin/python $SP/check_feed.py
.venv/bin/python $SP/check_team_page.py
```

Expected: `OK: feed renders in all modes; clicking '<team>' opens its Team page`, then `OK: Team page renders for all 51 teams`.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "$(cat <<'EOF'
Show the biggest Elo swings beside the power ranking

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UHhXUjjvGmWntso2jA3Fw6
EOF
)"
```

---

### Task 5: Whole-app check, visual check, README (~30 min)

**Files:**
- Modify: `README.md:45`
- Check: `$SP/check_all_pages.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing new.

- [ ] **Step 1: Write and run the all-pages check**

Create `$SP/check_all_pages.py`:

```python
"""Task 5 check: every page renders without an exception."""
import sys

REPO = "/Users/zack/projects/league-predictor"
sys.path.insert(0, REPO)

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(f"{REPO}/app.py", default_timeout=120).run()
for page in ["Power ranking", "Team", "Rating history", "Worlds odds", "Model health"]:
    at.radio(key="page").set_value(page).run()
    assert not at.exception, (page, [e.value for e in at.exception])
print("OK: all 5 pages render")
```

Run: `.venv/bin/python $SP/check_all_pages.py`
Expected: `OK: all 5 pages render`

- [ ] **Step 2: Launch the app and look at it**

Invoke the `run` skill to start `.venv/bin/streamlit run app.py` and drive it in the browser. Screenshot and confirm each item:

1. Power ranking: the Matches panel sits right of the table and scrolls.
2. An UPSET series shows an orange badge (not the literal text `:orange-badge[UPSET]`).
3. Deltas are green for gains, red for losses.
4. Clicking a team name in a card opens that team's Team page.
5. Clicking a row in the ranking table opens that team's Team page (AppTest cannot simulate this one).
6. Team page: chart and table render; the caption text matches the Global Constraints.

If item 2 shows literal text, `st.caption` is not rendering the badge directive: in `match_card` replace `st.caption(head)` with `st.markdown(head)` and re-run `$SP/check_feed.py`.

If item 5 does nothing, report it instead of improvising a different navigation scheme; the spec keeps the sidebar radio.

- [ ] **Step 3: Update the README page list**

In `README.md`, replace:

```
.venv/bin/streamlit run app.py           # power ranking, history, odds, model health
```

with:

```
.venv/bin/streamlit run app.py           # power ranking + match feed, team, history, odds, model health
```

- [ ] **Step 4: Confirm what is about to be committed**

```bash
git status --short
```

Expected: only `README.md` modified (plus any `app.py` fix from Step 2). If tracked files under `data/processed` other than `2026_series.*` show as modified, leave them unstaged and report it.

- [ ] **Step 5: Commit**

```bash
git add README.md app.py
git commit -m "$(cat <<'EOF'
List the match feed and Team page in the README

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UHhXUjjvGmWntso2jA3Fw6
EOF
)"
```
