# Match feed + Team page: showing how matches move Elo

Date: 2026-09-11
Status: approved in chat, awaiting spec review

## Goal

Show which matches mattered most and how each one moved the teams involved,
HLTV-style: a match feed beside the power ranking, plus a per-team breakdown.

## Decisions

| Question | Decision |
|---|---|
| Layout | Match feed column on the Power ranking page **and** a new Team page |
| "Important" | Biggest Elo swing (absolute net rating change over the series) |
| Number shown | Raw Elo delta (K=30), summed over the series |

Rejected: title-odds change per match (a Monte Carlo rerun per series is too
slow for the 6-hourly job); calibrated before/after replay (extra pipeline cost
for a number that equals the Elo delta on league games anyway).

## Why raw Elo delta is honest here

`calibrated = region_elo + (elo - league_mean)`. A league game between two
established teams of the same home league moves one team +d and the other -d,
so the league mean is unchanged and `calibrated` moves by exactly ±d. At
international events the league mean shifts by d/n (n ≈ 10), so `calibrated`
moves by ~90% of d. The UI states this in one caption.

## Constraint that shapes the design

`data/processed/*_team_games.parquet` is gitignored, so Streamlit Cloud cannot
replay Elo in the app. The pipeline must write a new, committed artifact.

## Data layer

### `clean.py`
- Add the raw `game` column (game number within a series) to `CONTEXT`.

### `elo.py`
- `build_matches` carries `game` through to each match row.
- New `replay(matches, k) -> DataFrame`: the single chronological Elo loop,
  returning one row per game with `winner_elo_before`, `loser_elo_before`,
  `delta` (points gained by the winner = points lost by the loser), and
  `p_winner` (pre-game expected win probability for the winner).
- `run_elo` gains an optional `game_log` list that its existing loop appends
  to; `replay` calls it and returns `(team table, game log)`. There is exactly
  **one** copy of the update formula, and `run_elo`'s output stays identical.
- New `group_series(games) -> DataFrame`: consecutive games between the same
  unordered team pair in the same league form one series; a new series starts
  when `game` does not increase over the previous game of that pair, or the gap
  since that game exceeds 12h.
- `run()` writes `data/processed/{year}_series.parquet` (+ `.csv` mirror) next
  to `team_elo`.

### `{year}_series` schema (one row per series)

| column | meaning |
|---|---|
| `series_id` | stable id: first game's `gameid` |
| `date` | first game's timestamp |
| `league`, `playoffs` | from the first game |
| `team_a`, `team_b` | series winner / loser; on a tie, alphabetical order |
| `wins_a`, `wins_b` | game score |
| `result` | `"win"` or `"draw"` |
| `games` | game-by-game winners joined by a pipe character (team names can contain commas) |
| `a_elo_before`, `a_elo_after`, `b_elo_before`, `b_elo_after` | raw Elo around the series |
| `delta_a` | net Elo change for `team_a`; `delta_b = -delta_a` (each game is zero-sum) |
| `swing` | `abs(delta_a)`, the importance sort key |
| `upset` | `team_a` won and `expected(a_elo_before, b_elo_before)` < 0.40 |

Edge cases:
- A series winner can have a negative `delta_a` (a heavy favourite winning 3-2).
  Displayed with its real sign.
- Tied series (1-1 in Bo2 formats) have `result = "draw"` and `upset = False`.

### Wiring
No change to `pipeline.py` or `.github/workflows/update.yml`: `elo.run()` writes
the file and the workflow already runs `git add data/processed`.

## UI (`app.py`)

### Power ranking page
- `st.columns`: existing ranking table left (unchanged), **Matches** panel right
  in a fixed-height scrollable container.
- Controls: `Biggest swings` (default) | `Latest`; window `7d / 30d / Season`
  (default 30d), measured back from the **latest game in the data**, not today.
- Filter: a series is shown if either team is in the page's league-filtered
  ranking. Top 15.
- Card: `date · league · Playoffs` line with an UPSET badge, then one row per
  team (name, games won, colored Elo delta), then a game-by-game line. Rows
  rather than `A 2-1 B` on one line because long team names do not fit a
  narrow column. Built from native Streamlit elements (bordered container,
  markdown color tags), no custom HTML.
- Clicking a team name opens the Team page for that team.
- Clicking a row in the ranking table also opens the Team page.

### Team page (new sidebar entry after Power ranking)
- Team selectbox: ranked teams in rank order, pre-selected from a click.
- Metrics row: calibrated rating + rank, raw Elo, series record and game record,
  Elo change over the last 30 days of data.
- Line chart: Elo after each series.
- Table: every series, newest first: date, league, stage, opponent, W/L/D,
  score, Elo change (`%+.1f`), Elo after, upset. Sortable by column header.
- Caption: "Elo change, not calibrated. League games move calibrated by the same
  amount; international games by ~90%."
- Missing `{year}_series.parquet` shows an `st.info` telling the user to run the
  pipeline, matching the other pages.

## Verification

There is no test suite; these are the checks.

1. `2026_team_elo.parquet` is identical before and after the `replay` refactor
   (`DataFrame.equals` against a saved copy).
2. `backtest.py --quick` numbers are unchanged.
3. For every team, `1500 + sum of its per-game deltas == final elo`.
4. Series sizes are plausible per league (e.g. LCK series have 2-3 games).
5. Streamlit `AppTest` renders every page, including a click-through to the
   Team page, without exceptions.
6. Launch the app locally and screenshot both pages.

## Out of scope
- Title-odds impact per match.
- Calibrated before/after values per series.
- Changing the sidebar to `st.navigation` or splitting `app.py` into modules.
- Non-ranked teams (LJL/CBLOL/PCS/VCS-only) on the Team page picker.
