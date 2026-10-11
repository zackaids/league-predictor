# Worlds 2026 Swiss & Knockout Odds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox syntax. Each task writes its check script first (scratchpad, not committed), sees it fail, implements, sees it pass, commits.

**Goal:** Replace the projected Monte Carlo with an exact-format Worlds 2026 Swiss + knockout simulator, and surface Play-In -> Swiss -> Knockout -> Title odds as one interactive "Worlds" page with play-through what-ifs.

**Architecture:** `monte_carlo.py` becomes a numpy engine over a `Field` + `Known` state (draw rules, Bo1/Bo3/Bo5, QF draw). `swiss.py` (mirrors `bracket.py`) holds the confirmed event: config, results from the series log, what-if pruning, outputs. A `st.components.v2` board (`board.py` + `assets/board.{js,css}`) renders all stages; `worlds_view.py` builds board data.

**Tech Stack:** Python 3.14 venv, numpy, pandas, Streamlit 1.61 (components v2), Altair.

**Spec:** `docs/superpowers/specs/2026-10-11-worlds-swiss-design.md`

**Testing convention:** the repo has no committed test suite. Check scripts live in the session scratchpad (`$SP`) and are not committed. `backtest.py --quick` output must be identical before/after.

## Global Constraints

- Use `.venv/bin/python`. Oracle's Elixir names: `LØS`, `Top Esports`, `Team Liquid`, `Cloud9` (display "Cloud9 Kia").
- League code `WLDs` also holds September LPL qualifier series: always filter by the stage's date window.
- Swiss: 16 teams, 5 rounds, 3 wins advance / 3 losses out; Bo3 iff record has 2 wins or 2 losses; Round 1 = T1 v T4 and T2 v T3 with no same-league pair; rounds 2-5 random within record, no rematch; QF draw: each 3-0 v a 3-2, 3-0s on opposite halves; knockout Bo5.
- Win odds always through `calibrate.prob_from_gap` (carries `RATING_SCALE`) and `bracket.p_series`.
- Never merge PRs; never force-push; do not commit locally regenerated data except `*_playin_field.csv` / `*_swiss_field.csv`.
- Every commit message ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- Play-In winner unresolved: odds must integrate over all four candidates, never crash.
- A half-published Bo3 (1-0) must not be read as a finished series.
- Series between two Swiss teams inside the window that match no drawn pair must produce a warning, not a silent drop.
- A stale what-if (earlier result changed) must be pruned, never trusted.
- A renamed team upstream must not block the daily ratings commit (pipeline stage is guarded).

## Interfaces (fixed here; later tasks rely on them)

```python
# monte_carlo.py
@dataclass
class Field:
    teams: list[str]            # 15 direct teams (slots 0-14), then Play-In candidates
    rating: np.ndarray          # [T]
    league: list[str]           # [T]
    tier: list[int]             # [15] tiers 1-4 of the direct teams (Play-In slot is tier 4)
    short: list[str]            # [T]
    playin: dict[str, float]    # candidate -> P(wins Play-In); sums to 1
    league_rule: bool = True    # Round 1 forbids same-league pairs
    n_direct: int = 15

@dataclass
class Known:
    playin_winner: str | None = None
    draws: dict[int, list[tuple[str, str]]] = {}          # round -> pairs (drawn order)
    results: dict[int, dict[tuple[str, str], str]] = {}   # round -> {(a,b): winner}
    qf: list[tuple[str, str]] | None = None               # 4 pairs, bracket order
    ko: dict[str, str] = {}                               # QF1..QF4, SF1, SF2, F -> winner

@dataclass
class Odds:                 # all arrays indexed by Field.teams order
    n: int; fallback: int
    swiss: np.ndarray       # P(advance)
    record: np.ndarray      # [T, 6] P(final record) cols: 3-0,3-1,3-2,2-3,1-3,0-3
    semis: np.ndarray; final: np.ndarray; title: np.ndarray
    occupancy: dict[int, dict[tuple[int, int], np.ndarray]]  # round -> (w,l) -> [T]
    meet: dict[int, np.ndarray]       # round -> [T,T] P(a plays b that round)
    qf_slot: np.ndarray               # [T,4]

def simulate(field, known=None, n=20000, seed=0) -> Odds
def sample_round(field, known, rng) -> list[tuple[str,str]]   # one legal draw for the next undrawn round
def sample_quarterfinals(field, known, rng) -> list[tuple[str,str]]
def match_prob(field, a, b, best_of) -> float                 # P(a wins)
def odds_table(field, odds, field_kind) -> pd.DataFrame       # the title-odds CSV schema

# swiss.py
def load_config(path=CONFIG) -> dict
def build_field(cfg, playin_probs, year) -> monte_carlo.Field
def resolve(cfg, series, year) -> Event
def known_state(event, whatif) -> tuple[Known, dict, list[str]]   # Known, pruned whatif, notes
def play_out(event, whatif, field, rng) -> dict                   # whatif with every open decision filled
def write_outputs(year, n, seed) -> pd.DataFrame
```

---

## Phase 1 - engine + correct title odds

### Task 1: Configs and Play-In tricodes
**Files:** Create `worlds_swiss.toml`; modify `worlds_bracket.toml` (add `short`).
- [ ] Check script: `tomllib` loads both; 15 teams, 4 per tier (T4 has 3 + Play-In); every `data_name or name` exists in `2026_calibrated_ratings.csv`.
- [ ] Write `worlds_swiss.toml` (dates 2026-10-23..31 / 11-03..14, Allen TX; teams with name/short/tier/data_name; empty `[draws]`, `[knockout]`, `[results.swiss]`, `[results.knockout]`) and add `short` to the four Play-In teams.
- [ ] Run check -> pass. Commit.

### Task 2: Engine (`monte_carlo.py`)
**Files:** Rewrite `monte_carlo.py` (keep `load_field_config`, `build_field`, `p_game`).
- [ ] Check script `check_engine.py` asserting, for `simulate(field, n=20000, seed=0)`: record counts per sim always 2/3/3/3/3/2 (via `record.sum(axis=0)` = [2,3,3,3,3,2]); `fallback == 0`; title sums to 1; `title <= final <= semis <= swiss`; equal-ratings field -> swiss in 0.5+-0.01, 3-0 in 0.125+-0.01, title 0.0625+-0.005; Round 1 pairs always tier1 v tier4 / tier2 v tier3 with no same-league pair (via `sample_round` x 500); Bo3 used exactly when 2 wins/2 losses (exposed through a debug counter); same seed -> identical; known draws/results respected; late-state (rounds 1-4 fixed, knockout open) agrees with brute-force enumeration within MC error; 20k sims < 1 s.
- [ ] Run -> fails (no `Field`).
- [ ] Implement `Field`, `Known`, `Odds`, `simulate`, `sample_round`, `sample_quarterfinals`, `match_prob`, `odds_table`, projected path (`simulate_and_write`); delete `sim_series`, `swiss_pairings`, `sim_swiss`, `sim_knockout`, `run`.
- [ ] Run -> pass. Commit.

### Task 3: `swiss.py` (config, results, outputs)
**Files:** Create `swiss.py`; extract `bracket.find_series` from `bracket.resolve_results` (no behaviour change).
- [ ] Check script with a synthetic series log: results matched inside the window only; a Bo3 at 1-0 not final; winner inferred from a later round's labeled draw; unmatched series -> note; invalid draws rejected (wrong tier, same league in R1, rematch, wrong record).
- [ ] Implement `load_config`, `build_field`, `resolve`, `known_state` (real facts only; what-if pruning in Task 9), `write_outputs`, CLI.
- [ ] Run -> pass; confirm Play-In results still resolve identically. Commit.

### Task 4: Pipeline wiring + history + requirements
**Files:** `pipeline.py`, `history.py`, `requirements.txt`.
- [ ] Wire: after the Play-In field stage, `swiss.write_outputs` guarded; fall back to projected `simulate_and_write` when `swiss` config absent; `history.record(year, odds=True)` skips odds on guarded failure; add `numpy`, `streamlit>=1.61`.
- [ ] Run `.venv/bin/python pipeline.py` (fast path, no fetch): completes, odds CSV has 19 teams with `field=confirmed`; `backtest.py --quick` output identical to before. Commit.
- [ ] Push, open PR for Phase 1.

## Phase 2 - board component + Worlds hub

### Task 5: Component spike (throwaway)
- [ ] Scratch app with two clickable cards via `st.components.v2.component`; verify click -> `on_*_change` callback -> one rerun with the pick, theme vars in light/dark, content height, 390px width. If this fails: STOP and ask.

### Task 6: `board.py` + assets
**Files:** `board.py`, `assets/board.js`, `assets/board.css`.
- [ ] Check (AppTest): board mounts with sample data; trigger callbacks deliver `pick`/`action`/`team`.
- [ ] Implement generic renderer per spec (columns/boxes/matches/rows, likely chips, action button, connectors, textContent-only DOM, keyboard-focusable rows).
- [ ] Commit.

### Task 7: Worlds hub (Play-In + Title odds)
**Files:** `worlds_view.py`, `app.py`.
- [ ] Hub page with stage switcher, shared what-if dict in `session_state`, Play-In board from `bracket`, title-odds stage via cached `simulate`; remove `page_bracket`/`bracket_card`/`bracket_row`/`page_odds`.
- [ ] AppTest: renders, baseline equals committed CSV. Commit; push; PR for Phase 2.

## Phase 3 - Swiss + knockout, play-through

### Task 8: Swiss + knockout boards
**Files:** `worlds_view.py`, `app.py`.
- [ ] Swiss board (rounds 1-5 + final records, likely chips, "Draw round N"), knockout board with connectors, team table.

### Task 9: What-if state, pruning, play-out
**Files:** `swiss.py`, `app.py`.
- [ ] Check script: real beats what-if; changing round-r result drops later hypothetical draws+picks; Play-In winner change drops hypothetical R1 draw; QF draw invalid vs records dropped.
- [ ] Implement `known_state` what-if handling and `play_out`; wire Draw/Play-out/Reset. AppTest + browser check. Commit; push; PR for Phase 3.

### Task 10: Docs
- [ ] Update CLAUDE.md (pipeline, modules, in-event workflow) and README. Commit with the last PR.
