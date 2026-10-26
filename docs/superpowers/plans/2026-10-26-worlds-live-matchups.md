# Worlds Live Matchups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Worlds page picks up every real Swiss draw, result and the quarterfinal draw from the lolesports.com schedule within minutes. Users can also choose a round's matchups by hand as a what-if, instead of only drawing one at random.

**Architecture:** `lolesports.live_facts` turns the public schedule into a `swiss.Live`: draws by round with each team's record going in, results, and the QF draw. `swiss.resolve_live` merges that under `worlds_swiss.toml`. The config always wins, and a schedule fact that fails validation is dropped with a page note; it never raises. The app reads the schedule through a 5-minute `st.cache_data`. The pipeline reads it too and commits it as `{year}_worlds_schedule.json`, which is the app's fallback. Manual matchups reuse the existing what-if storage (`wi["draws"][r]`, `wi["qf"]`) through two new validated setters, `swiss.set_draw` and `swiss.set_quarterfinals`, opened from new board actions as an `st.dialog`.

**Tech Stack:** Python 3.14 (`.venv`), pandas, numpy, Streamlit 1.61.1 (`st.dialog`, `st.components.v2`, `streamlit.testing.v1.AppTest`), and stdlib `urllib`. No new dependencies.

**Spec:** none separate. The Context, Global Constraints and Interfaces sections below are the spec.

## Context

**Symptom:** the Worlds tab does not follow new matchups.

**Root cause** (verified 2026-10-26):
- Draws reach the app only through hand-entered `[draws.N]` tables in `worlds_swiss.toml`. On `main` that file still has only `[draws.1]`.
- `lolesports.py` already reads the schedule, but by design it only *prints* TOML.
- The `update.yml` cron runs `pipeline.py`, which never touches draws.
- The committed series log on `main` already has all 8 Round 1 results, all 8 Round 2 results and 4 of 8 Round 3 results. `swiss.resolve` can only attach a result to a drawn pair, so 12 real series show up only as "matches no drawn pair -- add that round's draw" notes.
- As a result, `mc.simulate` re-simulates Rounds 2–5 from random draws (`_draw_by_record`), and the board offers a random "Draw round 2". The page is two rounds stale while the data is already there.
- lolesports.com currently shows Rounds 1–3 fully drawn. Rounds 4–5 and the knockout slots are TBD.

**User decisions:** do both (auto-sync and a manual matchup picker). The auto-sync runs live in the app (and in the pipeline), and the config stays authoritative.

**Stopgap until this ships (optional):** paste the output of `python lolesports.py` into `worlds_swiss.toml` and push. The checks below work either way.

**Non-goals:**
- Live sync for the Play-In (it is over; `bracket.py` is unchanged).
- The `pickem.py` / `crystal_ball.py` CLIs keep using config + Oracle's Elixir only.
- No changes to the engine, the ratings, or the `update.yml` schedule.
- No TOML writer.

## Global Constraints

- Run everything with `.venv/bin/python` from `/Users/zack/projects/league-predictor`. `requirements.txt` does not change.
- **The config wins:** a draw, QF draw or `[results.*]` entry in `worlds_swiss.toml` always beats the schedule. A schedule fact that fails validation is dropped with a note; it never raises in the app or the pipeline.
- **Result priority:** manual `[results.*]` (`"manual"`), then the Oracle's Elixir series log (`"auto"`), then lolesports.com (`"live"`), then inferred from the next draw (`"draw"`).
- **Schedule cache:** `SCHEDULE_TTL = 300` seconds, one fetch shared by all sessions. The app fetches with `timeout=5`; the pipeline uses the default 30.
- **Snapshot:** `paths.processed(year, "worlds_schedule", "json")`. It is written only by `swiss.write_outputs` (the pipeline or `python swiss.py`), never by the app.
- **QF slots from the schedule** follow start-time order (QF1+QF2 feed SF1, the 2025 precedent). They are applied only if they pass `check_quarterfinals` and the config has no QF draw.
- **Manual draws are what-ifs:** session-only, stored in the same `{"pairs": [[a, b], ...], "premise": [...]}` shape that `draw_round` / `draw_quarterfinals` use, so `swiss.state` prunes them unchanged.
- **The manual picker covers rounds 2–5 and the quarterfinals.** Round 1 keeps the random draw.
- **Import cycle:** `lolesports` imports `swiss` at module level, so `swiss` may import `lolesports` only inside `write_outputs`.
- **Tests:** there is no committed test suite, and this plan does not add one. Check scripts live in `$SP` (the executing session's scratchpad, or `$(mktemp -d)`) and are not committed. Re-export `SP` in every new shell.
- **Generated data:** never commit regenerated data (`data/processed/*`, `data/history/*`); Actions owns it. Stage files by name, never `git add -A`.
- **Git:** work on branch `worlds-live-matchups` and commit once per task. Commit messages are plain sentences (repo style) plus the co-author trailer your harness specifies. Do not push or open a PR unless the user asks.

## Review Focus

1. **lolesports.com unreachable or slow from Streamlit Cloud.** The page must still render, from the snapshot or the config, within the 5 s timeout, and the status caption must say which source it used. Tested in Task 3.
2. **A partly drawn round (TBD slots) or an in-progress Bo3 (1-0).** The round counts as undrawn and the series has no result. Tested in Task 1.
3. **The schedule disagrees with a hand-entered draw or result.** The config wins and a note appears; schedule rounds built on the disputed fact are dropped, never crashed on. Tested in Task 2.
4. **Start-time QF order puts both 3-0 teams in one half.** The schedule's QF draw is ignored with a note. Tested in Task 2 (synthetic).
5. **A manual draw whose premise changes** (a pick in the round before flips, or the real draw arrives). It is dropped, never trusted. Tested in Task 4.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `lolesports.py` | modify | `walk_swiss`, `api_quarterfinals`, `live_facts`, snapshot I/O, `schedule()`; `fetch_events(timeout=)`; the CLI checks reuse the walkers |
| `swiss.py` | modify | `Live`, `merge_live`, `live_conflicts`, `resolve_live`, `check_quarterfinals` (extracted), live result source in `resolve`, `write_outputs` uses the schedule, `met_before`, `draw_groups`, `set_draw`, `set_quarterfinals` |
| `worlds_view.py` | modify | `SOURCES["live"]`; `pick:<r>` / `pick:qf` board actions |
| `app.py` | modify | `SCHEDULE_TTL`, `fetch_schedule`, `load_event` returns a status, `on_action` dispatch, `matchups_dialog`, `page_worlds` status caption + dialog |
| `CLAUDE.md`, `worlds_swiss.toml` (comments only), `lolesports.py` docstring | modify | docs |

## Interfaces (fixed here; later tasks rely on them)

```python
# swiss.py
@dataclass
class Live:
    """Facts from the lolesports.com schedule (lolesports.live_facts), merged under the config."""
    draws: dict[int, list[tuple[str, str]]]        # complete rounds only, 1..k with no gaps
    labels: dict[int, dict[str, tuple[int, int]]]  # round -> team -> record going in
    results: dict[frozenset, dict]                 # Swiss pair -> {"winner": str, "score": "1-0"}
    qf: list[tuple[str, str]] | None               # QF1..QF4 in schedule start-time order
    ko: dict[frozenset, dict]                      # knockout pair -> {"winner", "score"}
    # every field defaults empty / None via dc_field(default_factory=...)

def check_quarterfinals(qf: list[tuple[str, str]], rec: dict[str, tuple[int, int]]) -> None  # ValueError
def merge_live(cfg: dict, live: Live, rounds: int, qf: bool) -> dict    # new dict; never mutates cfg
def live_conflicts(cfg: dict, live: Live) -> list[str]
def resolve_live(cfg: dict, series: pd.DataFrame, year: int = CURRENT_YEAR,
                 rows: pd.DataFrame | None = None, live: Live | None = None) -> Event
def met_before(known: mc.Known, r: int) -> set[frozenset]                # pairs drawn in rounds < r
def draw_groups(stt: State, r: int) -> list[tuple[tuple[int, int], list[str]]]
def set_draw(stt: State, wi: dict, r: int, pairs: list[tuple[str, str]]) -> None   # ValueError
def set_quarterfinals(stt: State, wi: dict, pairs: list[tuple[str, str]]) -> None  # ValueError
# cfg keys added by merge_live and read by resolve with .get(..., {}):
#   "results_live_swiss": {frozenset: {"winner", "score"}}, "results_live_ko": same

# lolesports.py
def fetch_events(since: str, timeout: float = 30) -> list[dict]          # existing, + timeout
def walk_swiss(matches: list[Match]) -> list[tuple[int, list[Match], dict[str, tuple[int, int]]]]
def api_quarterfinals(matches: list[Match]) -> list[tuple] | None
def live_facts(cfg: dict, bcfg: dict, events: list[dict]) -> swiss.Live  # ValueError on an unknown team code
def snapshot_path(year: int) -> Path
def save_snapshot(year: int, events: list[dict], since: str) -> None
def load_snapshot(year: int) -> list[dict] | None
def schedule(year: int, since: str) -> tuple[list[dict] | None, str]     # (events, where they came from)

# worlds_view.py
SOURCES["live"] = "from lolesports.com"
# board action ids: existing "draw:<r>", "draw:qf" (random); new "pick:<r>" (r >= 2), "pick:qf" (open the dialog)

# app.py
SCHEDULE_TTL = 300
def fetch_schedule(since: str) -> tuple[list[dict] | None, str]          # @st.cache_data(ttl=SCHEDULE_TTL)
def load_event() -> tuple[swiss.Event | None, str]                       # (event, status caption)
def matchups_dialog(stt, target: str) -> None                            # @st.dialog; target "2".."5" or "qf"
```

---

### Task 0: Branch, plan, fixtures

- [ ] **Step 1: Branch from a fresh `main`.** Local refs go stale because Actions commits data daily. The current branch, `swiss-pickem-optimizer`, is already merged.
```bash
cd /Users/zack/projects/league-predictor
git fetch origin
git switch -c worlds-live-matchups origin/main
```
- [ ] **Step 2:** Copy this plan to `docs/superpowers/plans/2026-10-26-worlds-live-matchups.md`, then `git add` that file and commit: "Add the Worlds live matchups plan".
- [ ] **Step 3: Capture the fixtures.** This needs network access; if `--json` fails, stop and report.
```bash
SP=<scratchpad>   # or: SP=$(mktemp -d)
.venv/bin/python lolesports.py --json > "$SP/schedule.json"
.venv/bin/python lolesports.py --file "$SP/schedule.json" > "$SP/cli_before.txt"; echo "exit=$?" >> "$SP/cli_before.txt"
```
Expected: `schedule.json` is a JSON list, and `cli_before.txt` ends with `exit=0` or `exit=1`.

Facts the checks pin (real history, stable regardless of capture time):
- **R2 draw**, 1-0 group: MKOI v CFO, KC v LYON, HLE v GEN, DK v T1.
- **R2 draw**, 0-1 group: TES v AL, LØS v G2, TSW v IG, TL v BLG.
- **R3 draw**, 2-0 group: MKOI v DK, LYON v GEN.
- **R3 draw**, 1-1 group: IG v HLE, TES v G2, CFO v T1, BLG v KC.
- **R3 draw**, 0-2 group: LØS v TSW, TL v AL.
- **R1 winners:** GEN, HLE, MKOI, KC, DK, CFO, T1, LYON.
- **R2 winners:** TES, MKOI, G2, LYON, IG, GEN, DK, BLG.
- **R3 so far:** IG beat HLE, TES beat G2, DK beat MKOI 2-0, GEN beat LYON 2-1.

All names in code are the config names: `"Movistar KOI"`, `"CTBC Flying Oyster"`, `"Karmine Corp"`, `"LYON"`, `"Hanwha Life Esports"`, `"Gen.G"`, `"Dplus Kia"`, `"T1"`, `"Top Esports"`, `"Anyone's Legend"`, `"LØS"`, `"G2 Esports"`, `"Team Secret Whales"`, `"Invictus Gaming"`, `"Team Liquid"`, `"Bilibili Gaming"`.

### Task 1: Read the schedule into `swiss.Live`

**Files:**
- Modify `lolesports.py`: `_page`, `fetch_events`, `check_swiss`, `check_knockout`; add `walk_swiss`, `api_quarterfinals`, `live_facts`.
- Modify `swiss.py`: add `Live` after `Event`.
- Check: `$SP/check_live.py`.

**Interfaces:**
- Consumes: the existing `parse`, `swiss_rounds`, `Match`, `KO_BLOCKS`, `mc._layout`.
- Produces: `swiss.Live`, `walk_swiss`, `api_quarterfinals`, `live_facts`, and `fetch_events(since, timeout)`.

- [ ] **Step 1:** Write `$SP/check_live.py`. It takes `SP` as `sys.argv[1]`, loads `cfg, bcfg = swiss.load_config(), bracket.load_config()` and the fixture, and defines `P = lambda ps: {frozenset(p) for p in ps}`. It asserts:
  - `P(live.draws[1]) == P(cfg["draws"][1])`.
  - `P(live.draws[2])` and `P(live.draws[3])` equal the Task 0 draws.
  - `live.labels[2]["Gen.G"] == (1, 0)`, `live.labels[3]["Movistar KOI"] == (2, 0)`, `live.labels[3]["T1"] == (1, 1)` and `live.labels[3]["LØS"] == (0, 2)`.
  - Every R1 and R2 winner from Task 0 matches `live.results[frozenset(pair)]["winner"]`.
  - `live.results[frozenset(("Movistar KOI", "Dplus Kia"))] == {"winner": "Dplus Kia", "score": "2-0"}`.
  - No fixture event with `state != "completed"` has its pair in `live.results`.
  - **TBD round:** after setting one R3 event's first team `code` to `"TBD"` (deep-copied fixture), `3 not in live.draws` and `2 in live.draws`.
  - **In-progress Bo3:** after setting the MKOI–DK event to `state="inProgress"`, with `gameWins` 1/0 and outcomes `None`, the pair is not in `results` and `4 not in live.draws`.
  - **Mismatched records:** after swapping CFO and AL between the R2 events "MKOI v CFO" and "TES v AL", `2 not in live.draws` and `1 in live.draws`.
  - **Unknown team code:** a code of `"XXX"` makes `live_facts` raise a `ValueError` whose message contains `"unknown team code"`.
- [ ] **Step 2:** Run `.venv/bin/python "$SP/check_live.py" "$SP"`. Expected: FAIL (`AttributeError ... live_facts`).
- [ ] **Step 3: Implement.**
  - `fetch_events(since, timeout=30)` passes `timeout` through `_page(token, timeout)` to `urlopen`.
  - **`walk_swiss(matches)`** returns `(r, matches_in_round, rec_in)` for each round in order. `rec_in` maps every team in the round to its record going in (0-0 for teams not seen yet).
    - Stop *before* a round that has a TBD slot or fewer matches than `sum(s for _, _, s in mc._layout(r)) // 2`.
    - Stop *after* the first round with an unfinished match.
    - Records come only from the schedule's own results.
  - **`check_swiss`:** rewrite it on top of `walk_swiss` (`ra, rb = rec_in[a], rec_in[b]`). Its output on the fixture must not change.
  - **`api_quarterfinals`:** lift it from `check_knockout`'s `api_qf` (all 4 QFs with both teams known, sorted by start time, else `None`). `check_knockout` then uses it.
  - **`live_facts(cfg, bcfg, events)`:**
    - Build `names` and `since` as `run()` does, then call `parse`.
    - For each walked round: if any match's two teams have different `rec_in`, stop (add neither this round nor any later one). Otherwise set `draws[r] = [m.teams ...]` and `labels[r] = rec_in`, and add a result for each done match.
    - `qf = api_quarterfinals(matches)`.
    - `ko` gets every done match in `KO_BLOCKS`, keyed by `frozenset(m.teams)`.
    - A `ValueError` from `parse` propagates.
- [ ] **Step 4:** Run the check. Expected: `PASS`. Then check that the CLI output is unchanged:
```bash
.venv/bin/python lolesports.py --file "$SP/schedule.json" > "$SP/cli_after.txt"; echo "exit=$?" >> "$SP/cli_after.txt"
diff "$SP/cli_before.txt" "$SP/cli_after.txt" && echo SAME
```
Expected: `SAME`.
- [ ] **Step 5:** Commit `lolesports.py swiss.py` with the message "Read Swiss draws and results from the lolesports.com schedule".

### Task 2: Merge the schedule under the config

**Files:**
- Modify `swiss.py`: `resolve` (result lookups and the QF check); add `check_quarterfinals`, `merge_live`, `live_conflicts`, `resolve_live`.
- Modify `worlds_view.py`: add to `SOURCES`.
- Check: `$SP/check_resolve_live.py`.

**Interfaces:**
- Consumes: `swiss.Live` and `lolesports.live_facts`.
- Produces: `check_quarterfinals`, `merge_live`, `live_conflicts` and `resolve_live` (signatures above), plus result source `"live"`.

- [ ] **Step 1:** Write `$SP/check_resolve_live.py`. Set `series = swiss.load_series(2026)`, `empty = pd.DataFrame(columns=swiss.SERIES_COLS)` and `live = live_facts(cfg, bcfg, fixture)`. It asserts:
  1. `resolve_live(cfg, series, 2026, live=None)` has the same `known` and `notes` as `resolve(cfg, series, 2026)`.
  2. **The live merge:** `ev = resolve_live(cfg, series, 2026, live=live)`:
     - `{1, 2, 3} <= set(ev.known.draws)`.
     - 8 results each in rounds 1 and 2.
     - No note contains `"matches no drawn pair"`.
     - If `cfg["draws"]` has only round 1, the config-only resolve has `>= 12` such notes (the bug, reproduced).
  3. **Live results fill an empty log:** with `empty`, every round-2 result's `info["swiss"][2][pair]["source"] == "live"`, and the R1 winners match Task 0.
  4. **A manual result beats the schedule:** set `cfg4 = {**cfg, "results_swiss": {frozenset(("Hanwha Life Esports", "Gen.G")): {"winner": "Hanwha Life Esports", "score": "1-0"}}}`. Then with `series`:
     - That R2 pair's winner is HLE, with source `"manual"`.
     - `3 not in ev.known.draws`.
     - A note starts with `"Ignored the lolesports.com draw for round 3"`.
  5. **An injected rematch:** in a deep copy of `live`, replace R3 pairs {IG, HLE} and {TES, G2} with (IG, G2) and (TES, HLE). TES v HLE replays Round 1. Then `3 not in draws`, `2 in draws`, and a note contains `"rematch"`.
  6. **A config draw that differs:** build `cfg6` with `draws[2]` = the live R2 draw with partners swapped inside the 1-0 group ((MKOI, LYON), (KC, CFO)), and `labels[2] = live.labels[2]`. With `empty`:
     - A note starts with `"Round 2: the draw in worlds_swiss.toml differs from lolesports.com"`.
     - `ev.known.draws[2] == cfg6["draws"][2]`.
  7. **A synthetic full event:**
     - `base = resolve(cfg, series, 2026)` and `tr = mc.run_trace(base.field, base.known, seed=1)`.
     - The script helper `fake_events(tr, cfg, bcfg)` emits one `state="completed"` event per series:
       - Swiss rounds 1–5 in `tr.draws` order, `startTime=f"2026-10-{22 + r}T{10 + j:02d}:00:00Z"`.
       - QF1–QF4 on `2026-11-0{3 + i}T22:00:00Z`, SF1 and SF2 on 11-07 and 11-08, the Final on 11-14.
       - Each event has `blockName` `"Swiss"`, `"Quarterfinals"`, `"Semifinals"` or `"Finals"`, team `code` = the config `short`, and `result = {"gameWins": 1 or 0, "outcome": "win"/"loss"}` from `tr.winners` / `tr.ko`.
     - Then `resolve_live(cfg, empty, 2026, live=live_facts(cfg, bcfg, fake))` has:
       - `sorted(draws) == [1, 2, 3, 4, 5]`.
       - `known.qf == tr.qf` and `known.ko == tr.ko`.
       - Every KO source is `"live"`.
  8. **The same synthetic event with both 3-0 teams in one half:** give the two QF events that contain a 3-0 team (from `swiss.records` of item 7's event) the two earliest QF start times. Then `known.qf is None`, and a note starts with `"Ignored the lolesports.com quarterfinal draw"` and contains `"opposite halves"`.
  9. **`check_quarterfinals` on a hand-built `rec`:**
     - `rec`: A and B 3-0; C, D and E 3-1; F, G and H 3-2; X 2-3.
     - `[("A","F"),("C","D"),("B","G"),("E","H")]` passes.
     - `[("A","F"),("B","G"),("C","D"),("E","H")]` raises a message containing `"opposite halves"`.
     - `[("A","C"),("F","D"),("B","G"),("E","H")]` raises one containing `"must face a 3-2 team"`.
     - Swapping H for X raises one containing `"not the eight teams on 3 wins"`.
- [ ] **Step 2:** Run it. Expected: FAIL (`AttributeError ... resolve_live`).
- [ ] **Step 3: Implement.**
  - **`check_quarterfinals(qf, rec)`:** first check for 4 pairs of 8 distinct teams, then move `resolve`'s inline checks (currently `swiss.py:308-319`) into it, keeping the messages verbatim. `resolve` calls it where the inline code was (only when `rec is not None`).
  - **`resolve`, Swiss results:** for each pair, use `manual`, then `found` (Oracle's Elixir), then `cfg.get("results_live_swiss", {}).get(frozenset((a, b)))` (source `"live"`), then the `_won` inference. If `found` and a live result disagree, keep Oracle's Elixir and add the note `f"Round {r}: {a} v {b} -- Oracle's Elixir says {x} won, lolesports.com says {y}; using Oracle's Elixir."`
  - **`resolve`, knockout results:** same order, with `results_live_ko` keyed by `frozenset(pair)`. The disagreement note uses `f"{key}: {a} v {b} -- ..."`.
  - **`merge_live(cfg, live, rounds, qf)`:**
    - Extend `draws`/`labels` with `live.draws[r]`/`live.labels[r]` for r from `len(cfg["draws"]) + 1` up to `rounds`, stopping at the first missing round.
    - Set `results_live_swiss = live.results` and `results_live_ko = live.ko`.
    - Set `quarterfinals = live.qf` only when `qf` is true, `cfg["quarterfinals"] is None`, and the merged draws cover `mc.ROUNDS`.
    - Build every dict new.
  - **`live_conflicts(cfg, live)`:** for each config round also in `live.draws` whose pair set differs, add `f"Round {r}: the draw in {CONFIG.name} differs from lolesports.com; using {CONFIG.name}."` For a config QF whose pair set differs from `live.qf`, add `f"Quarterfinals: the draw in {CONFIG.name} differs from lolesports.com; using {CONFIG.name}."`
  - **`resolve_live`:**
    1. If `live is None`, return `resolve(...)`.
    2. If `rows is None`, build it once with `field_rows(cfg, bracket.load_config(), pd.read_parquet(paths.processed(year, "calibrated_ratings")))`.
    3. Try `resolve(merge_live(cfg, live, mc.ROUNDS, True), ...)`. On success, prepend the `live_conflicts` notes and return.
    4. On `ValueError`, run `resolve(cfg, ...)` alone and let it raise: a broken config fails as it does today.
    5. Then, for `k` from `len(cfg["draws"])` to `max(live.draws, default=0)`, resolve `merge_live(cfg, live, k, False)`, keeping the last success.
    6. Stop at the first failure and note it:
       - At `k == len(cfg["draws"])`: `f"Ignored the lolesports.com results: {exc}"`.
       - At a later `k`: `f"Ignored the lolesports.com draw for round {k} and later: {exc}"`.
    7. If every `k` passed, the QF draw caused step 3's failure; note `f"Ignored the lolesports.com quarterfinal draw: {exc}"` using step 3's exception.
    8. Return the last good Event with `notes = conflicts + ignored + its own notes`.
  - **`worlds_view.SOURCES`:** add `"live": "from lolesports.com"`.
- [ ] **Step 4:** Run it. Expected: `PASS`. Re-run `check_live.py`. Expected: `PASS`.
- [ ] **Step 5:** Commit `swiss.py worlds_view.py` with the message "Merge the lolesports.com schedule under the Swiss config".

### Task 3: Wire the schedule into the app and the pipeline

**Files:**
- Modify `lolesports.py`: add `snapshot_path`, `save_snapshot`, `load_snapshot`, `schedule`.
- Modify `swiss.py`: `write_outputs`.
- Modify `app.py`: `load_event` (`app.py:446`), `page_worlds` (`app.py:733`); add `SCHEDULE_TTL` and `fetch_schedule` next to `load_event`.
- Check: `$SP/check_app.py` (AppTest) and pipeline runs.

**Interfaces:**
- Consumes: `live_facts` and `resolve_live`.
- Produces: `load_event() -> (event, status)`, plus the snapshot file.

- [ ] **Step 1:** Write `$SP/check_app.py`. Before each scenario, call `st.cache_data.clear()`, monkeypatch `lolesports` attributes, then run `AppTest.from_file("app.py", default_timeout=180)` with `at.session_state["page"] = "Worlds"`. In every scenario, assert `not at.exception`.
  - **(a)** `lolesports.fetch_events = lambda since, timeout=30: fixture`: some `at.caption` value contains `"last checked"`.
  - **(b)** `fetch_events` raises `urllib.error.URLError("offline")` and `load_snapshot = lambda year: None`: a caption contains `"worlds_swiss.toml only"`.
  - **(c)** As (b), but `load_snapshot` returns the fixture: a caption contains `"last pipeline run"`.
  - **(d)** The fixture with one Swiss team `code` set to `"XXX"`: a caption contains `"could not be read"`.
- [ ] **Step 2:** Run it. Expected: FAIL (no status caption).
- [ ] **Step 3: Implement.**
  - **`snapshot_path(year)`** returns `paths.processed(year, "worlds_schedule", "json")`.
  - **`save_snapshot`** keeps events with `blockName in BLOCKS` and `startTime >= since`, sorted by start time, and writes `json.dumps(kept, indent=1, sort_keys=True, ensure_ascii=False)` to a temp file, then `.replace()`s the target.
  - **`load_snapshot`** returns `None` when the file is missing.
  - **`schedule(year, since)`:**
    - Calls `fetch_events`, then `save_snapshot`, and returns `(events, "live from lolesports.com (snapshot saved)")`.
    - On `(urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc`, it loads the snapshot instead. If one exists, it returns `(events, f"lolesports.com unreachable ({exc!r}); using the last snapshot")`. If not, it returns `(None, f"lolesports.com unreachable ({exc!r}); no snapshot, config only")`.
  - **`swiss.write_outputs`:**
    - Lazy `import lolesports`, with a comment explaining the cycle.
    - `bcfg = bracket.load_config()`, then `events, src = lolesports.schedule(year, str(bcfg["start"]))`.
    - If there are events, `live = lolesports.live_facts(cfg, bcfg, events)`. On `ValueError`, set `live = None` and append `f"; could not be read: {exc}"` to `src`.
    - `ev = resolve_live(cfg, load_series(year), year, live=live)`.
    - Print `f"Schedule: {src}"` before the existing `Play-In winner ...` line.
  - **`app.py`:**
    - Add `SCHEDULE_TTL = 300` with a comment saying one read serves every visitor for 5 minutes.
    - Add `@st.cache_data(ttl=SCHEDULE_TTL, show_spinner=False) def fetch_schedule(since)`. It returns `(lolesports.fetch_events(since, timeout=5), f"{dt.datetime.now(dt.timezone.utc):%H:%M} UTC")`, or `(None, repr(exc))` on any `Exception`; the page must render.
    - Add `import lolesports` to the imports.
  - **`load_event()` source order:** the cached live fetch; otherwise `lolesports.load_snapshot(YEAR)`; otherwise no schedule. Then `lolesports.live_facts`, where a `ValueError` gives the "could not be read" status and `live=None`.
  - **`load_event()`** returns `(swiss.resolve_live(cfg, load_series(), YEAR, rows=rows, live=live), status)`, or `(None, "")` when there are no rows. Exact status copy:
    - Live: `f"Draws and results update from lolesports.com (last checked {stamp})."`
    - Snapshot: `"lolesports.com could not be reached; draws and results are from the last pipeline run."`
    - Neither: `"lolesports.com could not be reached; draws are from worlds_swiss.toml only."`
    - `ValueError` from `live_facts`: `f"The lolesports.com schedule could not be read ({exc}); draws are from worlds_swiss.toml only."`
  - **`page_worlds`:** unpack `event, status = load_event()`. After the `event is None` guard, add `st.caption(status)`.
- [ ] **Step 4:** Run `check_app.py`. Expected: `PASS`. Then run the pipeline stage by hand:
```bash
.venv/bin/python swiss.py | head -3      # expect "Schedule: live from lolesports.com (snapshot saved)" and "rounds drawn: [1, 2, 3"
test -s data/processed/2026_worlds_schedule.json && echo SNAPSHOT
.venv/bin/python -c "import urllib.error, lolesports, swiss
def off(*a, **k): raise urllib.error.URLError('offline')
lolesports.fetch_events = off; swiss.write_outputs(2026, 2000, 0)" | head -3   # expect "using the last snapshot", same rounds drawn
git checkout -- data/processed && rm -f data/processed/2026_worlds_schedule.json && git status --short
```
Expected: `SNAPSHOT`, both runs list the same rounds, and `git status` shows only the code files.
- [ ] **Step 5:** Commit `lolesports.py swiss.py app.py` with the message "Sync the Worlds page and odds with the lolesports.com schedule".

### Task 4: Validated manual draws (`swiss.py`)

**Files:**
- Modify `swiss.py`: add `met_before`, `draw_groups`, `set_draw`, `set_quarterfinals` in the what-ifs section.
- Check: `$SP/check_matchups.py`.

**Interfaces:**
- Consumes: `_check_round`, `_known_before`, `_premise`, `_qf_premise`, `records`, `check_quarterfinals`.
- Produces: the four functions above.

- [ ] **Step 1:** Write `$SP/check_matchups.py` with a deterministic event:
  - `live_t` is a deep copy of the fixture's `live` with `draws`/`labels` cut to rounds ≤ 3, every R3 pair removed from `results`, `qf=None` and `ko={}`.
  - `ev = resolve_live(cfg, empty, 2026, live=live_t)`, so R1 and R2 are decided and R3 is drawn but open.

  It asserts:
  - With `wi = swiss.new_whatif()` and `stt = swiss.state(ev, wi)`, `set_draw(stt, wi, 4, any_pairs)` raises `"cannot be drawn until"`.
  - **After picking the first team of every R3 pair** (`wi["picks"][swiss.pick_key(3, a, b)] = a`, then re-run `state`):
    - `dict(swiss.draw_groups(stt, 4))` has keys `{(2, 1), (1, 2)}`, 6 teams each, sorted by rating (descending).
  - **A legal draw:** the script helper builds `legal` from each group's first perfect matching that avoids `swiss.met_before(stt.known, 4)` (brute force).
    - `set_draw(stt, wi, 4, legal)` sets `wi["draws"][4]["premise"] == sorted(stt.known.results[3].values())`.
    - `swiss.state(ev, wi)` has `4 in hypo_draws` and the same pair set.
  - **Illegal draws** (each on a fresh copy of `wi`):
    - A rematch raises `"rematch"`. Use a `met_before` pair whose two teams share a group; with these picks, Gen.G v Invictus Gaming (Round 1) is in the 2-1 group, and G2–LØS and T1–LØS are in the 1-2 group.
    - A pair across groups (e.g. Dplus Kia v Hanwha Life Esports) raises `"same record"`.
    - A team used twice raises `"drawn twice"`.
    - `legal[:-1]` raises `"expected"`.
    - `set_draw(stt, wi, 2, stt.known.draws[2])` raises `"already drawn"`.
  - **The premise changes:** flipping one R3 pick drops round 4 (`4 not in swiss.state(ev, wi).known.draws`).
  - **The real draw arrives:** with the manual R4 draw still in `wi`, build `ev2` from `live_t` plus `draws[4]` = a legal pairing that differs from `legal`, and `labels[4]` = the records after the R3 picks. R3's results are then inferred from those labels. `s = swiss.state(ev2, wi)` must have `4 not in s.hypo_draws`, and `s.known.draws[4]` must equal the live pairs.
  - **The quarterfinals:**
    - `wf = swiss.play_out(ev, swiss.new_whatif(), np.random.default_rng(7))`, then `wf["qf"], wf["ko"] = None, {}`, `s = swiss.state(ev, wf)` and `rec = swiss.records(s.known)`.
    - With z = the 3-0 teams, t1 = the 3-1 teams and t2 = the 3-2 teams, `good = [(z[0], t2[0]), (t1[0], t1[1]), (z[1], t2[1]), (t1[2], t2[2])]`.
    - `set_quarterfinals(s, wf, good)`, then `swiss.state(ev, wf)` has `hypo_qf` and `known.qf == good`.
    - The 3-0 teams in QF1 and QF2 raise `"opposite halves"`.
    - Pairing a 3-0 team with a 3-1 team raises `"3-2"`.
- [ ] **Step 2:** Run it. Expected: FAIL (`AttributeError ... draw_groups`).
- [ ] **Step 3: Implement.**
  - **`met_before(known, r)`** returns the frozensets of the pairs in `known.draws[k]` for `k < r`.
  - **`draw_groups(stt, r)`** returns `[((w, l), teams)]` in `mc._layout(r)` order. Each team list is the Swiss teams on that record, from `records(stt.known, r - 1)` (everyone is 0-0 for r = 1), sorted by rating descending. The Swiss teams are `stt.field.teams[:mc.DIRECT] + [stt.known.playin_winner]`.
  - **`set_draw`** runs its checks in this order:
    1. `r in stt.known.draws and r not in stt.hypo_draws` raises `f"round {r} is already drawn"`.
    2. `before = _known_before(stt.known, r)`. If the premise is `None`, it raises `f"round {r} cannot be drawn until the round before it is decided"`.
    3. For each team: unknown raises `f"{t!r} is not in the Swiss stage"`; a finished record (3 wins or 3 losses) raises `f"{t} has finished the Swiss stage"`; a repeat raises `f"{t} is drawn twice"`. For each pair, unequal records raise `f"{a} ({w}-{l}) and {b} ({w2}-{l2}) are not on the same record"`.
    4. Then it calls `_check_round(r, pairs, lab, rec, met_before(before, r), tier, league, stt.field.league_rule)` with:
       - `lab = rec` restricted to the drawn teams;
       - `tier = dict(zip(field.teams[:DIRECT], field.tier)) | {winner: 4}`;
       - `league = dict(zip(field.teams, field.league))`.

       It stores `{"pairs": [list(p) ...], "premise": premise}`, as `draw_round` does.
  - **`set_quarterfinals`:**
    1. A real QF draw (`stt.known.qf is not None and not stt.hypo_qf`) raises `"the quarterfinals are already drawn"`.
    2. `before = replace(stt.known, qf=None, ko={})`. If `_qf_premise(before)` is `None`, it raises `"the quarterfinals cannot be drawn until the Swiss stage is decided"`.
    3. It calls `check_quarterfinals(pairs, records(before))`.
    4. It stores `wi["qf"] = {"pairs": ..., "premise": ...}`.
- [ ] **Step 4:** Run it. Expected: `PASS`. Re-run the Task 1–2 checks. Expected: `PASS`.
- [ ] **Step 5:** Commit `swiss.py` with the message "Add validated manual Swiss and quarterfinal draws".

### Task 5: "Choose matchups" actions and dialog

**Files:**
- Modify `worlds_view.py`: `swiss_board` (`:145-151`) and `knockout_board` (`:285-291`).
- Modify `app.py`: `worlds_callbacks.on_action` (`:513`), new `matchups_dialog`, and `page_worlds` (the caption at `:753-755`, plus opening the dialog).
- Check: `$SP/check_board.py`, AppTest, and a browser walkthrough.

**Interfaces:**
- Consumes: `draw_groups`, `met_before`, `set_draw`, `set_quarterfinals`, `records`.
- Produces: the `pick:<r>` and `pick:qf` actions, and the session key `"worlds_matchups"`.

- [ ] **Step 1:** Write `$SP/check_board.py`. Reuse Task 4's deterministic `ev`, with R3 decided by picks.
  - Set `odds = mc.simulate(stt.field, stt.known, 2000, 0)` and `teams = worlds_view.Teams.from_rows(ev.rows)`.
  - **The Swiss board:** in `worlds_view.swiss_board(teams, ev, stt, odds)`, the Round 4 column's action ids are `{"draw:4", "pick:4"}`. Neither is disabled.
  - **Before the R3 picks:** the `"pick:4"` action has `disabled=True`.
  - **The knockout board:** with Task 4's `wf` state, `worlds_view.knockout_board(...)` has QF action ids `{"draw:qf", "pick:qf"}`.
- [ ] **Step 2:** Run it. Expected: FAIL (no `pick:4`).
- [ ] **Step 3: Implement.**
  - **`swiss_board`:** when the draw action box is added and `r >= 2`, also append `{"action": {"id": f"pick:{r}", "label": f"Choose round {r} matchups", "disabled": not ready}}`.
  - **`knockout_board`:** next to the QF draw action, append `{"action": {"id": "pick:qf", "label": "Choose quarterfinal matchups", "disabled": not qf_ready}}`.
  - **`on_action`:**
    - Use `verb, kind = p["id"].split(":", 1)`. Today's `split(":")[1]` would turn `pick:4` into a random draw.
    - If `verb == "pick"`, set `st.session_state["worlds_matchups"] = kind` and return. A dialog cannot be opened from a widget callback, so `page_worlds` opens it after the board.
    - Otherwise, keep the existing random-draw code.
  - **`matchups_dialog(stt, target)`:** decorate with `@st.dialog("Choose the matchups", width="large")`, defined at module level (titles are fixed at decoration).
    - **Swiss target (`r = int(target)`):**
      - Intro caption: `"Pair teams on the same record who have not played each other. This is a what-if: it changes the odds for you only and is not saved."`
      - For each group from `swiss.draw_groups(stt, r)`, show a bold `f"{w}-{l} · Bo{3 if 2 in (w, l) else 1}"`.
      - If any of the group's teams have met, add `st.caption("Already played each other: " + ", ".join(f"{short[a]}–{short[b]}" ...))`, with `short = dict(zip(stt.field.teams, stt.field.short))`.
      - Then show `len(teams) // 2` rows of `st.columns(2)`, each with two `st.selectbox(..., teams, index=..., key=f"mu_{r}_{w}{l}_{i}_{side}", placeholder="Choose a team", label_visibility="collapsed")`.
      - Prefill `index` from the hypothetical draw when `r in stt.hypo_draws`; otherwise `None`.
    - **QF target:**
      - Intro caption: `"QF1 and QF2 feed Semifinal 1, QF3 and QF4 feed Semifinal 2. Each 3-0 team plays a 3-2 team, and the two 3-0 teams go in different halves. This is a what-if, not saved."`
      - Show 4 rows labelled QF1–QF4. The options are the 3-win teams from `rec = swiss.records(stt.known)`, with `format_func=lambda t: f"{t} ({rec[t][0]}-{rec[t][1]})"`.
      - Prefill from `stt.known.qf` when `stt.hypo_qf`.
    - **Apply** (`st.button("Apply", type="primary")`):
      - If any box is empty, show `st.error("Choose both teams in every match.")`.
      - Otherwise call `swiss.set_draw` or `swiss.set_quarterfinals` on `worlds_whatif()`.
      - On `ValueError`, show `st.error(str(exc))`. On success, call `st.rerun()` to close the dialog and recompute the odds.
  - **`page_worlds`:**
    - Replace the idle caption with `"Click a team on any open series to play out a scenario, or draw the next round yourself, at random or by choosing the matchups. Picks are yours alone and are not saved."`
    - After the stage renders, add `target = st.session_state.pop("worlds_matchups", None)`, followed by `if target is not None: matchups_dialog(stt, target)`. Popping the key means a closed dialog does not reopen.
- [ ] **Step 4:** Run `check_board.py` and `check_app.py`. Expected: `PASS`. Then do the browser walkthrough with `.venv/bin/streamlit run app.py` (by hand or with a browser tool) and confirm each of these:
  - Worlds → Swiss shows Rounds 1–3 with real pairs and results (footers "from Oracle's Elixir" / "from lolesports.com"), plus a "last checked" caption.
  - Round 4 shows "Draw round 4" and "Choose round 4 matchups", both disabled until R3 is decided.
  - Picking the open R3 winners enables both. "Choose round 4 matchups" opens the dialog with two 6-team groups.
  - A rematch shows the error inside the dialog, and the dialog stays open.
  - A legal draw closes the dialog, puts the pairs on the board with "Hypothetical draw — yours alone, not saved.", and changes the odds.
  - Reopening the dialog shows it prefilled.
  - Flipping an R3 pick removes the R4 draw.
  - "Play out the rest", then Knockout → "Choose quarterfinal matchups": a legal QF draw applies, and an illegal one shows an error.
  - Reset clears everything.
- [ ] **Step 5:** Commit `worlds_view.py app.py` with the message "Let users choose a round's matchups on the Worlds boards".

### Task 6: Docs

**Files:** `CLAUDE.md`, `worlds_swiss.toml` (comments only), and the `lolesports.py` module docstring.

- [ ] **Step 1: `CLAUDE.md`, "Running the event."** Replace "Facts are hand-entered ... no pipeline run needed" with:
  - Draws, results and the QF draw come from the lolesports.com schedule (`lolesports.live_facts`). The app reads it on a 5-minute cache and the pipeline reads it too.
  - The schedule is merged under `worlds_swiss.toml` by `swiss.resolve_live`: the config wins, and a schedule fact that fails validation is dropped with a page note.
  - The result priority is manual > Oracle's Elixir > lolesports.com > inferred.
  - The pipeline commits `{year}_worlds_schedule.json`, the app's fallback.
  - QF slots follow the schedule's start-time order, validated.

  Keep the `lolesports.py` CLI description (prints TOML to freeze a fact; `CONFLICT`, exit 1).
- [ ] **Step 2: `CLAUDE.md`, "What-ifs."** Add: hypothetical draws are random (`swiss.draw_round`) or chosen (`swiss.set_draw` / `set_quarterfinals`, validated by the same `_check_round` / `check_quarterfinals` as real draws), from the boards' "Choose … matchups" actions in an `st.dialog`.
- [ ] **Step 3: `worlds_swiss.toml` header and `[draws]` comment.** State that draws and results are read live from lolesports.com and that anything entered here overrides them. Drop "entered after Riot announces it".
- [ ] **Step 4: the `lolesports.py` docstring.** The CLI still never writes a file; `schedule()` / `save_snapshot()` are the pipeline's snapshot.
- [ ] **Step 5:** Commit the three files with the message "Document the live schedule and manual matchups".

### Task 7: End-to-end verification

- [ ] **Re-run every check script.** Expected: all `PASS`.
```bash
for c in check_live check_resolve_live check_app check_matchups check_board; do .venv/bin/python "$SP/$c.py" "$SP" || echo "FAIL $c"; done
```
- [ ] **Run `.venv/bin/python lolesports.py`.** It still prints TOML for anything the config lacks, and exits 0 when there is no conflict.
- [ ] **Repeat the Task 5 browser walkthrough once more on the final branch.**
- [ ] **Run `git status --short`.** Expected: clean. Then `git log --oneline origin/main..` should show 7 commits.
- [ ] **Report to the user.** Do not push. Include a post-merge check for them:
  - The deployed Worlds page should show the "last checked" caption.
  - If it instead says lolesports.com could not be reached, run `gh workflow run update.yml -f force=true` so the pipeline commits the snapshot the app falls back to.
