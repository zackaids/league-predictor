# Worlds 2026: Swiss & knockout odds — design

_Approved design (spec). Written 2026-10-11._


## Context

Worlds is live. The 16-team Swiss field is set except one slot, the Play-In winner
(FURIA / MVK / Cloud9 Kia / KC; Play-In runs Oct 15–18). Today:

- `monte_carlo.py` still simulates a **projected** field from `worlds_field.toml`, and
  that field is now wrong: it includes NIP, GAM and KC, and omits TES, LØS and the
  Play-In winner. Its format is generic: Round 1 drawn at random, greedy pairing, and
  a knockout seeded by rating. These wrong odds go into `title_odds_history` on every
  data-changed run.
- There is no Swiss view, and the Play-In page's native-widget layout looks weak.

Riot's primer publishes the actual draw rules (below), so the Swiss and quarterfinal
draws can be modeled faithfully.

**Course of action.** Fill the 16th slot probabilistically. Each simulation samples
the Play-In winner from the Play-In page's exact odds, including what-if picks. That
gives Swiss and title odds now for all 19 teams, and they sharpen as results land.
Replace the projected simulation with a vectorized engine for the real 2026 format.
Present Play-In → Swiss → Knockout as one interactive "Worlds" page built on a
custom HTML component.

**Decided with you:** full play-through what-ifs · custom component on
`st.components.v2` (available in your Streamlit 1.61.1) · a single "Worlds" hub page.

## Format being modeled

Sources: [lolesports primer](https://lolesports.com/news/worlds-2026-primer); Bo1/Bo3 from
[Liquipedia](https://liquipedia.net/leagueoflegends/World_Championship/2026/Swiss_Stage) and
[Hotspawn](https://www.hotspawn.com/league-of-legends/guide/lol-worlds-format); the
[2025 knockout draw](https://idcgames.com/en/league-of-legends/news/the-knockout-stage-draw-show-sets-the-stage-for-worlds-2025’s-final-showdown-2025-10-27-11-20-11688).

- **Swiss.** 16 teams over 5 rounds. Three wins advance, three losses eliminate.
- **Round 1** (drawn Oct 18). Tier 1 v Tier 4 and Tier 2 v Tier 3, with no same-league
  pair.
  - T1: GEN, AL, TL, G2
  - T2: LØS, TSW, HLE, BLG
  - T3: LYON, MKOI, T1, TES
  - T4: CFO, IG, DK, Play-In winner
- **Rounds 2–5** (drawn Oct 23/24/26/30). Random among teams on the same record. Never a
  rematch. Same-league pairs allowed.
- **Bo3 vs Bo1.** A match is Bo3 (Fearless) when it can advance or eliminate, i.e. the
  record has 2 wins or 2 losses. Every other match is Bo1.
- **Quarterfinal draw** (Oct 31). Each 3-0 team is drawn against one of the three 3-2 teams.
  The remaining four (three 3-1 teams and one 3-2) are drawn into two matches. All
  knockout series are Bo5 (QF Nov 3–6, SF Nov 7–8, Final Nov 14).
- **Assumptions.**
  - The two 3-0 teams sit on opposite halves, and the other two QFs land in random
    halves. This follows the 2025 precedent; the 2026 primer is silent. No region
    restriction applies.
  - A draw is modeled as uniform over legal pairings. Riot draws sequentially, and the
    difference is negligible.

## Architecture

```
worlds_bracket.toml ─▶ bracket.py ─(exact Play-In odds)─┐
worlds_swiss.toml ──▶ swiss.py ─(field, known draws/results, what-ifs)─▶ monte_carlo.py (engine)
series log (WLDs) ──┘
pipeline ─▶ {year}_swiss_field.csv, {year}_worlds_title_odds.csv (→ history)
app "Worlds" hub ─▶ worlds_view.py (board data) ─▶ board.py + assets/board.{js,css}
```

Module roles have no import cycles: `swiss` imports `bracket` and `monte_carlo`, and
`monte_carlo` never imports `swiss`.

| Module | Role |
|---|---|
| `monte_carlo.py` (rewritten) | Engine: format rules, vectorized simulation, draw samplers. Also keeps the projected-field path for seasons before the field is confirmed. |
| `swiss.py` (new, mirrors `bracket.py`) | The confirmed event: config, field, results from the series log, what-if state, outputs and CLI. |
| `worlds_view.py` (new, pure) | Turns facts and odds into board data per stage; checkable with asserts. |
| `board.py` + `assets/board.js`, `assets/board.css` (new) | The `st.components.v2` renderer. |

## Phase 1: engine and correct title odds (target: before Oct 15)

### `worlds_swiss.toml` (new)

Facts, like `worlds_bracket.toml`.

- **Dates and place.** Swiss window `start`/`end` (Oct 23–31), `knockout_start`/
  `knockout_end` (Nov 3–14), and `location`.
- **Teams.** `teams = [{ name, short, tier, data_name? }]` for the 15 direct teams, plus
  `playin_tier = 4`. The league for the Round 1 rule comes from `calibrated_ratings`,
  not hand entry.
- **Draws.** `[round.N]` holds each draw as Riot announces it, keyed by the record going
  in, e.g. `"1-0" = [["A","B"], …]`.
- **Quarterfinal draw.** `[knockout] quarterfinals = [[..],[..],[..],[..]]` in bracket
  order: QF1 and QF2 feed SF1, QF3 and QF4 feed SF2.
- **Manual results.** `[results]` overrides: Swiss matches keyed `"A v B"` (no pair meets
  twice in Swiss), knockout keyed `QF1` … `F`.
- **Play-In tricodes.** Add `short` (FUR, MVK, C9, KC) to the teams in
  `worlds_bracket.toml` and to `playin_field.csv`.

### `monte_carlo.py`

Rewrite the simulation. Keep `load_field_config`, `build_field` and `p_game` for the
projection, since comments elsewhere name `p_game`.

**Inputs.**
- `Field`: teams, rating, league, short, tiers (4×4 with a `PLAYIN` placeholder), and
  `playin` = {team: P(wins Play-In)}.
- `Known`: Swiss draws for rounds 1..k (pairs grouped by record), Swiss results, the QF
  draw, and knockout results. It is always a prefix: every round before k is fully
  decided, which `swiss.py` enforces.

**`simulate(field, known, n=20000, seed=0) -> Odds`**, vectorized over simulations with
numpy:
- **Series odds.** Build T×T matrices once: Bo1 from `calibrate.prob_from_gap`, Bo3 and
  Bo5 from `bracket.p_series`. Each series is then one random draw, not game-by-game.
- **Play-In slot.** Sample the winner per simulation from `field.playin`.
- **Round 1, if not drawn.** Precompute the legal tier1×tier4 permutations for each
  possible Play-In winner (the league rule), and the legal tier2×tier3 permutations.
  Pick one uniformly.
- **Rounds 2–5, if not drawn.**
  - Sort slots by (record, random). Group sizes are fixed by the format: 8/8, then
    4/8/4, then 6/6, then 6.
  - For each group, enumerate its perfect matchings (3, 15 or 105), mask any that
    contain a rematch, and pick uniformly among the rest.
  - If no legal matching exists, allow a rematch and count it. The count should be about 0.
- **Match format.** Bo3 exactly when the record has 2 wins or 2 losses.
- **Knockout.** Draw the quarterfinals per the rules above unless already drawn, then
  play QF, SF and Final as Bo5.
- **Known draws and results** override sampling.
- **Common random numbers.** Every round uses fixed-shape random blocks whether or not
  they are needed. A what-if pick then moves only what it actually affects; it does not
  reshuffle unrelated simulations.

**What `Odds` returns:**
- the final-record distribution (3-0 … 0-3)
- P(knockout), P(semis), P(final), P(title)
- P(team holds record r going into each round), for boxes not yet drawn
- P(A meets B) in the next undrawn round, which gives the Round 1 opponent odds
- P(team lands in each QF slot)

**Other functions.**
- `sample_round(field, known, rng)` and `sample_quarterfinals(field, known, rng)` make one
  legal draw for "Draw next round". They share the rule code with `simulate`.
- `match_prob(field, a, b, best_of)` gives the exact odds shown on each match.
- **Projected path** (`simulate_and_write`): same inputs as today (`worlds_field.toml`).
  Pseudo-tiers come from calibrated rank, with no Play-In. The league rule is dropped for
  a tier pairing only if it can't be satisfied.
- **Delete** `sim_series`, `swiss_pairings`, `sim_swiss`, `sim_knockout` and `run`.
  `pipeline.py` is the only caller of this module.
- **Shared odds-table writer.** Columns: `team, home_league, calibrated, swiss_%,
  knockout_%, semis_%, finals_%, title_%, 3-0_% … 0-3_%, field`. The `field` column is
  `projected` or `confirmed`, which marks the model change in `title_odds_history` without
  bumping `PIPELINE_VERSION` (that would wrongly flag rating history). Existing column
  names stay.

### `swiss.py` (new)

- **`load_config()` with validation.**
  - The field is 15 teams plus the Play-In slot, 4 per tier, all names known, record
    labels well-formed.
  - Draws are legal: tiers and the league rule in Round 1; same record, no rematch, and
    each active team exactly once in later rounds.
- **`is_current(year)`:** true when `cfg["start"].year == year`.
- **`build_field` / `write_field`:** write `{year}_swiss_field.csv` (team, data_name,
  short, tier, home_league, rating). The app reads this CSV, so an upstream rename can't
  break the page; this is the same reason `playin_field.csv` exists.
- **`resolve_results(cfg, series, playin_winner)`:**
  - Matches WLDs series inside the Swiss window to drawn pairs.
  - A series counts as finished when the leader has 1 win (Bo1) or 2 (Bo3), going by the
    pair's record group. A half-published Bo3 at 1-0 is therefore not read as final.
  - Knockout series are matched per slot and need 3 wins.
  - Manual `[results]` win over anything detected.
  - Extract the "earliest finished, unused series between a pair" loop from
    `bracket.resolve_results` into `bracket.find_series`, with no behavior change, and
    reuse it here.
- **Inference.** A later round's record groups imply the earlier round's winners. Any
  result Oracle's Elixir hasn't published yet is filled from the draw, with
  source = "draw".
- **Warnings.** Flag any WLDs series in the window between two Swiss teams that matches
  no drawn pair, e.g. "add round N's draw to worlds_swiss.toml".
- **`known_state(facts, whatif)` → (`Known`, pruned what-ifs, notes).** Pruning rules:
  - A real result or draw beats a what-if. A real draw replaces a hypothetical one.
  - Changing any result in round r drops the hypothetical draws for rounds after r, and
    the picks on them.
  - Play-In picks are pruned with `bracket.determined`.
  - A change of Play-In winner drops a hypothetical Round 1 draw.
  - A hypothetical QF draw and its knockout picks are dropped if the records no longer
    match.
- **`play_out(facts, whatif, field, rng)`:** fills every open decision once by
  model-weighted sampling: Play-In picks, draws, Swiss picks, the QF draw and knockout
  picks. This drives "Play out the rest".
- **`write_outputs(year, n, seed)` and CLI** (`python swiss.py`): write the field CSV and
  the title odds CSV, and print the table.

### `pipeline.py`, `history.py`, `requirements.txt`

- **Pipeline order.** After calibrate:
  1. Play-In field, with its existing guard.
  2. `swiss.write_outputs` when `swiss.is_current(year)`; otherwise
     `monte_carlo.simulate_and_write` (projected).
- **Guard on the confirmed path.** It gets the same guard as the Play-In stage. On error
  it reports loudly, keeps going, and passes `history.record(..., odds=False)` (new
  argument) so stale odds aren't re-stamped.
- **requirements.txt.** Add `numpy` (now imported directly) and require `streamlit>=1.61`
  (needed for components v2).

## Phase 2: board component and Worlds hub (target: Play-In page redesigned before Oct 15)

### 1. Throwaway spike in the scratchpad

Build a two-card component and confirm:
- a click calls `setTriggerValue`, the Python callback runs, and the app reruns once
  with the new pick;
- the `--st-*` theme variables work in light and dark mode;
- `height="content"` sizes the component correctly;
- the board scrolls horizontally at phone width.

If the callback can't read the trigger cleanly, read the result after mounting and call
`st.rerun()`. If v2 fails outright, stop and ask before falling back to static HTML with
native controls.

### 2. `board.py` + `assets/board.{js,css}`

One generic renderer. JS and CSS are read from files at import, because a component
that isn't an installed package needs inline strings.

- **Data contract:** `columns → boxes → matches → rows`.
  - A row carries team, short code, league, probability and state
    (open / picked / won / lost).
  - A box can carry `likely` chips (for slots not yet drawn) and an `action` button
    (e.g. "Draw round 2").
  - Optional `connectors` for single-elimination brackets.
- **Events:**
  - `pick {match, team}`
  - `action {id}`
  - `team {name}`: a small ↗ that opens the Team page, replacing today's team buttons
- **Visual requirements:**
  - Each row shows tricode, full name, a league marker (color plus a text label, never
    color alone), an odds bar with %, and a Bo1/Bo3/Bo5 tag.
  - Winners get ✓, losers are dimmed, picks show →. Real results look visibly different
    from what-ifs.
  - The knockout gets CSS-drawn connectors.
  - Rows are keyboard-focusable buttons.
  - The DOM is built with `textContent`, never `innerHTML` of data.
  - Colors come from `--st-*` variables with fallbacks. Load the `dataviz` skill before
    choosing league colors.

### 3. Worlds hub page (`app.py`)

Replaces "Worlds bracket" and "Worlds odds" in `PAGES`.

- **Stage switcher.** Play-In | Swiss | Knockout | Title odds. It opens on the live
  stage, judged from results rather than the date.
- **What-if state.** One dict in `st.session_state` under plain keys, not widget keys, so
  it survives switching stages and pages. It holds Play-In picks, hypothetical Swiss
  draws, Swiss picks, the QF draw and knockout picks. A bar shows "N what-ifs applied",
  **Play out the rest** and **Reset**.
- **Odds.** `simulate` is called through `st.cache_data`, keyed on a hashable snapshot of
  the field and known state. It uses the pipeline's n and seed, so the no-what-if
  baseline equals the committed CSV.
- **Play-In stage.** Today's logic (`bracket.resolve_results`, `determined`, `solve`)
  rendered on the board. Keep the "Path to Swiss" table.
- **Title odds stage.**
  - This replaces `page_odds`, now on the confirmed field with live what-ifs.
  - Adds a Δ-vs-baseline column when what-ifs are active.
  - Has a selector for Swiss / Knockout / Semis / Final / Title.
- **Title strip.** A one-line "title odds with your what-ifs" strip under every stage.
- **Next season.** When the configs aren't for `YEAR`, the hub shows only the projected
  title odds CSV, as today.
- **Remove** `page_bracket`, `bracket_card`, `bracket_row`, `page_odds` and their
  constants. Reuse `open_team`, `load_series`, `load_ratings` and `data_age_banner`.

## Phase 3: Swiss and knockout stages, play-through (target: before the Oct 18 Round 1 draw; live by Oct 23 at the latest)

The board builders live in `worlds_view.py`.

**Swiss board.**
- Columns: Round 1–5, then Final records.
- Each round has boxes by record, e.g. Round 3 has 2-0 / 1-1 / 0-2.
- Drawn matches show odds, picks and results.
- Undrawn boxes show the likely teams (from the occupancy odds). A "Draw round N" action
  appears once the previous round is decided. Until then it is disabled with a hint, e.g.
  "decide the Play-In first".
- Before the Round 1 draw, Round 1 shows Tier 1 v Tier 4 and Tier 2 v Tier 3, with each
  team's opponent odds.
- The Final column splits into 3-0 / 3-1 / 3-2 (advance) and 2-3 / 1-3 / 0-3 (out).

**Knockout board.**
- QF ×4 → SF ×2 → Final, with connectors.
- Before the draw: slot-occupancy chips and a "Draw quarterfinals" action.

**Team table** under the Swiss board: tier, league, rating, P(3-0 … 0-3), P(advance) and
P(title). It uses progress columns, as on today's Play-In table.

## Limits shown as captions in the app

- The 3-0 bracket-halves rule is an assumption; a real QF draw in the config overrides it.
- Draws are modeled as uniform over legal pairings.
- Games are independent; draft, side selection and travel are not modeled.
- Ratings are the current calibrated Elo with `RATING_SCALE`; rating logic is untouched.

## Verification

This follows the repo convention: no committed test suite. Checks are assert-based
scripts in the session scratchpad, each written before its code and seen to fail first,
plus `streamlit.testing.v1.AppTest`.

**Engine invariants** (n = 20k, fixed seed):
- Every simulation ends with exactly 2/3/3/3/3/2 teams at 3-0/3-1/3-2/2-3/1-3/0-3.
- No rematches (fallback count about 0).
- Round 1 is always tier1 v tier4 and tier2 v tier3, with no same-league pair.
- Bo3 is used exactly when a team has 2 wins or 2 losses.
- Both 3-0 teams face 3-2 teams, on opposite halves.
- Title odds sum to 100%.
- P(title) ≤ P(final) ≤ P(semis) ≤ P(knockout) ≤ P(swiss).

**Engine sanity and accuracy:**
- With equal ratings, each team advances 50% ±1, goes 3-0 12.5% ±1, and wins the title
  6.25% ±0.5.
- In a late state (Round 5 and knockout left, all draws fixed), the simulation matches a
  brute-force enumeration within Monte Carlo error.
- Known draws and results are respected.
- The same seed gives identical output.
- 20k simulations run in under 0.5 s locally.

**`swiss.py`:**
- The config loads, and the field's ratings equal `calibrated_ratings`.
- A synthetic series log produces the right results, and a Bo3 at 1-0 is not treated as
  final.
- A result is inferred from a later draw.
- The unmatched-series warning fires.
- What-if pruning holds:
  - changing a Round 1 pick after drawing Round 2;
  - a real draw replacing a hypothetical one;
  - a change of Play-In winner.

**Pipeline:**
- `.venv/bin/python pipeline.py` completes.
- The odds CSV has 19 teams with `field=confirmed`, and history appends.
- `backtest.py --quick` output is unchanged.

**App:**
- AppTest renders each stage without exceptions.
- The no-what-if baseline equals the committed CSV.
- A browser check (run skill or Chrome) covers light and dark mode, click-to-pick,
  Draw round, Play out the rest, Reset, and a ~390px width.

## Rollout and operations

- **Branching.** Run `git fetch` and `git pull --ff-only` on `main` first (Actions commits to
  `origin/main` every 6h). Branch `worlds-swiss` off the up-to-date `main`. The Play-In and
  CBLOL work is already merged there (PR #3, `f21fe8e`).
- **Commits.** Commit code, configs and the small new or changed field CSVs only; let
  Actions regenerate everything else. After merging, run the workflow manually with
  `force` so odds and history regenerate under the new model.
- **During the event** (document this in CLAUDE.md):
  - After each draw, add `[round.N]`, later `[knockout]`, to `worlds_swiss.toml` and push.
    The app reads the config live.
  - Results come from Oracle's Elixir or are inferred from the next draw. `[results]`
    covers any gaps.
- **Docs.** Update CLAUDE.md (pipeline diagram, the `monte_carlo.py` paragraph, new
  modules and configs, the component) and the README.

## Out of scope

- A title-odds-over-time chart from `title_odds_history`.
- Modeling Fearless draft, side selection or roster changes.
- Exact Swiss odds. The draw × outcome space is far too large; Monte Carlo noise at 20k
  simulations is ≤ ±0.35 pt (1 s.e.).

## After approval

Following your existing superpowers flow:
1. Save this as `docs/superpowers/specs/2026-10-11-worlds-swiss-design.md`.
2. Write the task-by-task plan with writing-plans, as
   `docs/superpowers/plans/2026-10-11-worlds-swiss.md`.
3. Execute phase by phase. Phase 1 ships independently, because it fixes the odds history.
