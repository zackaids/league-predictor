# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Predicts each team's odds of winning Worlds from Oracle's Elixir competitive LoL
match data (2014-present). A sequential data-science pipeline of standalone
scripts, orchestrated by `pipeline.py`, surfaced by a Streamlit app, and kept
current by a GitHub Actions cron.

There is no unit-test suite and no linter config. The correctness check that
matters here is `backtest.py` — see Validation below.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Common commands

```bash
# Full pipeline (this is normally what you want)
.venv/bin/python pipeline.py                    # fast path, current year, no fetch
.venv/bin/python pipeline.py --fetch            # fetch first; exits early if data unchanged
.venv/bin/python pipeline.py --fetch --full     # also rebuild region prior + backtest report
.venv/bin/python pipeline.py --year 2025 --full

# Validation
.venv/bin/python backtest.py                    # all three tests
.venv/bin/python backtest.py --quick            # skip the slow cross-region test
.venv/bin/python backtest.py --sweep-k          # K-factor sweep
.venv/bin/python backtest.py --playin           # does Play-In form predict the main stage? (reads every raw CSV)
.venv/bin/python backtest.py --report           # write data/processed/backtest_report.json

# History
.venv/bin/python history.py                     # append today's snapshot
.venv/bin/python history.py --backfill          # reconstruct the whole season

# Worlds odds (the pipeline runs this; standalone for debugging)
.venv/bin/python swiss.py                       # Swiss + knockout odds from worlds_swiss.toml
.venv/bin/python bracket.py                     # Play-In field + odds
.venv/bin/python lolesports.py                  # after a draw: prints the TOML to paste (read-only)

# Swiss Pick'Em (which teams to pick, from simulated Swiss runs)
.venv/bin/python pickem.py                      # picks + hold-out odds -> data/processed/{year}_pickem.csv
.venv/bin/python pickem.py --n 100000 --k 6     # fewer sims; only optimize P(>= 6 right)
.venv/bin/python pickem.py --no-playin          # picks if the Play-In results counted for nothing
.venv/bin/python pickem.py --rating "Karmine Corp=1708"   # pin any rating (name or tricode)

# Crystal Ball (the Worlds Pick'Em bonus questions)
.venv/bin/python crystal_ball.py                # answers + expected points -> data/processed/{year}_crystal_ball.csv
.venv/bin/python crystal_ball.py --backtest 2025   # replay a past Worlds; --no-playin, --n, --reps

# Frontend
.venv/bin/streamlit run app.py
```

Individual stages still run standalone (`clean.py`, `elo.py`, `calibrate.py`,
`monte_carlo.py` (projected field), `region_strength.py`, `features.py`), each taking `--year`.

## Data fetching

Oracle's Elixir's site is Cloudflare-protected and 403s scrapers, so
`fetch_data.py` pulls from the public Drive folder via `gdown`. Historical years
are frozen and cached; only the current year re-downloads by default.

```bash
.venv/bin/python fetch_data.py --discover        # verify Drive file ids vs hardcoded FILE_IDS
.venv/bin/python fetch_data.py --all             # all years (needed before --full)
```

If downloads start failing, run `--discover` first — it flags file-id drift
against `FILE_IDS`, which then needs manual repair. Drive also periodically
throttles the file outright (*"too many users have viewed or downloaded this
file recently"*) — that one is upstream and self-clearing, so `fetch_data`
retries, then raises `TransientFetchError`, and `pipeline.py` skips the run
(exit 0, `fetch_failed=true`) rather than failing the scheduled job. Downloads
land in a `.part` sidecar and are header-checked before replacing the cached
CSV, because a throttled Drive returns *200 OK* with an HTML error page that
would otherwise be hashed and committed as real content.

`_fetch_meta.json` records a `sha256` per year; the scheduler uses it to skip
recomputation, since the current-year file is republished constantly but only
sometimes actually differs.

Raw CSVs (~800MB) are gitignored. `data/processed/` and `data/history/` are
tracked — Actions commits them back, and Streamlit Cloud serves from them.

## Pipeline

```
fetch_data ─▶ clean ─▶ elo ─▶ calibrate ─▶ odds (swiss.py | monte_carlo.py) ─▶ history
                        ▲
        region_strength ┘   (cross-year, slow, --full only)
```

`features.py` (→ `{year}_team_profiles.parquet`) is an independent
recency/roster-weighted profile table. **Nothing consumes it.** Before wiring it
into the model, note that LPL has ~0% timeline coverage (`gd10/gd15/xpd15/csd15`
all NaN for the largest league), and `gspd`/`tower_diff`/`baron_diff` are
consequences of winning, so they leak.

## Architecture

**Grain.** `clean.py` keeps only `position == "team"` rows, giving one row per
team per game. Downstream stages consume that, *except* `region_strength.py` and
`calibrate.py`'s LCP resolution, which re-read raw yearly CSVs for cross-year and
per-player-roster data.

**Why calibration is separate from Elo.** `elo.py` is a single chronological Elo
pass, valid only within one *connected* pool. `MAJOR_SCOPE` restricts it to major
regions plus international events, so sealed minor pools don't farm Elo
internally. Its international half is derived from
`region_strength.INTL_LEAGUES` — do not maintain a second copy; MSI and Worlds
were once missing from a hand-written list and were silently dropped.

Only the six Worlds leagues in `calibrate.TOP_LEAGUE_REGION` (LCK, LPL, LEC, LCS,
LCP, CBLOL) get a calibrated rating and appear in the ranking; `elo.MAJOR_LEAGUES`
is the wider pool that links them onto one Elo scale. Add a league to the former
only if its teams can reach Worlds directly.

Raw Elo still carries its league's arbitrary drift, so `calibrate.py` splits each
team into `within_league` (offset from its league mean) and recombines it with an
independently computed `region_elo`:

```
calibrated = region_elo[region] + within_league
```

This is not a nicety. Measured point-in-time on 21 historical MSI/Worlds,
**raw Elo is worse than a coin flip** on cross-region games (0.735 log loss vs
0.693); calibrated scores 0.644 and wins 18/21 events.

**`RATING_SCALE`** (`calibrate.py`) shrinks calibrated gaps before converting to
a win probability. `calibrated` sums two independently-scaled Elo quantities, and
the sum is ~1.3× too wide — underdogs win materially more than it implies. Fitted
by `backtest.fit_shrinkage`; re-fit whenever rating logic changes.

**Guards worth keeping.** `home_league` requires a top-league appearance to match
the team's modal league across *all* domestic play (ERL rosters were being ranked
as LEC teams), so it needs the **unscoped** team-games table. Teams under
`MIN_GAMES` are flagged `low_sample` and excluded from league means, since the
mean is the baseline every other team is measured against.
`region_strength.run()` refuses to rebuild from fewer than 5 years, because a
fresh checkout has only the current year and would otherwise overwrite a
13-year prior with a worthless one.

**LCP** merges the old PCS/VCS/LJL/LCO regions, so `calibrate.py` re-anchors each
LCP team to its real constituent region from 2023-2025 history; unresolvable orgs
get a composite (`APAC*`).

**Worlds odds.** `monte_carlo.py` is a numpy engine for the real 2026 format:
tiered Round 1 with no same-league pair, same-record draws with no rematches in
rounds 2-5, Bo3 iff a team can advance or be eliminated, the quarterfinal draw
(each 3-0 v a 3-2, 3-0s on opposite halves -- the 2025 precedent, not stated for
2026), Bo5 knockout. A `Known` state pins what has really happened (or a
what-if) and everything else is sampled, so the same call serves the daily odds,
the live app and "play out the rest". The 16th Swiss team is the Play-In winner,
sampled from the Play-In page's exact odds until it is decided. Draws are uniform
over legal pairings (rejection sampling), games are independent, and the RNG is
seeded so scheduled reruns don't jitter the history. `swiss.py` is the confirmed
event on top of it (config, results, what-if layering); without
`worlds_swiss.toml` for the season, `monte_carlo.simulate_and_write` falls back to
a projected field from `worlds_field.toml`.

**Running the event** (`worlds_swiss.toml`). Facts are hand-entered and the app
reads the config live, so after Riot announces a draw add it and push -- no
pipeline run needed. Round 1 is drawn Oct 18; rounds 2-5 on Oct 23, 24, 26, 30;
the quarterfinals Oct 31. Draws are keyed by the record going in
(`[draws.2]` / `"1-0" = [["A", "B"], ...]`) and are validated against the format.
Results come from the series log (`WLDs`, inside the stage's dates); a Bo3 at 1-0
is not a result. Results Oracle's Elixir has not published yet are inferred from
the next round's draw or the quarterfinal draw, and `[results.swiss]` /
`[results.knockout]` override both. A series between two Swiss teams that no draw
explains is reported in the page notes, not dropped.
League code `WLDs` also holds September LPL qualifier series, which is why every window is a date range.
`lolesports.py` reads the schedule behind lolesports.com (public API, fresher than
OE for results), compares it with both configs and the series log, and prints the
TOML for any new Play-In result, Swiss draw, result or quarterfinal draw, plus any
`CONFLICT` with what is already entered (exit 1). It never writes the configs, and
the quarterfinal pairs come back in start-time order, so confirm QF1-QF4 against
the official bracket.

**What-ifs.** The Worlds page keeps picks, hypothetical draws and a hypothetical
quarterfinal draw in session state (`swiss.new_whatif`). `swiss.state` layers them
over the real facts: a real result or draw always wins, and a pick or draw whose
premise changed (an earlier result, the Play-In winner) is dropped along with
everything built on it. The bracket UI is `board.py` + `assets/board.{js,css}` on
`st.components.v2`; `worlds_view.py` builds its data without importing Streamlit.
The component reads `assets/` at import, so edit them with the app restarted.

**Pipeline guard.** The odds stage is guarded like the Play-In one: on failure
the previous odds file stays and `history.record(odds=False)` skips that run's
odds row, because stamping a stale file with today's date would present old odds
as fresh.

**Series log.** `elo.run()` also writes `{year}_series.parquet`: one row per
series with both teams' Elo before/after and the net change, which the app's
match feed and Team page read. It comes from the rating loop itself
(`run_elo(game_log=...)`), not a second replay, so it cannot drift from the
ratings. It must be committed: `team_games` is gitignored, so Streamlit Cloud
cannot rebuild it. Series are split when a pair's `game` number stops
increasing, or after a 12h gap when the number is missing. The feed shows raw
Elo deltas; for league games that equals the change in `calibrated`, for
internationals it is ~10% larger.

**Swiss Pick'Em.** `pickem.py` picks the teams for the Swiss bracket (counts and points in `pickem.toml`).
`monte_carlo.simulate(keep_runs=True)` keeps every run's final record, so picks are scored on whole simulated
stages. Most expected points depends only on per-team marginals and is solved exactly (DP over teams); best
chance of >= k points depends on the joint outcomes (two favorites who meet at 1-0 cannot both go 3-0) and is a
local search. The search runs on one set of sims and every reported number comes from an independent hold-out
(seed + 1), because a rare 8/8 hit is a few dozen runs and over-fits easily. The app's Pick'Em stage reuses
`swiss.state`, so what-ifs re-optimize the picks. A "Rating what-if" expander (and `--rating` / `--no-playin` on
the CLI) re-runs it with some ratings changed: the Play-In winner's rating includes ~11 games of Play-In Elo, and
`backtest.py --playin` (8 Worlds, 245 main-stage games) found that counting those games at 0x-0.5x predicts the
main stage better than the shipped 1x, though the gap is inside series-level noise; 1.5x is clearly worse. The
Play-In windows in that test are hand-read from daily game counts (`PLAYIN_WINDOWS`).

**Crystal Ball.** `crystal_ball.py` answers the 18 Pick'Em bonus questions (points per question in
`crystal_ball.toml`, so the best pick is the likeliest answer). The tournament is the existing engine --
Play-In walked with `bracket`, Swiss + knockout from `monte_carlo.run_trace` -- with each series expanded
into games and each game's stats bootstrapped from the teams' own last ~135 days, *same result*, matched to
the matchup's lopsidedness (a plain Elo's pre-game probability) and with Fearless locks inside a series.
Presence is the exception: it uses a regression on the newest patch's presence (all leagues), because the
pool simulation is overconfident about a meta that resets every Worlds (the newest patch counts x8). Most deaths
per game is a ranking of high-volume champions (the real leader is an engage support, not a lucky 5-game
champion), and highest win rate is a flat lottery: no signal beat a uniform guess in backtests. The shortest-game winner is each
team's share of the event's wins (the pool model gave the real winner 0% in 2022-25). Teemo and the Finals
MVP role are base rates; 15-kill games use the observed Worlds rate. Every calibration constant in the
toml cites the backtest that set it -- refit with `--backtest` rather than editing by hand. Two data traps:
league `WLDs` also holds the LCK/LPL regional qualifiers (use `worlds_window`), and the Play-In counts
toward the stats by default because the rules are silent (`--no-playin` to check what flips).

## Validation

`backtest.py` is the source of truth for whether a change helps. Three tests, all
walk-forward with no lookahead: in-season accuracy vs baselines, calibration
curves, and the point-in-time cross-region holdout. It must rebuild the region
prior from games strictly before each evaluated event — `region_strength.py`
normally builds it from all years at once, which would leak the event being
predicted into its own prediction.

Prefer justifying rating changes with a backtest number over reasoning alone;
every constant here was originally hand-picked and unverified.
