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
.venv/bin/python backtest.py --report           # write data/processed/backtest_report.json

# History
.venv/bin/python history.py                     # append today's snapshot
.venv/bin/python history.py --backfill          # reconstruct the whole season

# Frontend
.venv/bin/streamlit run app.py
```

Individual stages still run standalone (`clean.py`, `elo.py`, `calibrate.py`,
`monte_carlo.py`, `region_strength.py`, `features.py`), each taking `--year`.

## Data fetching

Oracle's Elixir's site is Cloudflare-protected and 403s scrapers, so
`fetch_data.py` pulls from the public Drive folder via `gdown`. Historical years
are frozen and cached; only the current year re-downloads by default.

```bash
.venv/bin/python fetch_data.py --discover        # verify Drive file ids vs hardcoded FILE_IDS
.venv/bin/python fetch_data.py --all             # all years (needed before --full)
```

If downloads start failing, run `--discover` first — it flags file-id drift
against `FILE_IDS`, which then needs manual repair. `_fetch_meta.json` records a
`sha256` per year; the scheduler uses it to skip recomputation, since the
current-year file is republished constantly but only sometimes actually differs.

Raw CSVs (~800MB) are gitignored. `data/processed/` and `data/history/` are
tracked — Actions commits them back, and Streamlit Cloud serves from them.

## Pipeline

```
fetch_data ─▶ clean ─▶ elo ─▶ calibrate ─▶ monte_carlo ─▶ history
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

**`monte_carlo.py`** builds a projected field from `worlds_field.toml` (slots and
already-qualified teams are assumptions, so they live in config) and simulates
Swiss + Bo5 knockout. Its RNG is seeded by default so scheduled reruns don't
jitter the odds when ratings haven't moved.

## Validation

`backtest.py` is the source of truth for whether a change helps. Three tests, all
walk-forward with no lookahead: in-season accuracy vs baselines, calibration
curves, and the point-in-time cross-region holdout. It must rebuild the region
prior from games strictly before each evaluated event — `region_strength.py`
normally builds it from all years at once, which would leak the event being
predicted into its own prediction.

Prefer justifying rating changes with a backtest number over reasoning alone;
every constant here was originally hand-picked and unverified.
