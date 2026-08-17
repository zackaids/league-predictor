# League of Legends Worlds 2026 Predictor

Predicting which team has the best chance of winning Worlds 2026, using
Oracle's Elixir competitive match data.

## Data

Source: Oracle's Elixir match data, served from a public Google Drive folder
(the same one behind https://oracleselixir.com/tools/downloads). One CSV per
year, 2014–2026. The website is Cloudflare-protected and 403s scrapers, so we
pull directly from Drive with `gdown`.

- Historical files (2014..last year) are frozen.
- The current-year file updates continuously as matches are played.
- Schema: ~165 columns. 12 rows per game = 5 players + 1 team aggregate, per side.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Fetching data

```bash
.venv/bin/python fetch_data.py                 # current year only (default)
.venv/bin/python fetch_data.py --all           # all years 2014-2026
.venv/bin/python fetch_data.py --years 2025 2026
.venv/bin/python fetch_data.py --discover       # verify Drive file ids vs hardcoded
.venv/bin/python fetch_data.py --all --force    # force re-download
```

Files land in `data/raw/`. Freshness is tracked in `data/raw/_fetch_meta.json`,
including a content hash so a rerun can tell a changed file from a re-download.
If a download breaks, `--discover` re-enumerates the folder and flags any file-id
drift to repair in `fetch_data.py`.

Raw CSVs (~800MB) are gitignored; re-fetch them with the commands above.

## Running it

```bash
.venv/bin/python pipeline.py --fetch     # fetch, re-rate, simulate, snapshot history
.venv/bin/streamlit run app.py           # power ranking, history, odds, model health
```

`pipeline.py` exits early when the raw data hasn't changed. A GitHub Actions cron
(`.github/workflows/update.yml`) runs it every 6 hours and commits the refreshed
tables; a Monday job additionally rebuilds the cross-year region prior.

## How the rating works

Elo is only valid within a *connected* pool of games, so a team's raw rating
carries its league's arbitrary drift — a 1700 in one league is not a 1700 in
another. `calibrate.py` splits each team into within-league skill and recombines
it with a region strength prior built from 13 years of international results:

```
calibrated = region_elo[region] + (team_elo - league_mean_elo)
```

This is load-bearing, not cosmetic. Measured point-in-time across 21 historical
MSI/Worlds, **raw Elo is worse than a coin flip** on cross-region games (0.735
log loss vs 0.693 for always-50%); the calibrated rating scores 0.644 and beats
raw Elo in 18 of 21 events.

## Validation

```bash
.venv/bin/python backtest.py             # in-season, calibration curves, cross-region holdout
```

Every constant in the pipeline was originally hand-picked and unmeasured.
`backtest.py` replays games chronologically and scores predictions that used only
prior information, against always-50% and side-base-rate baselines. Its findings
already changed the model: win probabilities were ~1.3x too spread out, which
compounded across simulated games and inflated the favourite's title odds.
