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

Files land in `data/raw/`. Freshness is tracked in `data/raw/_fetch_meta.json`.
If a download breaks, `--discover` re-enumerates the folder and flags any file-id
drift to repair in `fetch_data.py`.

Raw CSVs (~850MB) are gitignored; re-fetch them with the commands above.
