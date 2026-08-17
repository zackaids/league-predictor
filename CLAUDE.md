# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Predicts each team's odds of winning Worlds 2026 from Oracle's Elixir competitive
LoL match data (2014-2026). No tests, no build step, no linter configured — this
is a small sequential data-science pipeline of standalone scripts.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data fetching

Source is Oracle's Elixir's public Google Drive folder (the website itself is
Cloudflare-protected and 403s scrapers, so `fetch_data.py` pulls from Drive via
`gdown` instead). Historical years are frozen and cached; only the current year
re-downloads by default.

```bash
.venv/bin/python fetch_data.py                 # current year only (default)
.venv/bin/python fetch_data.py --all            # all years 2014-2026
.venv/bin/python fetch_data.py --years 2025 2026
.venv/bin/python fetch_data.py --discover       # verify Drive file ids vs hardcoded FILE_IDS
.venv/bin/python fetch_data.py --all --force    # force re-download
```

If downloads start failing, run `--discover` first — it re-enumerates the Drive
folder and flags any file-id drift against the hardcoded `FILE_IDS` map in
`fetch_data.py`, which then needs manual repair.

Raw CSVs (~850MB) are gitignored; freshness is tracked in `data/raw/_fetch_meta.json`.

## Pipeline

Each stage is a standalone script, run in this order, each reading the previous
stage's output from `data/processed/`:

```bash
.venv/bin/python clean.py             # raw 2026 CSV -> data/processed/2026_team_games.parquet
.venv/bin/python elo.py               # team_games -> data/processed/2026_team_elo.parquet
.venv/bin/python region_strength.py   # raw 2014-2026 CSVs -> data/processed/region_strength.csv
.venv/bin/python calibrate.py         # team_elo + team_games + region_strength -> 2026_calibrated_ratings.parquet
.venv/bin/python monte_carlo.py       # calibrated_ratings -> data/processed/2026_worlds_title_odds.csv
```

`features.py` (-> `2026_team_profiles.parquet`) is a separate, independent
analysis of recency/roster-weighted team strength profiles. It is **not**
consumed by `elo.py`/`calibrate.py`/`monte_carlo.py` — those run on raw Elo,
not on `features.py`'s weighted profiles.

## Architecture

**Grain**: `clean.py` collapses the raw per-player rows down to one row per
team per game (`position == "team"`), since the target is team outcomes, not
player stats. All downstream stages consume this team-per-game table, not raw
data — except `region_strength.py` and `calibrate.py`'s LCP resolution, which
re-read the raw yearly CSVs directly because they need cross-year or
per-player-roster data that `clean.py`'s single-year output doesn't carry.

**Why calibration is a separate step from Elo**: `elo.py` runs a single
chronological Elo pass, which is only valid within one *connected* pool of
games. Cross-region comparisons work at all only because international events
(First Stand, MSI, EWC, ...) put different leagues' teams in the same games,
chaining the regional pools onto one scale — but a team's raw Elo still
carries its home league's arbitrary drift (a 1700 in LCP isn't a 1700 in LCK).
`elo.py` restricts to `MAJOR_SCOPE` (major regions + linking international
events) so minor sealed pools don't farm Elo internally and float unrealistically
high.

`calibrate.py` fixes the cross-region comparability problem by splitting each
team's Elo into `within_league` skill (offset from its league's mean) and
recombining it with an independently-computed `region_elo` (from
`region_strength.py`, itself a *separate* Elo run over 13 years of
cross-region international results only). Both are in Elo points, so they add
directly: `calibrated = region_elo[region] + within_league`. This lets a
mid-table LCK team correctly outrank the best team in a weaker region.

LCP (the merged APAC league, 2025+) is a special case in `calibrate.py`: since
it merges what used to be separate PCS/VCS/LJL/LCO regions, each LCP team is
re-anchored to its real historical constituent region (via 2023-2025 domestic
league history in `lcp_team_regions`) rather than treated as one uniform
"APAC" region. Teams with no resolvable history fall back to a composite of
the four constituent regions' Elo (`_LCP_COMPOSITE` / displayed as `APAC*`).

**`monte_carlo.py`** builds a projected 16-team Worlds field from the top
`calibrated` teams per qualifying league (`DEFAULT_SLOTS`), simulates the
actual Worlds format game-by-game from `calibrated` Elo win probabilities
(Swiss stage with Bo1/Bo3 depending on record, then Bo5 single-elim
knockout), and outputs title/finals/knockout odds per team. Field
composition (slot counts, guaranteed teams) is hardcoded and needs updating
as the real Worlds qualification picture becomes known.
