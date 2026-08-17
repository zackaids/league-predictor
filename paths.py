"""Central path + year helpers.

Every stage used to hardcode `2026` in its input/output paths, which made the
pipeline impossible to run for another year or from a scheduler. All path
construction now goes through here.

Layout:
    data/raw/{year}_LoL_esports_match_data_from_OraclesElixir.csv   (gitignored, ~80MB/yr)
    data/processed/{year}_{name}.{parquet,csv}                      (tracked, small)
    data/processed/region_strength.csv                              (cross-year, no year prefix)
    data/history/{name}.parquet                                     (tracked, append-only)
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
HISTORY_DIR = ROOT / "data" / "history"

META_PATH = RAW_DIR / "_fetch_meta.json"

CURRENT_YEAR = dt.date.today().year

# Oracle's Elixir publishes one file per year under this exact name.
RAW_STEM = "_LoL_esports_match_data_from_OraclesElixir"

# First year Oracle's Elixir data exists.
FIRST_YEAR = 2014


def raw_csv(year: int) -> Path:
    return RAW_DIR / f"{year}{RAW_STEM}.csv"


def raw_years_available() -> list[int]:
    """Years whose raw CSV is actually present on disk, ascending."""
    years = []
    for p in RAW_DIR.glob(f"*{RAW_STEM}.csv"):
        try:
            years.append(int(p.name[:4]))
        except ValueError:
            continue
    return sorted(years)


def processed(year: int, name: str, ext: str = "parquet") -> Path:
    return PROCESSED_DIR / f"{year}_{name}.{ext}"


def region_strength_csv() -> Path:
    """Cross-year artifact, so no year prefix."""
    return PROCESSED_DIR / "region_strength.csv"


def history(name: str) -> Path:
    return HISTORY_DIR / f"{name}.parquet"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, HISTORY_DIR):
        d.mkdir(parents=True, exist_ok=True)
