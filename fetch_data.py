"""Reliable fetcher for Oracle's Elixir LoL esports match data.

Source of truth is the public Google Drive folder that oracleselixir.com/tools/downloads
serves from:
    https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH

The website itself sits behind Cloudflare and 403s automated requests, so we go
straight to Drive. Each year is a separate CSV with a stable Drive file id. The
historical files (2014..last year) effectively never change; the current-year
file is updated continuously as new matches are played.

Usage:
    python fetch_data.py                 # fetch current year only (default)
    python fetch_data.py --years 2024 2025 2026
    python fetch_data.py --all           # fetch every year
    python fetch_data.py --discover      # re-enumerate Drive folder, print live ids
    python fetch_data.py --all --force   # ignore freshness cache, re-download
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import gdown

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

RAW_DIR = Path(__file__).parent / "data" / "raw"
META_PATH = RAW_DIR / "_fetch_meta.json"

# Stable Drive file ids per year (captured from a folder enumeration).
# If a download starts failing, run `--discover` to refresh these.
FILE_IDS: dict[int, str] = {
    2014: "12syQsRH2QnKrQZTQQ6G5zyVeTG2pAYvu",
    2015: "1qyckLuw0-hJM8XqFhlV9l1xAbr3H78T_",
    2016: "1muyfpaIqk8_0BFkgLCWXDGNgWSXoPBwG",
    2017: "11fx3nNjSYB0X8vKxLAbYOrS2Bu6avm9A",
    2018: "1GsNetJQOMx0QJ6_FN8M1kwGvU_GPPcPZ",
    2019: "11eKtScnZcpfZcD3w3UrD7nnpfLHvj9_t",
    2020: "1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d",
    2021: "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
    2022: "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
    2023: "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
}

CURRENT_YEAR = dt.date.today().year


def filename(year: int) -> str:
    return f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def save_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True))


def discover() -> dict[int, str]:
    """Re-enumerate the Drive folder and return a fresh {year: file_id} map.

    Used to detect/repair drift if the hardcoded ids ever change.
    """
    files = gdown.download_folder(
        FOLDER_URL, skip_download=True, quiet=True, use_cookies=False
    )
    found: dict[int, str] = {}
    for f in files or []:
        name = Path(f.path).name
        try:
            year = int(name.split("_", 1)[0])
        except (ValueError, IndexError):
            continue
        found[year] = f.id
    return dict(sorted(found.items()))


def fetch_year(year: int, force: bool = False) -> Path:
    """Download one year's CSV. Skips current-version historical files unless forced."""
    if year not in FILE_IDS:
        raise KeyError(f"No known Drive file id for {year}. Run --discover.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / filename(year)
    meta = load_meta()
    is_historical = year < CURRENT_YEAR

    # Historical files don't change: if present and previously fetched, skip.
    if not force and is_historical and dest.exists() and str(year) in meta:
        print(f"[{year}] up to date (historical), skipping")
        return dest

    url = f"https://drive.google.com/uc?id={FILE_IDS[year]}"
    print(f"[{year}] downloading -> {dest.name}")
    gdown.download(url, str(dest), quiet=False)

    meta[str(year)] = {
        "file_id": FILE_IDS[year],
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "bytes": dest.stat().st_size,
    }
    save_meta(meta)
    return dest


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", help="specific years to fetch")
    p.add_argument("--all", action="store_true", help="fetch every known year")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.add_argument("--discover", action="store_true", help="re-enumerate Drive folder and print live ids")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.discover:
        live = discover()
        print(json.dumps(live, indent=2))
        drift = {y: i for y, i in live.items() if FILE_IDS.get(y) != i}
        if drift:
            print("\nWARNING: ids differ from hardcoded FILE_IDS:", file=sys.stderr)
            print(json.dumps(drift, indent=2), file=sys.stderr)
        else:
            print("\nAll live ids match hardcoded FILE_IDS.")
        return 0

    if args.all:
        years = sorted(FILE_IDS)
    elif args.years:
        years = args.years
    else:
        years = [CURRENT_YEAR]

    for year in years:
        fetch_year(year, force=args.force)

    print("\nDone. Files in", RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
