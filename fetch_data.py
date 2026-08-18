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
import hashlib
import json
import sys
import time
from pathlib import Path

import gdown

import paths
from paths import CURRENT_YEAR, META_PATH, RAW_DIR

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

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

# Drive rate-limits heavily-shared public files ("Too many users have viewed or
# downloaded this file recently"). It is by far the most common way a scheduled
# run dies, it has nothing to do with this repo, and it clears itself -- so it
# gets its own exception type that callers can choose to skip a run over.
MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = (15, 60)

# Substrings of upstream errors that mean "come back later", not "you're broken".
_TRANSIENT_MARKERS = (
    "too many users have viewed",
    "quota",
    "rate limit",
    "try again later",
    "temporarily unavailable",
    "service unavailable",
    "internal error",
)

# Every Oracle's Elixir CSV starts with this column.
_CSV_HEADER_PREFIX = b"gameid,"


class TransientFetchError(RuntimeError):
    """Drive refused the download for a reason that typically clears on its own."""


def _is_transient(exc: BaseException) -> bool:
    # requests' exceptions subclass IOError/OSError, as do plain socket errors.
    if isinstance(exc, (OSError, TransientFetchError)):
        return True
    return any(m in str(exc).lower() for m in _TRANSIENT_MARKERS)


def _validate_csv(path: Path) -> None:
    """Reject anything that isn't actually one of these CSVs.

    Under quota Drive answers *200 OK* with an HTML "Quota exceeded" page, and a
    dropped connection leaves a truncated file. Either one written to `dest`
    would be hashed and recorded as legitimate new content -- the scheduler would
    then commit a bogus sha256 that every later run measures against.
    """
    with open(path, "rb") as f:
        head = f.read(len(_CSV_HEADER_PREFIX))
    if head != _CSV_HEADER_PREFIX:
        raise TransientFetchError(
            f"{path.name}: not a match-data CSV (starts with {head!r}); "
            "Drive most likely served an error page"
        )


def _download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest`, retrying transient failures.

    Writes to a sidecar file and renames only once the payload looks right, so a
    failed attempt can never corrupt the cached CSV.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            tmp.unlink(missing_ok=True)
            if gdown.download(url, str(tmp), quiet=False) is None:
                raise TransientFetchError(f"gdown returned no file for {url}")
            _validate_csv(tmp)
            tmp.replace(dest)
            return
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if not _is_transient(exc):
                raise
            if attempt == MAX_ATTEMPTS:
                raise TransientFetchError(
                    f"giving up after {MAX_ATTEMPTS} attempts: {exc}"
                ) from exc
            wait = RETRY_WAIT_SECONDS[min(attempt - 1, len(RETRY_WAIT_SECONDS) - 1)]
            print(f"  transient download failure ({exc.__class__.__name__}); "
                  f"retrying in {wait}s [{attempt}/{MAX_ATTEMPTS}]", file=sys.stderr)
            time.sleep(wait)


def sha256(path: Path) -> str:
    """Content hash, so the scheduler can tell a changed file from a re-download."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def fetch_year(year: int, force: bool = False) -> tuple[Path, bool]:
    """Download one year's CSV.

    Returns (path, content_changed). `content_changed` is what the scheduler acts
    on: the current-year file is re-downloaded constantly but only actually
    differs when new matches have been played.
    """
    if year not in FILE_IDS:
        raise KeyError(f"No known Drive file id for {year}. Run --discover.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = paths.raw_csv(year)
    meta = load_meta()
    is_historical = year < CURRENT_YEAR

    # Historical files don't change: if present and previously fetched, skip.
    if not force and is_historical and dest.exists() and str(year) in meta:
        print(f"[{year}] up to date (historical), skipping")
        return dest, False

    url = f"https://drive.google.com/uc?id={FILE_IDS[year]}"
    print(f"[{year}] downloading -> {dest.name}")
    _download(url, dest)

    digest = sha256(dest)
    prev = meta.get(str(year), {}).get("sha256")
    changed = prev != digest

    meta[str(year)] = {
        "file_id": FILE_IDS[year],
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "bytes": dest.stat().st_size,
        "sha256": digest,
    }
    save_meta(meta)
    print(f"[{year}] {'content CHANGED' if changed else 'content identical to last fetch'}")
    return dest, changed


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

    try:
        changed = [y for y in years if fetch_year(y, force=args.force)[1]]
    except TransientFetchError as exc:
        print(f"\nUpstream is throttling us, nothing downloaded: {exc}", file=sys.stderr)
        print("This clears on its own; try again later.", file=sys.stderr)
        return 1

    print("\nDone. Files in", RAW_DIR)
    print("Years with new content:", changed or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
