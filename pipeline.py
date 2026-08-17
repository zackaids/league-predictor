"""Run the rating pipeline end to end, for schedulers and humans alike.

Two paths, because the stages have very different costs:

  fast (default)  clean -> elo -> calibrate -> monte_carlo, for one season.
                  Reads a single raw CSV (~80MB).
  full            additionally rebuilds region_strength, which reads *every*
                  raw CSV (13 files, ~790MB). The region prior only moves when
                  an international event completes, so it has no business
                  running on the frequent schedule.

With --fetch, the current year's CSV is downloaded first and the pipeline exits
early when its content hash is unchanged. Oracle's Elixir republishes the file
continuously, so "was it re-downloaded" is a useless signal; "did the bytes
change" is the real one.

Usage:
    python pipeline.py                      # fast, current year, no fetch
    python pipeline.py --fetch              # fetch first, skip work if unchanged
    python pipeline.py --fetch --full       # also rebuild the region prior
    python pipeline.py --year 2025 --full
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager

import paths
from paths import CURRENT_YEAR


@contextmanager
def stage(name: str):
    print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}")
    t0 = time.monotonic()
    yield
    print(f"-- {name} finished in {time.monotonic() - t0:.1f}s")


def emit_github_output(**kv) -> None:
    """Expose results to a GitHub Actions workflow, when running in one."""
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a") as f:
        for k, v in kv.items():
            f.write(f"{k}={str(v).lower() if isinstance(v, bool) else v}\n")


def run(year: int = CURRENT_YEAR, *, fetch: bool = False, full: bool = False,
        n_sims: int = 20000, seed: int | None = 0, force: bool = False) -> bool:
    """Run the pipeline. Returns True if it actually did work."""
    # Imported lazily so `--help` doesn't pay for pandas.
    import calibrate
    import clean
    import elo
    import monte_carlo

    paths.ensure_dirs()

    if fetch:
        import fetch_data
        with stage(f"fetch {year}"):
            _, changed = fetch_data.fetch_year(year, force=force)
        if not changed and not force:
            print("\nRaw data unchanged since last fetch -- nothing to recompute.")
            emit_github_output(changed=False)
            return False

    if full:
        import region_strength
        with stage("region_strength (all years)"):
            region_strength.run()
    elif not paths.region_strength_csv().exists():
        # calibrate.py cannot run without it, so bootstrap on first use.
        import region_strength
        with stage("region_strength (missing, bootstrapping)"):
            region_strength.run()

    with stage(f"clean {year}"):
        clean.run(year)
    with stage(f"elo {year}"):
        elo.run(year)
    with stage(f"calibrate {year}"):
        calibrate.run(year)
    with stage(f"monte_carlo {year}"):
        monte_carlo.simulate_and_write(year, n_sims, seed)

    import history
    with stage("record history snapshot"):
        history.record(year)

    emit_github_output(changed=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--fetch", action="store_true", help="download the raw CSV first")
    ap.add_argument("--full", action="store_true",
                    help="also rebuild region_strength from every year (slow)")
    ap.add_argument("--n-sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0,
                    help="Monte Carlo seed for reproducible odds (-1 for random)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the raw data is unchanged")
    args = ap.parse_args()

    run(args.year, fetch=args.fetch, full=args.full, n_sims=args.n_sims,
        seed=None if args.seed == -1 else args.seed, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
