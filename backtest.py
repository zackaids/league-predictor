"""Does any of this actually predict games? Nothing here was measured before.

Every constant in the pipeline (K=30, region k=24, year_pull=0.25, HALF_LIFE=45)
was picked by hand and never checked against an outcome. This module turns "the
ratings look plausible" into a number, so later changes can be justified instead
of argued about.

THREE TESTS
-----------
1. Walk-forward, in-season. Replay the season chronologically; predict each game
   using ONLY games already played, then update. Scored with log loss (a proper
   scoring rule, so it punishes confident-and-wrong far harder than accuracy
   does), Brier, and accuracy, against two baselines that cost nothing:
   a constant 0.5, and the blue-side base rate.

2. Calibration curve. Bin predictions and compare predicted probability to the
   observed win rate in each bin. This is the direct test of whether the model
   is overconfident -- if it says 80% and wins 65%, the Monte Carlo title odds
   built on top of it are inflated.

3. Cross-region holdout (the important one). calibrate.py claims that
   `region_prior + within_region_skill` makes teams from different leagues
   comparable. That claim is testable: for each historical MSI/Worlds, build
   ratings from that year's DOMESTIC games before the event, build the region
   prior from international games strictly before the event, then predict the
   event's cross-region games. Compare calibrated vs raw Elo. If calibration
   does not beat raw Elo here, it is not earning its complexity.

POINT-IN-TIME DISCIPLINE
------------------------
Test 3 recomputes the region prior from games strictly before each event.
region_strength.py normally builds it from all years at once, which would leak
the outcome of the very event being predicted back into its own prediction.
That is the whole reason this is a separate module rather than a flag on the
existing scripts.

Run:
    python backtest.py                # everything
    python backtest.py --quick        # skip the historical cross-region test
    python backtest.py --sweep-k      # K-factor sweep
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import elo
import paths
import region_strength
from paths import CURRENT_YEAR

BASE = elo.BASE
EPS = 1e-15


# ---------------------------------------------------------------- framing

def orient(team_rows: pd.DataFrame, team_col: str = "team",
           won_col: str = "won") -> pd.DataFrame:
    """One row per game: blue vs red, label = did blue win.

    Elo's own `build_matches` produces winner/loser, which makes the label
    constant-1 and hides any side effect. Blue/red gives a genuine 0/1 target
    and lets the side advantage show up in the baselines.
    """
    b = (team_rows[team_rows["side"] == "Blue"]
         [["gameid", "date", "league", team_col, won_col]]
         .rename(columns={team_col: "blue", won_col: "blue_won"}))
    r = (team_rows[team_rows["side"] == "Red"][["gameid", team_col]]
         .rename(columns={team_col: "red"}))
    m = b.merge(r, on="gameid", how="inner")
    m["blue_won"] = m["blue_won"].astype(bool)
    return m.sort_values(["date", "gameid"]).reset_index(drop=True)


# ---------------------------------------------------------------- scoring

def metrics(p, y) -> dict:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
    }


def shrink_prob(diff, scale: float) -> np.ndarray:
    """Elo win probability with the rating gap multiplied by `scale`."""
    return 1.0 / (1.0 + 10 ** (-scale * np.asarray(diff, float) / 400.0))


def fit_shrinkage(diff, y, grid=None) -> tuple[float, float]:
    """Find the multiplier on rating gaps that minimizes log loss.

    The calibrated scale adds a region prior on top of a within-region offset,
    and nothing ever checked that the sum lands on a real probability scale.
    When underdogs win more often than the gap implies, the gaps are too wide.
    A single global multiplier is the smallest honest fix, and it drops straight
    into monte_carlo.p_game.
    """
    grid = np.arange(0.30, 1.51, 0.01) if grid is None else grid
    losses = [metrics(shrink_prob(diff, s), y)["log_loss"] for s in grid]
    i = int(np.argmin(losses))
    return float(grid[i]), float(losses[i])


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed per probability bin. Overconfidence shows up as
    predicted > observed at the top and predicted < observed at the bottom."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = (df.groupby("bin", observed=True)
             .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")))
    out["gap"] = out["predicted"] - out["observed"]
    return out.round(3)


# ---------------------------------------------------------------- walk-forward

def walk_forward(matches: pd.DataFrame, k: float = elo.K_DEFAULT,
                 base: float = BASE) -> tuple[pd.DataFrame, dict, dict]:
    """Predict-then-update over the match list in order.

    Returns (predictions, final_ratings, games_played). `n_prior` on each
    prediction is the smaller of the two teams' game counts at the time, so a
    burn-in can be applied without re-running.
    """
    rating: dict[str, float] = {}
    played: dict[str, int] = {}
    rows = []

    for m in matches.itertuples(index=False):
        rb, rr = rating.get(m.blue, base), rating.get(m.red, base)
        p = elo.expected(rb, rr)  # reuse the shipped formula, don't restate it
        rows.append((m.date, m.gameid, m.league, m.blue, m.red,
                     int(m.blue_won), p, min(played.get(m.blue, 0), played.get(m.red, 0))))

        s = 1.0 if m.blue_won else 0.0
        rating[m.blue] = rb + k * (s - p)
        rating[m.red] = rr + k * ((1 - s) - (1 - p))
        played[m.blue] = played.get(m.blue, 0) + 1
        played[m.red] = played.get(m.red, 0) + 1

    preds = pd.DataFrame(rows, columns=["date", "gameid", "league", "blue",
                                        "red", "y", "p", "n_prior"])
    return preds, rating, played


# ---------------------------------------------------------------- test 1 + 2

def in_season(year: int = CURRENT_YEAR, k: float = elo.K_DEFAULT,
              burn_in: int = 10, all_leagues: bool = False) -> pd.DataFrame:
    tg = elo.scoped_team_games(year, all_leagues)
    matches = orient(tg)
    preds, _, _ = walk_forward(matches, k)
    evaluated = preds[preds["n_prior"] >= burn_in]

    print(f"\nWalk-forward on {year}: {len(matches)} games, "
          f"{len(evaluated)} scored after a {burn_in}-game burn-in per team")

    y = evaluated["y"].to_numpy()
    blue_rate = float(y.mean())
    rows = {
        f"elo (K={k:g})": metrics(evaluated["p"], y),
        "baseline: always 0.5": metrics(np.full(len(y), 0.5), y),
        f"baseline: blue rate ({blue_rate:.3f})": metrics(np.full(len(y), blue_rate), y),
    }
    table = pd.DataFrame(rows).T[["n", "log_loss", "brier", "accuracy"]]
    table["n"] = table["n"].astype(int)
    print("\n=== TEST 1: in-season predictive accuracy ===")
    print(table.round(4).to_string())

    print("\n=== TEST 2: calibration (predicted vs observed) ===")
    print(calibration_table(evaluated["p"], y).to_string())
    _verdict(evaluated["p"].to_numpy(), y)
    return evaluated


def _verdict(p, y) -> None:
    """Flag over/under-confidence in the tails, where it matters for the sims."""
    hi, lo = p >= 0.7, p <= 0.3
    msgs = []
    if hi.sum() >= 30:
        msgs.append(f"predictions >=0.70: said {p[hi].mean():.3f}, won {y[hi].mean():.3f}")
    if lo.sum() >= 30:
        msgs.append(f"predictions <=0.30: said {p[lo].mean():.3f}, won {y[lo].mean():.3f}")
    for m in msgs:
        print("  " + m)
    if hi.sum() >= 30 and p[hi].mean() - y[hi].mean() > 0.03:
        print("  -> OVERCONFIDENT at the top end; Monte Carlo title odds will be inflated.")


# ---------------------------------------------------------------- test 3

def _year_team_rows(year: int) -> pd.DataFrame:
    d = pd.read_csv(paths.raw_csv(year),
                    usecols=["gameid", "league", "date", "teamname", "position",
                             "result", "side"],
                    low_memory=False)
    t = d[d["position"] == "team"].copy()
    t["date"] = pd.to_datetime(t["date"])
    t["won"] = t["result"].astype(bool)
    # Raw files call it `teamname`; clean.py renames it to `team`. Provide both:
    # orient() expects `team`, region_strength.home_regions() expects `teamname`.
    t["team"] = t["teamname"]
    return t


def cross_region(k: float = elo.K_DEFAULT, events=("MSI", "WLDs"),
                 min_games: int = 10) -> pd.DataFrame:
    """Predict historical international games from pre-event information only."""
    print("\nLoading cross-region international games (all years)...")
    intl = region_strength.load_intl_games()
    intl["date"] = pd.to_datetime(intl["date"])

    results = []
    pooled_raw, pooled_cal, pooled_y, pooled_diff = [], [], [], []
    for year in paths.raw_years_available():
        t = _year_team_rows(year)
        if not set(events) & set(t["league"].unique()):
            continue
        regions = region_strength.home_regions(t)

        for event in events:
            rows = t[t["league"] == event]
            if rows["gameid"].nunique() < 5:
                continue
            start = rows["date"].min()

            # Train on this year's DOMESTIC games only, strictly before the event.
            train = t[(~t["league"].isin(region_strength.INTL_LEAGUES))
                      & (t["date"] < start)]
            if train.empty:
                continue
            _, rating, played = walk_forward(orient(train), k)

            # Region prior from international games strictly before the event.
            prior_games = intl[intl["date"] < start]
            if prior_games.empty:
                continue
            prior = region_strength.region_elo(prior_games).set_index("region")["region_elo"]

            # Mean rating per region, for the within-region offset.
            pool = defaultdict(list)
            for team, r in rating.items():
                reg = regions.get(team)
                if reg and played.get(team, 0) >= min_games:
                    pool[reg].append(r)
            region_mean = {reg: float(np.mean(v)) for reg, v in pool.items()}

            p_raw, p_cal, ys, diffs = [], [], [], []
            for m in orient(rows).itertuples(index=False):
                rb, rr = rating.get(m.blue), rating.get(m.red)
                gb, gr = regions.get(m.blue), regions.get(m.red)
                if rb is None or rr is None or not gb or not gr or gb == gr:
                    continue
                if min(played.get(m.blue, 0), played.get(m.red, 0)) < min_games:
                    continue
                cb = prior.get(gb, BASE) + (rb - region_mean.get(gb, BASE))
                cr = prior.get(gr, BASE) + (rr - region_mean.get(gr, BASE))
                p_raw.append(elo.expected(rb, rr))
                p_cal.append(elo.expected(cb, cr))
                diffs.append(cb - cr)
                ys.append(int(m.blue_won))

            if len(ys) < 10:
                continue
            pooled_raw += p_raw
            pooled_cal += p_cal
            pooled_y += ys
            pooled_diff += diffs
            raw_m, cal_m = metrics(p_raw, ys), metrics(p_cal, ys)
            results.append({
                "year": year, "event": event, "n": len(ys),
                "raw_logloss": round(raw_m["log_loss"], 4),
                "cal_logloss": round(cal_m["log_loss"], 4),
                "raw_acc": round(raw_m["accuracy"], 3),
                "cal_acc": round(cal_m["accuracy"], 3),
                "cal_better": cal_m["log_loss"] < raw_m["log_loss"],
            })
            print(f"  {year} {event}: {len(ys)} cross-region games, "
                  f"raw {raw_m['log_loss']:.4f} vs calibrated {cal_m['log_loss']:.4f}")

    df = pd.DataFrame(results)
    if df.empty:
        print("No evaluable cross-region events found.")
        return df

    print("\n=== TEST 3: cross-region calibration, point-in-time ===")
    print(df.to_string(index=False))
    w = df["n"]
    raw, cal = np.average(df["raw_logloss"], weights=w), np.average(df["cal_logloss"], weights=w)
    print(f"\n  games-weighted log loss:  raw Elo {raw:.4f}   calibrated {cal:.4f}")
    print(f"  calibrated wins {int(df['cal_better'].sum())}/{len(df)} events")
    print(f"  -> calibration {'HELPS' if cal < raw else 'DOES NOT HELP'} "
          f"({'-' if cal < raw else '+'}{abs(cal - raw):.4f} log loss)")
    if raw > 0.6931:
        print(f"  -> NOTE: raw Elo ({raw:.4f}) is WORSE than a coin flip on "
              f"cross-region games. Calibration is not a refinement here, it is "
              f"what makes cross-region prediction work at all.")

    # This is the scale monte_carlo.py actually simulates on, so its calibration
    # is what decides whether the title odds are trustworthy.
    print("\n=== calibrated cross-region predictions: calibration curve ===")
    print(calibration_table(pooled_cal, pooled_y).to_string())
    _verdict(np.asarray(pooled_cal), np.asarray(pooled_y, dtype=float))

    scale, fitted_loss = fit_shrinkage(pooled_diff, pooled_y)
    print(f"\n=== fitted rating-gap multiplier ===")
    print(f"  best scale {scale:.2f}  (1.00 = leave gaps as they are)")
    print(f"  log loss {cal:.4f} -> {fitted_loss:.4f}")
    if scale < 1.0:
        print(f"  -> gaps are {1 / scale:.2f}x too wide; shrinking them by {scale:.2f} "
              f"is what monte_carlo.p_game should use.")
        print("\n  calibration curve after shrinking:")
        print(calibration_table(shrink_prob(pooled_diff, scale), pooled_y).to_string())
    return df


# ---------------------------------------------------------------- sweep

def sweep_k(year: int = CURRENT_YEAR, ks=(10, 16, 20, 24, 30, 40, 50),
            burn_in: int = 10) -> pd.DataFrame:
    tg = elo.scoped_team_games(year)
    matches = orient(tg)
    rows = []
    for k in ks:
        preds, _, _ = walk_forward(matches, k)
        ev = preds[preds["n_prior"] >= burn_in]
        m = metrics(ev["p"], ev["y"])
        rows.append({"K": k, **{x: round(m[x], 4) for x in ("log_loss", "brier", "accuracy")}})
    df = pd.DataFrame(rows)
    best = df.loc[df["log_loss"].idxmin(), "K"]
    print("\n=== K-factor sweep (in-season walk-forward) ===")
    print(df.to_string(index=False))
    print(f"\n  best K by log loss: {best}   (shipped default is {elo.K_DEFAULT})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    ap.add_argument("--k", type=float, default=elo.K_DEFAULT)
    ap.add_argument("--burn-in", type=int, default=10,
                    help="skip games where either team has fewer prior games")
    ap.add_argument("--quick", action="store_true",
                    help="skip the historical cross-region test (reads every raw CSV)")
    ap.add_argument("--sweep-k", action="store_true")
    args = ap.parse_args()

    in_season(args.year, args.k, args.burn_in)
    if args.sweep_k:
        sweep_k(args.year, burn_in=args.burn_in)
    if not args.quick:
        cross_region(args.k)


if __name__ == "__main__":
    main()
