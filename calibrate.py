"""Cross-region calibration: put every top-league team on ONE comparable scale.

Problem (see modeling notes): pure Elo is only valid within a connected pool, so
a team's raw Elo carries its league's arbitrary drift. A 1700 in LCP is not a
1700 in LCK.

Fix: split each team's rating into
    within-league skill   = team_elo - (mean team_elo of its league)
    league strength level = region_elo of that league   (region_strength.py)
and recombine:
    calibrated = region_elo[region] + (team_elo - league_mean_elo)

Both terms are in Elo points (400 = 10x odds), so they add directly. The region
term is anchored by 13 years of international results, so a mid-table LCK team can
correctly outrank the best LCP team.

Scope = the five Worlds-qualifying top leagues (LCK, LPL, LEC, LCS, LCP). Domestic
feeders (LJL, PCS, VCS, ...) are excluded: their teams can't reach Worlds unless
they're in one of these (e.g. HAWKS/DFM play LCP, not just LJL).

Inputs : data/processed/2026_team_elo.parquet    (elo.py, default scope)
         data/processed/2026_team_games.parquet   (to find each team's home league)
         data/processed/region_strength.csv        (region_strength.py)
Output : data/processed/2026_calibrated_ratings.parquet (+ .csv)
"""
from __future__ import annotations

import argparse

import pandas as pd

import paths
from paths import CURRENT_YEAR
from region_strength import LEAGUE_TO_REGION, INTL_LEAGUES
TOP_LEAGUE_REGION = {"LCK": "KR", "LPL": "CN", "LEC": "EU", "LCS": "NA", "LCP": "APAC"}

# Multiplier applied to calibrated rating gaps when converting to a win
# probability.
#
# `calibrated` is a SUM of two independently-scaled Elo quantities (a region
# prior plus a within-region offset), and nothing ever checked that the sum
# lands on a real probability scale. It does not: measured on 1,579 historical
# cross-region games, underdogs win materially more often than the unshrunk
# formula implies (the 0.1-0.2 predicted bin actually won 31% of the time).
# Gaps are ~1.32x too wide, so we shrink by 0.76.
#
# This matters most for monte_carlo.py, where a too-wide gap compounds over
# every simulated game and inflates the favourite's title odds.
# Re-fit with `python backtest.py` whenever the rating logic changes.
RATING_SCALE = 0.76

# Below this many rated games, a team's Elo is dominated by where it started
# rather than how it played. Such teams stay in the table but are flagged and
# kept out of the league-mean baseline.
MIN_GAMES = 20


def prob_from_gap(gap: float) -> float:
    """Win probability for a calibrated rating gap, with the fitted shrinkage."""
    return 1.0 / (1.0 + 10 ** (-RATING_SCALE * gap / 400.0))
# LCP merged PCS+VCS+LJL+LCO; anchor its teams to their real constituent region.
LCP_CONSTITUENTS = ["PCS", "VN", "JP", "OCE"]


def lcp_team_regions(teams: list[str], year: int = CURRENT_YEAR) -> dict[str, str]:
    """For each LCP team, its constituent region = modal domestic (non-LCP,
    non-international) league across the 3 seasons before `year`. Fallback
    handled by caller."""
    hist = {}
    lookback = [y for y in paths.raw_years_available() if year - 3 <= y < year]
    for y in lookback:
        d = pd.read_csv(paths.raw_csv(y), usecols=["league", "teamname", "position"],
                        low_memory=False)
        t = d[(d["position"] == "team") & (d["teamname"].isin(teams))]
        t = t[~t["league"].isin(INTL_LEAGUES | {"LCP"})]
        for team, g in t.groupby("teamname"):
            regions = g["league"].map(LEAGUE_TO_REGION).dropna()
            for r in regions:
                hist.setdefault(team, []).append(r)
    return {t: max(set(rs), key=rs.count) for t, rs in hist.items()}


def home_league(tg: pd.DataFrame, region_elo: pd.Series,
                year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Each team's home top-league = modal league among the 5 qualifiers.
    LCP teams are re-anchored to their real constituent region (PCS/VN/JP/OCE).

    `tg` must be the UNSCOPED team-games table. The check below compares a
    team's top-league games against all its domestic play, and the ERL leagues
    that expose imposters (EM, LFL, ...) sit outside elo.MAJOR_SCOPE.

    A team only counts as belonging to a top league if that league is also its
    modal league across ALL domestic play. Counting top-5 games alone classified
    Karmine Corp Blue -- 28 EMEA Masters + 20 LFL + 11 LEC games -- as an LEC
    team, putting an ERL roster in the Worlds power ranking and dragging the LEC
    mean that every other LEC team is measured against.
    """
    domestic = tg[~tg["league"].isin(INTL_LEAGUES)]
    true_home = domestic.groupby("team")["league"].agg(lambda s: s.value_counts().idxmax())

    top = tg[tg["league"].isin(TOP_LEAGUE_REGION)]
    rows, rejected = [], []
    for team, g in top.groupby("team"):
        league = g["league"].value_counts().idxmax()
        if true_home.get(team) != league:
            rejected.append(f"{team} ({true_home.get(team)}, not {league})")
            continue
        rows.append({"team": team, "home_league": league,
                     "region": TOP_LEAGUE_REGION[league]})
    if rejected:
        print(f"  excluded {len(rejected)} team(s) whose real home is not a top league: "
              f"{', '.join(rejected)}")
    df = pd.DataFrame(rows)

    lcp = df[df["home_league"] == "LCP"]["team"].tolist()
    resolved = lcp_team_regions(lcp, year)
    composite = round(region_elo[LCP_CONSTITUENTS].mean(), 1)  # fallback for new orgs
    df["_composite_lcp"] = composite
    df.loc[df["home_league"] == "LCP", "region"] = (
        df.loc[df["home_league"] == "LCP", "team"].map(resolved)
        .fillna("_LCP_COMPOSITE")
    )
    return df


def load_region_prior() -> pd.Series:
    # keep_default_na=False: the region code "NA" (North America) must NOT be
    # parsed as a missing value.
    return pd.read_csv(paths.region_strength_csv(),
                       keep_default_na=False).set_index("region")["region_elo"]


def combine(homes: pd.DataFrame, elo: pd.DataFrame, region: pd.Series,
            min_games: int = MIN_GAMES) -> pd.DataFrame:
    """Recombine within-league skill with the region prior onto one scale.

    Split out from `calibrate()` so history.py can replay it for past snapshots
    without duplicating -- and drifting from -- the formula.
    """
    df = homes.merge(elo[["team", "elo", "n_games", "win_rate"]], on="team", how="left")

    # A rating built on a handful of games is noise, and it is contagious: the
    # league mean is the baseline every other team in that league is measured
    # against, so one 11-game team that went 2-9 shifts everybody. Flag them,
    # and compute the mean from established teams only.
    df["low_sample"] = df["n_games"].fillna(0) < min_games
    established = df[~df["low_sample"]]
    league_mean = (df["home_league"].map(established.groupby("home_league")["elo"].mean())
                   .fillna(df.groupby("home_league")["elo"].transform("mean")))
    df["within_league"] = (df["elo"] - league_mean).round(1)
    # region_elo: composite fallback for LCP teams with no constituent history
    df["region_elo"] = df["region"].map(region)
    df.loc[df["region"] == "_LCP_COMPOSITE", "region_elo"] = df["_composite_lcp"]
    df["region_elo"] = df["region_elo"].round(1)
    df["region"] = df["region"].replace("_LCP_COMPOSITE", "APAC*")
    df["calibrated"] = (df["region_elo"] + df["within_league"]).round(1)

    return df.sort_values("calibrated", ascending=False).reset_index(drop=True)[
        ["team", "home_league", "region", "elo", "within_league",
         "region_elo", "calibrated", "n_games", "win_rate", "low_sample"]
    ]


def calibrate(year: int = CURRENT_YEAR) -> pd.DataFrame:
    elo = pd.read_parquet(paths.processed(year, "team_elo"))
    tg = pd.read_parquet(paths.processed(year, "team_games"))
    region = load_region_prior()
    return combine(home_league(tg, region, year), elo, region)


def win_prob(df: pd.DataFrame, a: str, b: str) -> float:
    ra = df.loc[df["team"] == a, "calibrated"].iloc[0]
    rb = df.loc[df["team"] == b, "calibrated"].iloc[0]
    return prob_from_gap(ra - rb)


def run(year: int = CURRENT_YEAR) -> pd.DataFrame:
    df = calibrate(year)
    paths.ensure_dirs()
    df.to_parquet(paths.processed(year, "calibrated_ratings"), index=False)
    df.to_csv(paths.processed(year, "calibrated_ratings", "csv"), index=False)
    print(f"Wrote {len(df)} calibrated ratings (5 Worlds-qualifying leagues)\n")
    print("=== TOP 20 — CROSS-REGION CALIBRATED POWER RANKING ===")
    print(df.head(20).to_string(index=False))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=CURRENT_YEAR)
    run(ap.parse_args().year)


if __name__ == "__main__":
    main()
