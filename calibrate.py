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

import glob
from pathlib import Path

import pandas as pd

from region_strength import LEAGUE_TO_REGION, INTL_LEAGUES

OUT_DIR = Path("data/processed")
TOP_LEAGUE_REGION = {"LCK": "KR", "LPL": "CN", "LEC": "EU", "LCS": "NA", "LCP": "APAC"}
# LCP merged PCS+VCS+LJL+LCO; anchor its teams to their real constituent region.
LCP_CONSTITUENTS = ["PCS", "VN", "JP", "OCE"]


def lcp_team_regions(teams: list[str]) -> dict[str, str]:
    """For each LCP team, its constituent region = modal domestic (non-LCP,
    non-international) league across 2023-2025. Fallback handled by caller."""
    hist = {}
    for f in sorted(glob.glob("data/raw/202[345]_*OraclesElixir.csv")):
        d = pd.read_csv(f, usecols=["league", "teamname", "position"], low_memory=False)
        t = d[(d["position"] == "team") & (d["teamname"].isin(teams))]
        t = t[~t["league"].isin(INTL_LEAGUES | {"LCP"})]
        for team, g in t.groupby("teamname"):
            regions = g["league"].map(LEAGUE_TO_REGION).dropna()
            for r in regions:
                hist.setdefault(team, []).append(r)
    return {t: max(set(rs), key=rs.count) for t, rs in hist.items()}


def home_league(tg: pd.DataFrame, region_elo: pd.Series) -> pd.DataFrame:
    """Each team's home top-league = modal league among the 5 qualifiers.
    LCP teams are re-anchored to their real constituent region (PCS/VN/JP/OCE)."""
    top = tg[tg["league"].isin(TOP_LEAGUE_REGION)]
    rows = []
    for team, g in top.groupby("team"):
        league = g["league"].value_counts().idxmax()
        rows.append({"team": team, "home_league": league,
                     "region": TOP_LEAGUE_REGION[league]})
    df = pd.DataFrame(rows)

    lcp = df[df["home_league"] == "LCP"]["team"].tolist()
    resolved = lcp_team_regions(lcp)
    composite = round(region_elo[LCP_CONSTITUENTS].mean(), 1)  # fallback for new orgs
    df["_composite_lcp"] = composite
    df.loc[df["home_league"] == "LCP", "region"] = (
        df.loc[df["home_league"] == "LCP", "team"].map(resolved)
        .fillna("_LCP_COMPOSITE")
    )
    return df


def calibrate() -> pd.DataFrame:
    elo = pd.read_parquet(OUT_DIR / "2026_team_elo.parquet")
    tg = pd.read_parquet(OUT_DIR / "2026_team_games.parquet")
    # keep_default_na=False: the region code "NA" (North America) must NOT be
    # parsed as a missing value.
    region = pd.read_csv(OUT_DIR / "region_strength.csv",
                         keep_default_na=False).set_index("region")["region_elo"]

    homes = home_league(tg, region)
    df = homes.merge(elo[["team", "elo", "n_games", "win_rate"]], on="team", how="left")

    league_mean = df.groupby("home_league")["elo"].transform("mean")
    df["within_league"] = (df["elo"] - league_mean).round(1)
    # region_elo: composite fallback for LCP teams with no constituent history
    df["region_elo"] = df["region"].map(region)
    df.loc[df["region"] == "_LCP_COMPOSITE", "region_elo"] = df["_composite_lcp"]
    df["region_elo"] = df["region_elo"].round(1)
    df["region"] = df["region"].replace("_LCP_COMPOSITE", "APAC*")
    df["calibrated"] = (df["region_elo"] + df["within_league"]).round(1)

    return df.sort_values("calibrated", ascending=False).reset_index(drop=True)[
        ["team", "home_league", "region", "elo", "within_league",
         "region_elo", "calibrated", "n_games", "win_rate"]
    ]


def win_prob(df: pd.DataFrame, a: str, b: str) -> float:
    ra = df.loc[df["team"] == a, "calibrated"].iloc[0]
    rb = df.loc[df["team"] == b, "calibrated"].iloc[0]
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def main() -> None:
    df = calibrate()
    df.to_parquet(OUT_DIR / "2026_calibrated_ratings.parquet", index=False)
    df.to_csv(OUT_DIR / "2026_calibrated_ratings.csv", index=False)
    print(f"Wrote {len(df)} calibrated ratings (5 Worlds-qualifying leagues)\n")
    print("=== TOP 20 — CROSS-REGION CALIBRATED POWER RANKING ===")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
