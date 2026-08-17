"""Historical region strength from international results (2014-2026).

Answers "which regions consistently show up and win" and produces a region
strength prior used to calibrate cross-region team ratings (see calibrate.py).

Method:
  1. For each year, a team's home region = the region of the domestic league it
     played the most games in that year (international events don't count).
  2. Take every international game (Worlds, MSI, IEM, First Stand, EWC), tag both
     sides with their home region, and keep only cross-region games.
  3. Run a region-level Elo over those games in chronological order across all
     years, with a mild pull toward 1500 at each year boundary (region strength
     drifts; recent results matter more).

Also reports plain historical dominance stats: international win rate and number
of Worlds appearances per region.

Output: data/processed/region_strength.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

import paths

# Verified against every raw CSV 2014-2026. `IWCI` (2016) and `MSC` (2020) are
# real cross-region events that were previously missing, so they were silently
# dropped from the region prior. `Rift Rivals`/`First Stand` were listed but
# match no league code in any year -- the real code is `FST`.
# Deliberately NOT included: `CCWS` (German ERL) and `Asia Master` (academy
# teams) look international but are not; their rosters confirm it.
INTL_LEAGUES = {"WLDs", "MSI", "IEM", "IWCI", "MSC", "FST", "EWC"}

# domestic league code (any year) -> stable macro-region
LEAGUE_TO_REGION = {
    # Korea
    "LCK": "KR", "Champions": "KR", "LCK CL": "KR", "LCKC": "KR",
    # China
    "LPL": "CN", "LSPL": "CN", "LDL": "CN",
    # Europe / EMEA
    "EU LCS": "EU", "LEC": "EU", "EU CS": "EU",
    # North America / Americas
    "NA LCS": "NA", "LCS": "NA", "NA CS": "NA", "LTA": "NA", "LTA N": "NA", "LTA S": "NA",
    # Taiwan/HK/SEA -> Pacific
    "LMS": "PCS", "PCS": "PCS", "GPL": "PCS",
    # Vietnam
    "VCS": "VN", "VCS A": "VN",
    # Japan
    "LJL": "JP",
    # Asia-Pacific merged top league (2025+)
    "LCP": "APAC",
    # Brazil / LatAm
    "CBLOL": "BR", "LLA": "LATAM", "CLS": "LATAM", "LLN": "LATAM", "CLASH": "LATAM",
    # Turkey / CIS / OCE (wildcards)
    "TCL": "TR", "LCL": "CIS", "LCO": "OCE", "OPL": "OCE", "OCS": "OCE",
}


def home_regions(year_df: pd.DataFrame) -> dict[str, str]:
    """team -> home region for one year, by modal DOMESTIC league."""
    dom = year_df[~year_df["league"].isin(INTL_LEAGUES)].copy()
    dom["region"] = dom["league"].map(LEAGUE_TO_REGION)
    dom = dom.dropna(subset=["region"])
    out = {}
    for team, g in dom.groupby("teamname"):
        out[team] = g["region"].value_counts().idxmax()
    return out


def load_intl_games() -> pd.DataFrame:
    """All cross-region international games across years, tagged with regions."""
    rows = []
    for year in paths.raw_years_available():
        d = pd.read_csv(paths.raw_csv(year),
                        usecols=["gameid", "league", "date", "teamname", "position", "result"],
                        low_memory=False)
        team = d[d["position"] == "team"].copy()
        region = home_regions(team)
        intl = team[team["league"].isin(INTL_LEAGUES)].copy()
        for gid, g in intl.groupby("gameid"):
            if len(g) != 2:
                continue
            won = g[g["result"] == 1]
            lost = g[g["result"] == 0]
            if len(won) != 1 or len(lost) != 1:
                continue
            w, l = won.iloc[0], lost.iloc[0]
            rw, rl = region.get(w["teamname"]), region.get(l["teamname"])
            if not rw or not rl or rw == rl:
                continue
            rows.append({"year": year, "date": w["date"], "event": w["league"],
                         "win_region": rw, "lose_region": rl})
    return pd.DataFrame(rows).sort_values(["date"]).reset_index(drop=True)


def region_elo(games: pd.DataFrame, k: float = 24, year_pull: float = 0.25) -> pd.DataFrame:
    rating: dict[str, float] = {}
    wins: dict[str, int] = {}
    tot: dict[str, int] = {}
    cur_year = None

    def get(r): return rating.get(r, 1500.0)

    for row in games.itertuples(index=False):
        if cur_year is not None and row.year != cur_year:
            # drift each region toward 1500 between years
            for r in rating:
                rating[r] += (1500.0 - rating[r]) * year_pull
        cur_year = row.year
        w, l = row.win_region, row.lose_region
        rw, rl = get(w), get(l)
        ew = 1 / (1 + 10 ** ((rl - rw) / 400))
        rating[w] = rw + k * (1 - ew)
        rating[l] = rl - k * (1 - ew)
        for r, win in ((w, 1), (l, 0)):
            wins[r] = wins.get(r, 0) + win
            tot[r] = tot.get(r, 0) + 1

    out = pd.DataFrame({
        "region": list(rating),
        "region_elo": [round(rating[r], 1) for r in rating],
        "intl_games": [tot[r] for r in rating],
        "intl_wins": [wins[r] for r in rating],
        "intl_winrate": [round(wins[r] / tot[r], 3) for r in rating],
    }).sort_values("region_elo", ascending=False).reset_index(drop=True)
    return out


def run(min_years: int = 5) -> pd.DataFrame:
    """Rebuild the cross-year region prior. Reads every raw CSV, so it is the
    slow stage -- only rerun it when an international event has completed."""
    years = paths.raw_years_available()
    if len(years) < min_years:
        # Raw CSVs are gitignored, so a fresh checkout (CI especially) has only
        # the current year. Recomputing from a partial archive would quietly
        # replace a prior built on 13 years of results with a near-worthless
        # one, and every calibrated rating downstream would inherit it.
        raise RuntimeError(
            f"Refusing to rebuild the region prior from {len(years)} year(s) of raw "
            f"data {years}: it is built from cross-year international results, and a "
            f"partial archive would overwrite a good prior with a bad one. "
            f"Run `python fetch_data.py --all` first."
        )
    print(f"Loading international games across {len(years)} years...")
    games = load_intl_games()
    print(f"  {len(games)} cross-region international games")
    print(f"  years {games['year'].min()}-{games['year'].max()}")

    elo = region_elo(games)
    paths.ensure_dirs()
    elo.to_csv(paths.region_strength_csv(), index=False)
    print("\n=== HISTORICAL REGION STRENGTH (recency-weighted Elo) ===")
    print(elo.to_string(index=False))

    # recent-only view (last 3 years) for "who's strong NOW"
    recent = region_elo(games[games["year"] >= games["year"].max() - 2])
    print("\n=== last 3 years only ===")
    print(recent[["region", "region_elo", "intl_games", "intl_winrate"]].head(8).to_string(index=False))
    return elo


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    run()


if __name__ == "__main__":
    main()
