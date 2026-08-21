"""Streamlit front end for the Worlds predictor.

Reads the committed parquet/CSV outputs directly -- at ~50 teams and a few
thousand history rows there is nothing a database would buy here.

Deliberately includes a Model health page. The ratings were unvalidated for the
project's whole life, so the evidence that they work (and where they don't)
belongs in front of anyone reading the rankings, not buried in a script.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import streamlit as st

import backtest
import history
import paths
from paths import CURRENT_YEAR

st.set_page_config(page_title="Worlds Predictor", page_icon="🏆", layout="wide")

YEAR = CURRENT_YEAR


# ------------------------------------------------------------------ loading

@st.cache_data
def load_ratings(year: int = YEAR) -> pd.DataFrame:
    return pd.read_parquet(paths.processed(year, "calibrated_ratings"))


@st.cache_data
def load_odds(year: int = YEAR) -> pd.DataFrame:
    path = paths.processed(year, "worlds_title_odds", "csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_history() -> pd.DataFrame:
    h = history.load()
    if not h.empty:
        h["snapshot_date"] = pd.to_datetime(h["snapshot_date"])
    return h


@st.cache_data
def load_freshness() -> dict:
    return json.loads(paths.META_PATH.read_text()) if paths.META_PATH.exists() else {}


@st.cache_data
def load_report() -> dict | None:
    return backtest.load_report()


# ------------------------------------------------------------------ movement

# Movement is measured against a snapshot a chosen distance back, never against
# "the previous row": snapshots land weekly during a backfill but daily once the
# scheduler takes over, so the previous row is usually yesterday and every team
# would read as flat.
#
# "1 day" is that same measurement with days=1, not a special "previous row"
# case -- across the backfilled stretch of a season the snapshot nearest
# yesterday can be several days old, and the caption names the date it actually
# landed on. It is not the default: one day of play moves very few teams.
MOVE_WINDOWS = {"1 day": 1, "1 week": 7, "2 weeks": 14, "1 month": 30, "Season": None}
DEFAULT_WINDOW = "1 week"

SPARK_POINTS = 12


def _baseline_date(dates: pd.Series, days: int | None) -> pd.Timestamp:
    """The snapshot nearest to `days` ago -- nearest, not the newest one that
    is fully that old. Snapshot spacing jumps from weekly to daily mid-season,
    so "at least a week old" silently reaches back a fortnight instead."""
    if days is None:
        return dates.min()
    earlier = dates[dates < dates.max()]
    if earlier.empty:
        return dates.min()
    target = dates.max() - pd.Timedelta(days=days)
    return earlier.iloc[(earlier - target).abs().argmin()]


@st.cache_data
def load_movement(days: int | None) -> tuple[pd.DataFrame, str]:
    """Per-team rank/rating change and a recent `calibrated` sparkline.

    Ranks come from the *unfiltered* ranking on both ends. A team's arrow has
    to mean "it climbed", not "the league filter changed who is above it", so
    the movement column deliberately ignores the sidebar filters.
    """
    h = load_history()
    if h.empty:
        return pd.DataFrame(), ""

    dates = h["snapshot_date"].drop_duplicates().sort_values()
    base_date = _baseline_date(dates, days)
    base = (h[h["snapshot_date"] == base_date]
            .rename(columns={"rank": "prev_rank", "calibrated": "prev_calibrated"})
            [["team", "prev_rank", "prev_calibrated"]])

    recent = set(dates.nlargest(SPARK_POINTS))
    trend = (h[h["snapshot_date"].isin(recent)]
             .sort_values("snapshot_date")
             .groupby("team")["calibrated"].apply(list)
             .rename("trend").reset_index())

    return base.merge(trend, on="team", how="outer"), base_date.date().isoformat()


def _arrow(delta) -> str:
    """Rank delta as a glyph. Positive delta = moved up the table."""
    if pd.isna(delta):
        return "new"
    delta = int(delta)
    if delta > 0:
        return f"\u25b2 {delta}"
    if delta < 0:
        return f"\u25bc {abs(delta)}"
    return "\u2013"


def _move_style(val: str) -> str:
    if val.startswith("\u25b2"):
        return "color: #16a34a; font-weight: 600"
    if val.startswith("\u25bc"):
        return "color: #dc2626; font-weight: 600"
    return "color: rgba(128, 128, 128, 0.8)"


def data_age_banner() -> None:
    """The whole point of the scheduler is fresh data, so make staleness loud."""
    h = load_history()
    if h.empty:
        return
    as_of = pd.to_datetime(h["as_of_date"]).max().date()
    days = (dt.date.today() - as_of).days
    msg = f"Ratings reflect games through **{as_of}** ({days} days ago)."
    (st.warning if days > 14 else st.caption)(msg)


# ------------------------------------------------------------------ pages

def page_ranking() -> None:
    st.header("Cross-region power ranking")
    st.caption(
        "`calibrated` = region prior + within-league skill, both in Elo points. "
        "Raw Elo alone is not comparable across leagues -- see Model health."
    )
    df = load_ratings()

    c1, c2, c3 = st.columns([1, 1, 3])
    show_low = c1.checkbox("Include low-sample teams", value=False,
                           help="Teams with fewer than 20 rated games; their ratings are mostly noise.")
    window = c2.selectbox("Movement vs", list(MOVE_WINDOWS),
                          index=list(MOVE_WINDOWS).index(DEFAULT_WINDOW),
                          help="How far back the \u25b2/\u25bc column compares to.")
    leagues = c3.multiselect("Leagues", sorted(df["home_league"].unique()),
                             default=sorted(df["home_league"].unique()))

    # Rank over the whole table first: movement must survive the filters below,
    # otherwise deselecting a league would look like every team climbed.
    view = history.ranked(df)
    move, baseline = load_movement(MOVE_WINDOWS[window])
    if not move.empty:
        view = view.merge(move, on="team", how="left")
        view["move"] = (view["prev_rank"] - view["rank"]).map(_arrow)
        view["\u0394 rating"] = view["calibrated"] - view["prev_calibrated"]
        # A team with no history at all would leave NaN in an otherwise
        # list-valued column, which Arrow refuses to serialise.
        view["trend"] = view["trend"].map(lambda v: v if isinstance(v, list) else [])

    view = view[view["home_league"].isin(leagues)]
    if not show_low:
        view = view[~view["low_sample"]]
    view = view.sort_values("rank").reset_index(drop=True)
    # Places gained, kept before the display columns are dropped so the movers
    # line reads a number instead of parsing its own arrows back out.
    places = (view["prev_rank"] - view["rank"]) if "move" in view else None
    view.insert(0, "#", view["rank"])
    view["win_rate"] = view["win_rate"] * 100
    view = view.drop(columns=["rank", "prev_rank", "prev_calibrated"], errors="ignore")

    if "move" in view:
        view = view[["#", "move", "\u0394 rating", "trend"]
                    + [c for c in view.columns
                       if c not in {"#", "move", "\u0394 rating", "trend"}]]

    column_config = {
        "calibrated": st.column_config.ProgressColumn(
            "calibrated", format="%.1f",
            min_value=float(view["calibrated"].min()),
            max_value=float(view["calibrated"].max())),
        "win_rate": st.column_config.NumberColumn("win rate", format="%.1f%%"),
        "within_league": st.column_config.NumberColumn("within league", format="%+.1f"),
        "elo": st.column_config.NumberColumn(format="%.1f"),
        "region_elo": st.column_config.NumberColumn("region elo", format="%.1f"),
        "low_sample": st.column_config.CheckboxColumn("low sample"),
        "move": st.column_config.TextColumn(
            "move", help="Places gained or lost against the whole ranking, "
                         "not just the leagues shown."),
        "\u0394 rating": st.column_config.NumberColumn("\u0394 rating", format="%+.1f"),
        "trend": st.column_config.LineChartColumn(
            "trend", help=f"`calibrated` over the last {SPARK_POINTS} snapshots."),
    }

    # ProgressColumn rather than a pandas background_gradient: the latter needs
    # matplotlib, which is not worth a dependency for one column of shading.
    # The Styler is only carrying the arrow colours -- st.dataframe cannot tint
    # a single cell from column_config alone.
    styled = view.style.map(_move_style, subset=["move"]) if "move" in view else view
    st.dataframe(styled, width="stretch", hide_index=True, column_config=column_config)

    if "move" in view:
        st.caption(f"Movement measured against the {baseline} snapshot.")
        _movers_caption(view["team"], places)

    n_low = int(df["low_sample"].sum())
    if n_low and not show_low:
        st.caption(f"{n_low} low-sample team(s) hidden.")


def _movers_caption(teams: pd.Series, places: pd.Series) -> None:
    """One line naming the biggest riser and faller, so the table has a lede."""
    moved = pd.DataFrame({"team": teams, "places": places}).dropna()
    moved = moved[moved["places"] != 0]
    if moved.empty:
        st.caption("No rank changes over this window.")
        return
    bits = []
    up = moved.nlargest(1, "places").iloc[0]
    if up["places"] > 0:
        bits.append(f":green[\u25b2 {up['team']} +{int(up['places'])}]")
    down = moved.nsmallest(1, "places").iloc[0]
    if down["places"] < 0:
        bits.append(f":red[\u25bc {down['team']} {int(down['places'])}]")
    st.caption("Biggest movers: " + " \u00b7 ".join(bits))


def page_history() -> None:
    st.header("Rating history")
    h = load_history()
    if h.empty:
        st.info("No history yet. Run `python history.py --backfill`.")
        return

    latest = h[h["snapshot_date"] == h["snapshot_date"].max()]
    default = latest.nlargest(6, "calibrated")["team"].tolist()
    teams = st.multiselect("Teams", sorted(h["team"].unique()), default=default)
    metric = st.radio("Metric", ["calibrated", "elo", "rank"], horizontal=True)

    if not teams:
        st.info("Pick at least one team.")
        return

    wide = (h[h["team"].isin(teams)]
            .pivot_table(index="snapshot_date", columns="team", values=metric))
    st.line_chart(wide, height=420)
    if metric == "rank":
        st.caption("Lower is better; the axis is not inverted.")


def page_odds() -> None:
    st.header(f"Worlds {YEAR} — projected title odds")
    odds = load_odds()
    if odds.empty:
        st.info("No odds yet. Run `python monte_carlo.py`.")
        return

    st.warning(
        "The Worlds field is **not confirmed**. This is a projected field built "
        "from the top calibrated teams per league, per `worlds_field.toml`.",
        icon="⚠️",
    )

    top = odds.head(3)
    for col, (_, r) in zip(st.columns(3), top.iterrows()):
        col.metric(r["team"], f"{r['title_%']:.1f}%", f"{r['knockout_%']:.0f}% to knockouts")

    metric = st.radio("Show", ["title_%", "finals_%", "knockout_%"], horizontal=True)
    st.bar_chart(odds.set_index("team")[metric].sort_values(ascending=False), height=380)
    st.dataframe(odds, width="stretch", hide_index=True)


def page_health() -> None:
    st.header("Model health")
    report = load_report()
    if not report:
        st.info("No backtest report. Run `python backtest.py --report`.")
        return

    st.caption(f"Report generated {report['generated_at']} · "
               f"K={report['k']} · rating-gap scale={report['rating_scale']}")

    ins = report["in_season"]
    st.subheader("In-season prediction (walk-forward, no lookahead)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Log loss", f"{ins['model']['log_loss']:.4f}",
              f"{ins['model']['log_loss'] - ins['baseline_half']['log_loss']:+.4f} vs coin flip",
              delta_color="inverse")
    c2.metric("Accuracy", f"{ins['model']['accuracy']:.1%}",
              f"{ins['model']['accuracy'] - ins['baseline_blue']['accuracy']:+.1%} vs side base rate")
    c3.metric("Games scored", f"{ins['model']['n']:,}")

    cr = report.get("cross_region")
    if cr:
        st.subheader("Cross-region calibration (historical MSI / Worlds, point-in-time)")
        st.markdown(
            f"Calibrated ratings beat raw Elo in **{cr['cal_better']}/{cr['n_events']}** events "
            f"across **{cr['pooled_n']:,}** cross-region games."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Elo log loss", f"{cr['raw_logloss']:.4f}")
        c2.metric("Calibrated log loss", f"{cr['cal_logloss']:.4f}",
                  f"{cr['cal_logloss'] - cr['raw_logloss']:+.4f}", delta_color="inverse")
        c3.metric("Fitted gap scale", f"{cr['fitted_scale']:.2f}")

        if cr["raw_logloss"] > 0.6931:
            st.info(
                f"Raw Elo ({cr['raw_logloss']:.4f}) is worse than a coin flip (0.6931) on "
                "cross-region games. The region calibration is not a refinement here — it is "
                "what makes cross-region prediction work at all.",
                icon="💡",
            )

        st.dataframe(pd.DataFrame(cr["events"]), width="stretch", hide_index=True)

        st.subheader("Calibration curve — calibrated cross-region predictions")
        st.caption("`gap` = predicted − observed. Positive means overconfident.")
        st.dataframe(pd.DataFrame(cr["calibration"]), width="stretch", hide_index=True)

    st.subheader("Data freshness")
    meta = load_freshness()
    if meta:
        rows = [{"year": y, "fetched_at": m.get("fetched_at"),
                 "MB": round(m.get("bytes", 0) / 1e6, 1)} for y, m in sorted(meta.items())]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ------------------------------------------------------------------ shell

PAGES = {
    "Power ranking": page_ranking,
    "Rating history": page_history,
    "Worlds odds": page_odds,
    "Model health": page_health,
}

st.sidebar.title("🏆 Worlds Predictor")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("Oracle's Elixir data · updated by GitHub Actions")

data_age_banner()
PAGES[choice]()
