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

    c1, c2 = st.columns([1, 3])
    show_low = c1.checkbox("Include low-sample teams", value=False,
                           help="Teams with fewer than 20 rated games; their ratings are mostly noise.")
    leagues = c2.multiselect("Leagues", sorted(df["home_league"].unique()),
                             default=sorted(df["home_league"].unique()))

    view = df[df["home_league"].isin(leagues)]
    if not show_low:
        view = view[~view["low_sample"]]
    view = view.reset_index(drop=True)
    view.insert(0, "#", view.index + 1)
    view["win_rate"] = view["win_rate"] * 100

    # ProgressColumn rather than a pandas background_gradient: the latter needs
    # matplotlib, which is not worth a dependency for one column of shading.
    st.dataframe(
        view, width="stretch", hide_index=True,
        column_config={
            "calibrated": st.column_config.ProgressColumn(
                "calibrated", format="%.1f",
                min_value=float(view["calibrated"].min()),
                max_value=float(view["calibrated"].max())),
            "win_rate": st.column_config.NumberColumn("win rate", format="%.1f%%"),
            "within_league": st.column_config.NumberColumn("within league", format="%+.1f"),
            "elo": st.column_config.NumberColumn(format="%.1f"),
            "region_elo": st.column_config.NumberColumn("region elo", format="%.1f"),
            "low_sample": st.column_config.CheckboxColumn("low sample"),
        },
    )

    n_low = int(df["low_sample"].sum())
    if n_low and not show_low:
        st.caption(f"{n_low} low-sample team(s) hidden.")


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
