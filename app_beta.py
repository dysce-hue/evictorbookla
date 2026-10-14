"""Separate policy-visualization prototype for EvictorBookLA."""

from __future__ import annotations

import base64
import json
import os
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from policy_metrics import (
    CAUSE_ORDER,
    FORECAST_FEATURES,
    FORECAST_FEATURE_LABELS,
    FORECAST_MODEL_VERSION,
    annual_cumulative,
    building_risk_portfolio,
    calendar_pattern,
    cause_district,
    compressed_pareto,
    hotspot_persistence,
    prepare_notices,
    predictive_building_forecast,
    property_concentration,
    rent_box_summary,
    repeat_intervals,
)


DATA_PATH = os.path.join("data", "cleaned_evictions_final.csv")
BLUE = "#2563eb"
GOLD = "#d69e2e"
INK = "#172033"


st.set_page_config(page_title="EVICTOR BOOK BETA", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Loading LAHD notice data…")
def load_data(path: str, modified_at: float) -> pd.DataFrame:
    del modified_at
    raw = pd.read_csv(path, low_memory=False)
    return prepare_notices(raw)


@st.cache_data(show_spinner="Training and validating the experimental forecast…")
def load_predictive_forecast(path: str, modified_at: float, as_of_date: str, model_version: str) -> dict[str, object]:
    """Rebuild the candidate forecast when the validated source file changes."""
    del modified_at, model_version
    raw = pd.read_csv(path, low_memory=False)
    prepared = prepare_notices(raw)
    return predictive_building_forecast(prepared, as_of_date=as_of_date)


def empty_state(message: str) -> None:
    st.info(message)


def alpha_address_url(address: str, start_date, end_date) -> str:
    """Open the selected address with the same period in the Alpha map."""
    view = {"address": address, "start_date": str(start_date), "end_date": str(end_date)}
    encoded = base64.b64encode(json.dumps(view).encode("utf-8")).decode("ascii")
    return f"/?view={quote(encoded)}"


def render_address_inspector(
    records: pd.DataFrame,
    portfolio: pd.DataFrame,
    addresses: list[str],
    key_prefix: str,
    heading: str = "Address inspector",
) -> None:
    """Render an address menu, profile, timeline, and source records."""
    available = portfolio[portfolio["full_address"].isin(addresses)].copy()
    if available.empty:
        empty_state("No inspectable addresses are available for this selection.")
        return
    available = available.sort_values(["risk_score", "notices"], ascending=False)
    profiles = available.set_index("full_address")
    options = available["full_address"].tolist()

    st.subheader(heading)
    selected = st.selectbox(
        "Choose an address",
        options,
        key=f"{key_prefix}_address",
        format_func=lambda address: (
            f"{address} — score {profiles.at[address, 'risk_score']:.1f} · "
            f"{int(profiles.at[address, 'notices']):,} notices · "
            f"{int(profiles.at[address, 'filing_dates']):,} filing dates"
        ),
    )
    profile = profiles.loc[selected]
    history = records[records["full_address"].eq(selected)].copy()

    metrics = st.columns(4)
    metrics[0].metric("Attention score", f"{profile['risk_score']:.1f}")
    metrics[1].metric("Priority tier", str(profile["priority_tier"]))
    metrics[2].metric(
        "Notices / filing dates",
        f"{int(profile['notices']):,} / {int(profile['filing_dates']):,}",
    )
    metrics[3].metric("Rent owed", f"${profile['total_rent_owed']:,.0f}")

    link_col, detail_col = st.columns([1, 2])
    with link_col:
        st.link_button(
            "Open address in Alpha 3D map",
            alpha_address_url(selected, start_date, end_date),
            width="stretch",
        )
    with detail_col:
        district = "Unknown" if pd.isna(profile["Council_District"]) else str(int(profile["Council_District"]))
        st.caption(
            f"Council district {district} · ZIP {profile['Zip'] or 'Unknown'} · "
            f"First notice {profile['first_notice']:%b %d, %Y} · Latest {profile['latest_notice']:%b %d, %Y}"
        )

    history["month"] = history["Date_Filed"].dt.to_period("M").dt.to_timestamp()
    timeline = (
        history.groupby(["month", "Cause_Category"], as_index=False)
        .size()
        .rename(columns={"size": "notices"})
    )
    fig = px.bar(
        timeline,
        x="month",
        y="notices",
        color="Cause_Category",
        labels={"month": "Month", "notices": "Notices", "Cause_Category": "Cause"},
        color_discrete_sequence=[BLUE, GOLD, "#64748b", "#f97316", "#7c3aed", "#0f766e", "#be123c", "#94a3b8"],
    )
    fig.update_layout(height=390, legend_orientation="h", legend_y=-0.3, margin={"l": 20, "r": 20, "t": 20, "b": 90})
    st.plotly_chart(fig, width="stretch")

    daily = (
        history.groupby(["Date_Filed", "Cause_Category"], as_index=False)
        .agg(notices=("full_address", "size"), reported_rent_owed=("Rent_Owed_num", "sum"))
        .sort_values("Date_Filed", ascending=False)
        .rename(
            columns={
                "Date_Filed": "Filing date",
                "Cause_Category": "Cause category",
                "notices": "Notices",
                "reported_rent_owed": "Reported rent owed",
            }
        )
    )
    daily["Filing date"] = daily["Filing date"].dt.date
    st.dataframe(
        daily,
        hide_index=True,
        width="stretch",
        column_config={"Reported rent owed": st.column_config.NumberColumn(format="$%.0f")},
    )


def render_predictive_inspector(records: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Show the forecast, history, and additive model contributions for one address."""
    if forecast.empty:
        return
    profiles = forecast.set_index("full_address")
    options = forecast["full_address"].head(2000).tolist()
    st.subheader("Inspect an experimental forecast")
    selected = st.selectbox(
        "Choose a forecasted address",
        options,
        key="forecast_address",
        format_func=lambda address: (
            f"{address} — {profiles.at[address, 'forecast_probability']:.1%} · "
            f"{profiles.at[address, 'forecast_tier']} · {profiles.at[address, 'momentum_label']}"
        ),
    )
    profile = profiles.loc[selected]
    metrics = st.columns([1, 1.4, 1])
    metrics[0].metric("90-day probability", f"{profile['forecast_probability']:.1%}")
    metrics[1].metric(
        "Approx. 95% CI",
        f"{100 * profile['ci_lower_95']:.1f}–{100 * profile['ci_upper_95']:.1f}%"
        if pd.notna(profile['ci_lower_95']) and pd.notna(profile['ci_upper_95']) else "Unavailable",
        help="Approximate 95% confidence interval for the estimated notice probability, conditional on the selected model.",
    )
    metrics[2].metric("Notices in 90 days", f"{int(profile['notices_90']):,}")
    st.caption(
        f"Activity momentum: {profile['momentum_label']}. "
        f"Top positive model contributions: {profile['top_drivers']}. "
        "These explain the model score, not the causes of future notices."
    )

    contribution_rows = []
    for feature in FORECAST_FEATURES:
        contribution_rows.append(
            {
                "Driver": FORECAST_FEATURE_LABELS[feature],
                "Contribution": float(profile[f"contribution_{feature}"]),
            }
        )
    contributions = pd.DataFrame(contribution_rows).sort_values("Contribution")
    driver_fig = px.bar(
        contributions,
        x="Contribution",
        y="Driver",
        orientation="h",
        color="Contribution",
        color_continuous_scale=[(0, "#2563eb"), (0.5, "#e2e8f0"), (1, "#be123c")],
        color_continuous_midpoint=0,
        labels={"Contribution": "Contribution to model log-odds"},
    )
    driver_fig.update_layout(height=430, coloraxis_showscale=False, margin={"l": 20, "r": 20, "t": 15, "b": 20})
    st.plotly_chart(driver_fig, width="stretch")

    history = records[records["full_address"].eq(selected)].copy()
    if not history.empty:
        history["month"] = history["Date_Filed"].dt.to_period("M").dt.to_timestamp()
        timeline = history.groupby("month", as_index=False).size().rename(columns={"size": "notices"})
        timeline_fig = px.bar(
            timeline,
            x="month",
            y="notices",
            labels={"month": "Month received", "notices": "Notices"},
        )
        timeline_fig.update_traces(marker_color=BLUE)
        timeline_fig.update_layout(height=320, margin={"l": 20, "r": 20, "t": 15, "b": 20})
        st.plotly_chart(timeline_fig, width="stretch")


def render_mathematical_methods() -> None:
    """Use the shared ALPHA/BETA mathematical reference."""
    from methods_ui import render_mathematical_methods as render_shared_methods
    render_shared_methods()

if not os.path.exists(DATA_PATH):
    st.error(f"LAHD notice data was not found at {DATA_PATH}.")
    st.stop()


notices = load_data(DATA_PATH, os.path.getmtime(DATA_PATH))
min_date = notices["Date_Filed"].min().date()
max_date = notices["Date_Filed"].max().date()

st.markdown(
    """
    <style>
    :root { --beta-ink: #172033; --beta-blue: #2563eb; }
    .block-container { padding-top: 2rem; max-width: 1500px; }
    h1, h2, h3 { color: var(--beta-ink); }
    [data-testid="stMetric"] { background: #f7f9fc; border: 1px solid #e4e9f2; border-radius: 12px; padding: 12px; }
    [data-testid="stMetric"] * { color: #172033 !important; }
    :focus-visible { outline: 3px solid #ffbf47 !important; outline-offset: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("EVICTOR BOOK BETA")
st.info("These are reported EVICTION NOTICES—not completed evictions, court judgments, unique tenants, or confirmed displacement. The predictor estimates another reported notice within 90 days, not whether a tenant will be evicted.")
st.caption("Policy analysis and experimental forecasts. Use the navigation above to return to the ALPHA 3D map.")

st.sidebar.header("EVICTOR BOOK BETA")
date_value = st.sidebar.date_input(
    "Notice date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_value, (tuple, list)) and len(date_value) == 2:
    start_date, end_date = date_value
elif isinstance(date_value, (tuple, list)) and len(date_value) == 1:
    start_date = end_date = date_value[0]
else:
    start_date = end_date = date_value

district_options = sorted(notices["Council_District"].dropna().astype(int).unique().tolist())
selected_districts = st.sidebar.multiselect("Council districts", district_options)
cause_options = [cause for cause in CAUSE_ORDER if cause in set(notices["Cause_Category"])]
selected_causes = st.sidebar.multiselect("Cause categories", cause_options)
address_query = st.sidebar.text_input("Address contains")

filtered = notices[
    notices["Date_Filed"].dt.date.between(start_date, end_date)
].copy()
if selected_districts:
    filtered = filtered[filtered["Council_District"].isin(selected_districts)]
if selected_causes:
    filtered = filtered[filtered["Cause_Category"].isin(selected_causes)]
if address_query.strip():
    filtered = filtered[filtered["full_address"].str.contains(address_query.strip(), case=False, na=False)]

if filtered.empty:
    st.warning("No notices match the selected filters.")
    st.stop()

latest_date = filtered["Date_Filed"].max()
latest_month_end = latest_date + pd.offsets.MonthEnd(0)
latest_month_partial = latest_date.normalize() < latest_month_end.normalize()
address_date_nodes = filtered[["full_address", "Date_Filed"]].drop_duplicates().shape[0]
address_date_counts = filtered.groupby("full_address")["Date_Filed"].nunique()
repeat_address_share = float(address_date_counts.gt(1).mean() * 100) if len(address_date_counts) else 0.0

metric_cols = st.columns(3)
metric_cols[0].metric("Notice records", f"{len(filtered):,}")
metric_cols[1].metric("Unique addresses", f"{filtered['full_address'].nunique():,}")
metric_cols[2].metric("Address-date nodes", f"{address_date_nodes:,}")
secondary_metrics = st.columns(2)
secondary_metrics[0].metric("Repeat-address share", f"{repeat_address_share:.1f}%")
secondary_metrics[1].metric("Latest notice received", latest_date.strftime("%b %d, %Y"))
if latest_month_partial:
    st.warning("The latest month may be incomplete. Treat comparisons involving it as provisional.")

view_options = [
    "Time patterns",
    "Concentration",
    "Causes & rent",
    "Repeat filings",
    "Persistent hotspots",
    "Building risk portfolio",
    "Predictive risk (experimental)",
    "Data quality",
    "Mathematical methods",
]
selected_view = st.radio("Analysis view", view_options, horizontal=True, label_visibility="collapsed")

if selected_view == "Time patterns":
    st.subheader("Year-over-year cumulative notice activity")
    cumulative = annual_cumulative(filtered)
    if cumulative.empty:
        empty_state("No dated records are available for this view.")
    else:
        fig = px.line(
            cumulative,
            x="day_of_year",
            y="cumulative_notices",
            color="year",
            custom_data=["calendar_day", "notices"],
            labels={"day_of_year": "Calendar day", "cumulative_notices": "Cumulative notices", "year": "Year"},
            color_discrete_sequence=["#94a3b8", "#64748b", GOLD, BLUE],
        )
        fig.update_traces(hovertemplate="%{customdata[0]}<br>Cumulative: %{y:,}<br>Daily: %{customdata[1]:,}<extra>%{fullData.name}</extra>")
        fig.update_layout(height=470, legend_orientation="h", legend_y=1.08, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    st.subheader("When notices are received")
    pattern = calendar_pattern(filtered)
    pivot = pattern.pivot(index="weekday", columns="month", values="notices")
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = list(pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%b"))
    pivot = pivot.reindex(index=weekdays, columns=months).fillna(0)
    heat = px.imshow(
        pivot,
        labels={"x": "Month", "y": "Day received", "color": "Notices"},
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=",.0f",
    )
    heat.update_layout(height=430)
    st.plotly_chart(heat, width="stretch")

elif selected_view == "Concentration":
    ranked, concentration = property_concentration(filtered)
    summary_cols = st.columns(3)
    summary_cols[0].metric("Notices from top 1% of addresses", f"{concentration['top_1_pct']:.1f}%")
    summary_cols[1].metric("Notices from top 5% of addresses", f"{concentration['top_5_pct']:.1f}%")
    summary_cols[2].metric("Notices from top 10% of addresses", f"{concentration['top_10_pct']:.1f}%")
    pareto = compressed_pareto(ranked)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pareto["address_percentile"],
        y=pareto["cumulative_share"],
        mode="lines",
        name="Cumulative notice share",
        line={"color": BLUE, "width": 3},
        customdata=pareto[["rank", "notices"]],
        hovertemplate="Top %{x:.2f}% of addresses<br>%{y:.1f}% of notices<br>Rank: %{customdata[0]:,}<br>Address notices: %{customdata[1]:,}<extra></extra>",
    ))
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line={"color": "#94a3b8", "dash": "dot"})
    fig.update_layout(height=500, xaxis_title="Top share of addresses", yaxis_title="Cumulative share of notices", xaxis_ticksuffix="%", yaxis_ticksuffix="%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        ranked.head(50).rename(columns={"full_address": "Address", "notices": "Notices", "rank": "Rank"})[["Rank", "Address", "Notices"]],
        hide_index=True,
        width="stretch",
    )

elif selected_view == "Causes & rent":
    st.subheader("Cause composition by council district")
    district_causes = cause_district(filtered)
    if district_causes.empty:
        empty_state("Council district values are unavailable for this selection.")
    else:
        matrix = district_causes.pivot(index="cause", columns="district", values="district_share").reindex(CAUSE_ORDER).fillna(0)
        fig = px.imshow(
            matrix,
            labels={"x": "Council district", "y": "Cause category", "color": "District share"},
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto=".1f",
        )
        fig.update_coloraxes(colorbar_ticksuffix="%")
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")
        st.caption("Each cell is the percentage of that district’s selected notices assigned to the cause category.")

    st.subheader("Reported rent owed by cause")
    rent_summary = rent_box_summary(filtered)
    if rent_summary.empty:
        empty_state("No positive reported rent-owed values are available for this selection.")
    else:
        box = go.Figure()
        for row in rent_summary.itertuples():
            box.add_trace(go.Box(
                name=row.Cause_Category,
                q1=[row.q1],
                median=[row.median],
                q3=[row.q3],
                lowerfence=[row.lower],
                upperfence=[row.upper],
                boxpoints=False,
                marker_color=BLUE,
                hovertemplate=f"{row.Cause_Category}<br>Records: {row.count:,}<br>Median: $%{{median:,.0f}}<extra></extra>",
            ))
        box.update_layout(height=560, showlegend=False, yaxis_title="Reported rent owed", yaxis_tickprefix="$", xaxis_title="")
        st.plotly_chart(box, width="stretch")
        st.caption("For readability, this comparison includes positive values at or below the selected data’s 95th percentile. Whiskers show the 5th and 95th percentiles.")

elif selected_view == "Repeat filings":
    intervals, interval_hist = repeat_intervals(filtered)
    st.subheader("Time between repeated filing dates at the same address")
    if interval_hist.empty:
        empty_state("No addresses have multiple distinct filing dates in this selection.")
    else:
        fig = px.bar(interval_hist, x="interval", y="address_intervals", labels={"interval": "Days between filing dates", "address_intervals": "Address intervals"})
        fig.update_traces(marker_color=BLUE, hovertemplate="%{x} days<br>%{y:,} address intervals<extra></extra>")
        fig.update_layout(height=470, xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

        repeat_addresses = (
            filtered.groupby("full_address")["Date_Filed"].nunique().sort_values(ascending=False)
        )
        choices = repeat_addresses[repeat_addresses.gt(1)].head(1000).index.tolist()
        selected_address = st.selectbox("Inspect an address timeline", choices)
        if selected_address:
            history = filtered[filtered["full_address"].eq(selected_address)].copy()
            history["month"] = history["Date_Filed"].dt.to_period("M").dt.to_timestamp()
            timeline = history.groupby(["month", "Cause_Category"], as_index=False).size().rename(columns={"size": "notices"})
            fig = px.bar(timeline, x="month", y="notices", color="Cause_Category", labels={"month": "Month", "notices": "Notices", "Cause_Category": "Cause"})
            fig.update_layout(height=390, legend_orientation="h", legend_y=-0.25)
            st.plotly_chart(fig, width="stretch")

elif selected_view == "Persistent hotspots":
    st.subheader("Persistent notice activity by grid cell")
    precision = st.radio("Grid size", options=[2, 3], format_func=lambda x: "About 1 km" if x == 2 else "About 100 m", horizontal=True)
    hotspots = hotspot_persistence(filtered, precision=precision)
    if hotspots.empty:
        empty_state("Mapped coordinates are unavailable for this selection.")
    else:
        fig = px.scatter_map(
            hotspots,
            lat="grid_lat",
            lon="grid_long",
            size="notices",
            color="persistence_pct",
            hover_name="addresses",
            hover_data={"active_months": True, "notices": True, "persistence_pct": ":.1f", "grid_lat": False, "grid_long": False},
            color_continuous_scale="YlOrBr",
            labels={"persistence_pct": "Months active (%)", "active_months": "Active months", "notices": "Notices", "addresses": "Unique addresses"},
            map_style="carto-positron",
            zoom=8.7,
            center={"lat": 34.05, "lon": -118.3},
            height=650,
        )
        fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, width="stretch")
        st.caption("Persistence is the percentage of months in the selected period in which a grid cell contains at least one notice. It is descriptive, not a spatial-significance test.")

        st.markdown("---")
        st.subheader("Choose a hotspot")
        selectable_hotspots = hotspots.head(1000).reset_index(drop=True)
        hotspot_index = st.selectbox(
            "Hotspot grid cell",
            selectable_hotspots.index.tolist(),
            key="hotspot_cell",
            format_func=lambda index: (
                f"{selectable_hotspots.at[index, 'grid_lat']:.{precision}f}, "
                f"{selectable_hotspots.at[index, 'grid_long']:.{precision}f} — "
                f"{int(selectable_hotspots.at[index, 'notices']):,} notices · "
                f"{int(selectable_hotspots.at[index, 'addresses']):,} addresses · "
                f"{int(selectable_hotspots.at[index, 'active_months']):,} active months"
            ),
        )
        selected_cell = selectable_hotspots.loc[hotspot_index]
        cell_records = filtered[
            filtered["lat"].round(precision).eq(selected_cell["grid_lat"])
            & filtered["long"].round(precision).eq(selected_cell["grid_long"])
        ].copy()
        portfolio = building_risk_portfolio(filtered)
        cell_profiles = portfolio[portfolio["full_address"].isin(cell_records["full_address"].unique())]
        hotspot_metrics = st.columns(4)
        hotspot_metrics[0].metric("Notices", f"{len(cell_records):,}")
        hotspot_metrics[1].metric("Addresses", f"{cell_records['full_address'].nunique():,}")
        hotspot_metrics[2].metric("Median score", f"{cell_profiles['risk_score'].median():.1f}")
        hotspot_metrics[3].metric("High-priority", f"{cell_profiles['priority_tier'].eq('High priority').sum():,}")
        render_address_inspector(
            filtered,
            portfolio,
            cell_records["full_address"].drop_duplicates().tolist(),
            key_prefix="hotspot",
            heading="Inspect addresses in this hotspot",
        )

elif selected_view == "Building risk portfolio":
    st.subheader("Baseline building risk portfolio")
    portfolio = building_risk_portfolio(filtered)
    if portfolio.empty:
        empty_state("No known addresses are available for this portfolio.")
    else:
        st.caption(
            "The attention score is relative to the current filters: 30% notice-volume percentile, "
            "25% distinct-filing-date percentile, 25% recent-notice percentile over the trailing 180 days, "
            "and 20% reported-rent-owed percentile. It is a descriptive prioritization tool, not a prediction "
            "of an eviction or a finding of wrongdoing."
        )
        high_priority = portfolio[portfolio["priority_tier"].eq("High priority")]
        high_notice_share = high_priority["notices"].sum() / portfolio["notices"].sum() * 100
        cards = st.columns(4)
        cards[0].metric("Addresses scored", f"{len(portfolio):,}")
        cards[1].metric("High-priority", f"{len(high_priority):,}")
        cards[2].metric("Notice share", f"{high_notice_share:.1f}%")
        cards[3].metric("Reference date", portfolio["reference_end"].max().strftime("%b %d, %Y"))

        control_a, control_b = st.columns([2, 1])
        tier_order = ["High priority", "Elevated", "Monitor", "Baseline"]
        with control_a:
            selected_tiers = st.multiselect("Priority tiers", tier_order, default=tier_order)
        with control_b:
            minimum_score = st.slider("Minimum attention score", 0, 100, 0)
        portfolio_view = portfolio[
            portfolio["priority_tier"].isin(selected_tiers)
            & portfolio["risk_score"].ge(minimum_score)
        ].copy()

        if portfolio_view.empty:
            empty_state("No buildings match the portfolio controls.")
        else:
            top = portfolio_view.head(30).sort_values("risk_score")
            fig = px.bar(
                top,
                x="risk_score",
                y="full_address",
                orientation="h",
                color="priority_tier",
                custom_data=["notices", "filing_dates", "recent_notices", "total_rent_owed"],
                labels={"risk_score": "Attention score", "full_address": "Address", "priority_tier": "Tier"},
                color_discrete_map={
                    "High priority": "#be123c",
                    "Elevated": "#f97316",
                    "Monitor": GOLD,
                    "Baseline": "#64748b",
                },
            )
            fig.update_traces(
                hovertemplate=(
                    "%{y}<br>Score: %{x:.1f}<br>Notices: %{customdata[0]:,}<br>"
                    "Filing dates: %{customdata[1]:,}<br>Recent notices: %{customdata[2]:,}<br>"
                    "Reported rent owed: $%{customdata[3]:,.0f}<extra></extra>"
                )
            )
            fig.update_layout(height=max(520, len(top) * 23), legend_orientation="h", legend_y=1.08)
            st.plotly_chart(fig, width="stretch")

            portfolio_table = portfolio_view.rename(
                columns={
                    "full_address": "Address",
                    "risk_score": "Attention score",
                    "priority_tier": "Priority tier",
                    "notices": "Notices",
                    "filing_dates": "Filing dates",
                    "recent_notices": "Recent notices",
                    "total_rent_owed": "Reported rent owed",
                    "latest_notice": "Latest notice",
                    "Council_District": "Council district",
                    "Zip": "ZIP",
                }
            )
            st.dataframe(
                portfolio_table[
                    [
                        "Address",
                        "Attention score",
                        "Priority tier",
                        "Notices",
                        "Filing dates",
                        "Recent notices",
                        "Reported rent owed",
                        "Latest notice",
                        "Council district",
                        "ZIP",
                    ]
                ].head(500),
                hide_index=True,
                width="stretch",
                column_config={
                    "Attention score": st.column_config.NumberColumn(format="%.1f"),
                    "Reported rent owed": st.column_config.NumberColumn(format="$%.0f"),
                    "Latest notice": st.column_config.DateColumn(format="MMM D, YYYY"),
                },
            )
            render_address_inspector(
                filtered,
                portfolio,
                portfolio_view["full_address"].tolist(),
                key_prefix="portfolio",
                heading="Inspect a portfolio address",
            )

elif selected_view == "Predictive risk (experimental)":
    from forecast_ui import render_forecast

    st.subheader("90-day notice forecast")
    forecast_result = load_predictive_forecast(
        DATA_PATH, os.path.getmtime(DATA_PATH), str(end_date), FORECAST_MODEL_VERSION
    )
    render_forecast(
        forecast_result,
        set(filtered["full_address"].dropna().unique()),
        notices[notices["Date_Filed"].dt.date.le(end_date)],
        render_predictive_inspector,
    )

elif selected_view == "Data quality":
    total = len(filtered)
    quality = pd.DataFrame(
        {
            "Measure": [
                "Missing mapped coordinates",
                "Missing council district",
                "Unknown or unspecified cause",
                "Missing or non-positive rent owed",
                "Unknown address",
            ],
            "Records": [
                int(filtered[["lat", "long"]].isna().any(axis=1).sum()),
                int(filtered["Council_District"].isna().sum()),
                int(filtered["Cause_Category"].eq("Unknown / not specified").sum()),
                int(filtered["Rent_Owed_num"].isna().sum() + filtered["Rent_Owed_num"].le(0).fillna(False).sum()),
                int(filtered["full_address"].eq("Unknown address").sum()),
            ],
        }
    )
    quality["Share"] = quality["Records"] / total * 100
    st.dataframe(quality.style.format({"Records": "{:,.0f}", "Share": "{:.1f}%"}), hide_index=True, width="stretch")
    st.markdown(
        """
        **Interpretation limits**

        - These records describe eviction notices received by LAHD, not completed court evictions or verified tenant displacement.
        - The latest month can be incomplete because of reporting lag.
        - Geographic views exclude records without usable coordinates.
        - Rent owed is reported in the source and is not independently verified.
        """
    )

elif selected_view == "Mathematical methods":
    render_mathematical_methods()

st.divider()
st.caption("Source: Los Angeles Housing Department eviction notices. Beta reads the same validated local source file as Alpha and does not duplicate it.")
