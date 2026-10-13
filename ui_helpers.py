"""Small compatibility and chart-layout helpers shared by the Streamlit UI."""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def embed_html(content, height=400, scrolling=False):
    """Use the supported HTML iframe API, retaining older-local compatibility."""
    if hasattr(st, "iframe"):
        return st.iframe(content, height=height, alt="EVICTOR BOOK interactive visualization")
    return components.html(content, height=height, scrolling=scrolling)


def readable_timeline(figure):
    figure.update_layout(
        height=540, margin=dict(l=80, r=30, t=60, b=160),
        legend=dict(orientation="h", x=0, y=-0.28, title_text="Cause category"),
        xaxis=dict(title="Date received", automargin=True),
        yaxis=dict(title="Notice records", automargin=True, tickformat=",.0f"),
    )
    return figure


def readable_neighborhoods(figure, rate=False):
    figure.update_layout(
        height=520, margin=dict(l=170, r=35, t=25, b=60),
        showlegend=False,
        xaxis=dict(title="Notices per 1,000 renter households" if rate else "Notice records",
                   automargin=True, tickformat=",.0f"),
        yaxis=dict(title=None, automargin=True, categoryorder="total ascending"),
    )
    return figure


def address_timeline(history, monthly=True):
    """Aggregate notice volume, not people, into readable calendar bars."""
    work = history.copy()
    work["period"] = (work["Date_Filed"].dt.to_period("M").dt.to_timestamp()
                      if monthly else work["Date_Filed"].dt.normalize())
    rows = work.groupby(["period", "Cause_Category"], as_index=False).agg(
        notices=("Eviction_Count", "sum"), rent_owed=("Total_Rent_Owed", "sum"))
    figure = px.bar(rows, x="period", y="notices", color="Cause_Category",
                    hover_data={"rent_owed": ":$,.0f"},
                    labels={"period": "Month received" if monthly else "Date received",
                            "notices": "Notice records", "Cause_Category": "Cause category",
                            "rent_owed": "Reported rent owed"}, title="Reported-notice timeline")
    figure.update_traces(width=(20 if monthly else 3) * 86400000)
    readable_timeline(figure)
    figure.update_xaxes(title="Month received" if monthly else "Date received")
    return figure


def routine_chart(summary, sizes, share=False, yearly=True, area="All Areas"):
    """Show independent shares; stack only additive notice counts."""
    measure = "share_percent" if share else "notices"
    figure = go.Figure()
    for cohort, color in [("Routine", "#eb4b00"), ("Infrequent", "#484b80")]:
        rows = summary[summary["cohort"].eq(cohort)]
        figure.add_trace(go.Scatter(
            x=rows["period"], y=rows[measure], mode="lines+markers",
            stackgroup=None if share else "notices",
            name=f"{cohort} · {sizes.get(cohort, 0):,} addresses",
            line={"color": color, "width": 3},
            customdata=rows[["notices", "share_percent"]].to_numpy(),
            hovertemplate="%{x|%b %Y}<br>%{customdata[0]:,.0f} notices<br>%{customdata[1]:.1f}% of period notices<extra>%{fullData.name}</extra>"))
    dates = sorted(summary["period"].unique())
    figure.update_layout(
        title=f"{'Share of' if share else 'Contribution to'} notice volume by {'year' if yearly else 'month'} — {area}",
        height=540, margin=dict(l=85, r=30, t=70, b=115),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        yaxis=dict(title="Share of notices (%)" if share else "Notice records", rangemode="tozero", automargin=True),
        xaxis=dict(title=None, tickmode="array" if yearly else "auto",
                   tickvals=dates if yearly else None,
                   ticktext=[pd.Timestamp(d).strftime("%Y") for d in dates] if yearly else None,
                   tickformat="%Y" if yearly else "%b %Y", automargin=True))
    if share:
        figure.update_yaxes(range=[0, 100], ticksuffix="%")
    return figure
