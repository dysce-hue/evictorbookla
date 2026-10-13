"""Small compatibility and chart-layout helpers shared by the Streamlit UI."""
import streamlit as st
import streamlit.components.v1 as components


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
