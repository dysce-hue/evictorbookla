"""One entry point for the ALPHA map and BETA policy workspace."""
import streamlit as st

st.set_page_config(page_title="EVICTOR BOOK", layout="wide", initial_sidebar_state="expanded")

page = st.navigation(
    [
        st.Page("app.py", title="ALPHA · 3D map", icon="🗺️", default=True),
        st.Page("app_beta.py", title="BETA · Policy & forecasts", icon="📊", url_path="policy"),
        st.Page("math_page.py", title="Mathematical methods", icon="📐", url_path="methods"),
    ],
    position="top",
)
page.run()
