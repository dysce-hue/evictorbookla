"""Actual official parcel/case context, conservatively dated at acquisition."""
import json
from functools import lru_cache
from pathlib import Path
import pandas as pd
import streamlit as st

DIRECTORY = Path("data/property_context")

@lru_cache(maxsize=2)
def _read(stamps):
    del stamps
    return (pd.read_csv(DIRECTORY/"county_parcels_current.csv",dtype={"AIN":str}),
            pd.read_csv(DIRECTORY/"lahd_cases_current.csv",dtype={"apn":str}),
            json.loads((DIRECTORY/"provenance.json").read_text()))

def context():
    paths = [DIRECTORY/name for name in ["county_parcels_current.csv","lahd_cases_current.csv","provenance.json"]]
    return _read(tuple(p.stat().st_mtime_ns for p in paths)) if all(p.exists() for p in paths) else None

def render_context_summary():
    loaded = context()
    if loaded is None:
        st.info("Official parcel/enforcement data has not been downloaded on this host.")
        return
    parcels,cases,meta = loaded
    st.markdown("### Actual official property data acquired")
    st.write(f"{len(parcels):,} matched county parcels and {len(cases):,} LAHD enforcement-case rows; observed {meta['observed_at'][:10]}.")
    st.warning("Current-source context, not historically verified model inputs. Parcel IDs are not building IDs; reported unit fields are not verified rental-unit counts. These records are excluded from old forecast training because historical publication dates are unknown.")
    st.markdown(f"Sources: [County parcel API]({meta['parcel_source']}) · [LAHD enforcement cases](https://data.lacity.org/Community-Economic-Development/LAHD-Property-Look-Up-for-Investigation-and-Enforc/eagk-wq48)")
    st.caption("Use the forecast address inspector to see parcel-matched records. Case history describes filed enforcement cases, not necessarily confirmed violations. No ownership-change data has been obtained.")
    rent_path = DIRECTORY/"rent_provenance.json"
    if rent_path.exists():
        rent = json.loads(rent_path.read_text())
        st.write(f"Zillow rent vintage: {rent['zip_rows']} LA County ZIP rows, through {rent['latest_month']}; observed {rent['observed_at'][:10]}.")
        st.caption("Actual current asking-rent data is archived, but old release-time vintages are unavailable. It is not fed into retrospective forecasts or treated as actual rent paid.")

def render_address_context(records,address):
    loaded = context()
    if loaded is None or "Apn" not in records:
        return
    parcels,cases,meta = loaded
    with st.expander("Official parcel and enforcement context (not model inputs)"):
        apns = records.loc[records.full_address.eq(address),"Apn"].fillna("").astype(str).str.replace(r"\.0$","",regex=True).str.replace(r"\D","",regex=True)
        apns = set(apns[apns.str.len().eq(10)])
        matches = parcels[parcels.AIN.isin(apns)].copy()
        if matches.empty:
            st.info("No exact county APN match for this address. No unit count or building identity is inferred.")
            return
        matches["Reported units (up to 5 building lines)"] = matches[[f"Units{i}" for i in range(1,6)]].apply(pd.to_numeric,errors="coerce").replace(0,float("nan")).sum(axis=1,min_count=1)
        st.dataframe(matches,hide_index=True,width="stretch")
        matched_cases = cases[cases.apn.fillna("").str.replace(r"\D","",regex=True).isin(apns)].copy()
        st.write(f"{len(matched_cases):,} source enforcement-case rows matched to these parcels.")
        if not matched_cases.empty:
            st.dataframe(matched_cases.sort_values("case_filed_date",ascending=False).head(100),hide_index=True,width="stretch")
        st.caption(f"Observed {meta['observed_at'][:10]}. Exact APN matching does not verify a building/unit link. Current unit counts and closure status must not be backdated. This context does not change the displayed forecast probability.")
