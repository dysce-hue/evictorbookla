import os
import json
import base64
import copy
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import streamlit as st
from ui_helpers import embed_html, readable_neighborhoods, address_timeline, routine_chart
import plotly.express as px
import plotly.graph_objects as go
from fred_data import fetch_rent_cpi, RENT_SERIES

from feature_engineering import (
    address_profiles,
    anomaly_table,
    correlation_table,
    forecast_monthly,
    grid_hotspots,
    property_profiles,
    report_html,
    routine_filer_summary,
    selection_geojson,
)
from data_refresh import METADATA_PATH, SOURCE_CSV, refresh as refresh_lahd_data

# ==============================
# CONFIG
# ==============================
def get_config_value(name):
    """Read configuration from Streamlit secrets, then the environment."""
    try:
        value = st.secrets.get(name)
    except Exception:
        # Streamlit raises when no secrets file exists; environment variables
        # remain available for local and non-Streamlit deployments.
        value = None
    return value or os.getenv(name, "")


MAPBOX_TOKEN = get_config_value("MAPBOX_TOKEN")
DATA_DIR = "data"
EVICTION_CSV = os.path.join(DATA_DIR, "cleaned_evictions_final.csv")
ZORI_ZIP_CSV = os.path.join(DATA_DIR, "zillow_zori_la_zip_monthly.csv")
ZCTA_GEOJSON = os.path.join(DATA_DIR, "la_zcta_2020.geojson")

# Shapefiles (same as Shiny)
SUBDIV_SHP = os.path.join(DATA_DIR, "County_Subdivisions.shp")
NBHD_SHP = os.path.join(DATA_DIR, "8494cd42-db48-4af1-a215-a2c8f61e96a22020328-1-621do0.x5yiu.shp")

# Optional pre-aggregated census file
NEIGHBORHOOD_CENSUS_CSV = os.path.join(DATA_DIR, "neighborhood_census.csv")

# FRED CPI config
FRED_API_KEY = get_config_value("FRED_API_KEY")
FRED_SERIES_ID = RENT_SERIES  # US city average rent of primary residence

CAUSE_CATEGORY_ORDER = [
    "Non-payment of rent",
    "Lease violation",
    "Nuisance / disturbance",
    "Owner/family occupancy",
    "Withdrawal / demolition",
    "Multiple stated causes",
    "Other single cause",
    "Unknown / not specified",
]


def normalize_zip(values):
    """Return five-digit ZIP labels while preserving missing values as blanks."""
    normalized = values.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return normalized.where(normalized.str.fullmatch(r"\d{5}"), "")


def split_cause_terms(value):
    """Return unique LAHD cause terms without repeated combination wording."""
    if not isinstance(value, str):
        return []
    terms = []
    for part in value.split(","):
        term = " ".join(part.split()).strip()
        if term and term.lower() not in {"nan", "none"} and term not in terms:
            terms.append(term)
    return terms


def cause_category(value):
    """Map each address-date node to one mutually exclusive chart category."""
    terms = split_cause_terms(value)
    meaningful = [term for term in terms if term.lower() != "unknown"]
    if not meaningful:
        return "Unknown / not specified"
    if len(meaningful) > 1:
        return "Multiple stated causes"

    term = meaningful[0].lower()
    if "non-payment of rent" in term:
        return "Non-payment of rent"
    if "violation of the rental agreement" in term:
        return "Lease violation"
    if "disorderly behavior" in term or "nuisance" in term:
        return "Nuisance / disturbance"
    if "owner/family occupancy" in term:
        return "Owner/family occupancy"
    if "withdrawing units" in term or "demolition" in term:
        return "Withdrawal / demolition"
    return "Other single cause"

# ==============================
# DATA LOADING — EVICTIONS
# ==============================
@st.cache_data(persist="disk", max_entries=3, show_spinner="Preparing mapped notice data…")
def load_evictions(path, modified_at=None):
    if not os.path.exists(path):
        st.error(f"Eviction CSV not found at: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"Zip": str, "Apn": str}, low_memory=False)

    expected_cols = [
        "Council District", "Rent Owed", "Date Received",
        "full_address", "long", "lat", "Cause", "Apn"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns in CSV: {missing}")

    df = df.copy()

    # Preserve optional source fields so advanced filters appear automatically
    # when refreshed datasets contain them.
    for col in ["Zip", "Bedroom", "Notice Type", "Eviction Category", "Owner", "Owner Name", "Unit Count"]:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce stringish columns & handle NAs
    for col in ["Council District", "full_address", "Cause", "Apn"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # LAHD's CSV sometimes parses district numbers as floats (for example,
    # "3.0"). Store a canonical integer-looking label for filtering, popups,
    # and exports while retaining non-numeric source values if they occur.
    district_numeric = pd.to_numeric(df["Council District"], errors="coerce")
    df.loc[district_numeric.notna(), "Council District"] = (
        district_numeric[district_numeric.notna()].astype(int).astype(str)
    )
    df.loc[district_numeric.eq(0), "Council District"] = ""

    # Clean Rent Owed numeric
    if "Rent Owed" in df.columns:
        df["Rent_Owed_num"] = (
            df["Rent Owed"]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    else:
        df["Rent_Owed_num"] = np.nan

    # Date received → Date_Filed
    if "Date Received" in df.columns:
        df["Date_Filed"] = pd.to_datetime(
            df["Date Received"], errors="coerce", format="mixed"
        )
    else:
        df["Date_Filed"] = pd.NaT

    # Drop rows without coordinates or valid date
    df = df.dropna(subset=["long", "lat", "Date_Filed"])

    # Aggregate like Shiny: by address + filing date
    group_cols = ["full_address", "Date_Filed"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            Eviction_Count=("Apn", "size"),
            Council_District=("Council District", "first"),
            Total_Rent_Owed=("Rent_Owed_num", "sum"),
            lat=("lat", "first"),
            long=("long", "first"),
            Cause=(
                "Cause",
                lambda x: ", ".join(
                    sorted(
                        set(
                            [c for c in x if isinstance(c, str) and c.strip() != ""]
                        )
                    )
                )
            ),
            Apn=(
                "Apn",
                lambda x: ", ".join(
                    sorted(
                        set(
                            [a for a in x if isinstance(a, str) and a.strip() != ""]
                        )
                    )
                )
            ),
            Zip=("Zip", "first"),
            Bedroom=("Bedroom", "first"),
            Notice_Type=("Notice Type", "first"),
            Eviction_Category=("Eviction Category", "first"),
            Owner=("Owner", "first"),
            Owner_Name=("Owner Name", "first"),
            Unit_Count=("Unit Count", "first"),
        )
    )

    # Repeat count per address
    rpt = (
        agg.groupby("full_address")["Date_Filed"]
        .size()
        .reset_index(name="repeat_count")
    )
    agg = agg.merge(rpt, on="full_address", how="left")

    # Replace NaN with safe values for JSON / plotting
    agg["Total_Rent_Owed"] = agg["Total_Rent_Owed"].fillna(0.0)
    agg["Eviction_Count"] = agg["Eviction_Count"].fillna(0).astype(int)
    agg["repeat_count"] = agg["repeat_count"].fillna(0).astype(int)
    agg["Cause"] = agg["Cause"].fillna("")
    agg["Cause_Category"] = agg["Cause"].map(cause_category)
    agg["Apn"] = agg["Apn"].fillna("")
    agg["Zip"] = normalize_zip(agg["Zip"])
    for col in ["Notice_Type", "Eviction_Category", "Owner", "Owner_Name"]:
        agg[col] = agg[col].fillna("").astype(str)

    return agg


@st.cache_data(persist="disk", max_entries=3, show_spinner="Loading Zillow ZIP rent data…")
def load_zori_zip(path, modified_at):
    """Load the supplied Zillow ZORI series at ZIP-by-month grain."""
    del modified_at
    if not os.path.exists(path):
        return pd.DataFrame(), []

    zori = pd.read_csv(path, dtype={"RegionName": str}, low_memory=False)
    required = {"RegionName", "RegionType"}
    if not required.issubset(zori.columns):
        return pd.DataFrame(), []

    month_columns = sorted(
        column
        for column in zori.columns
        if pd.notna(pd.to_datetime(column, format="%Y-%m-%d", errors="coerce"))
    )
    if not month_columns:
        return pd.DataFrame(), []

    zori = zori[zori["RegionType"].eq("zip")].copy()
    zori["ZIP"] = normalize_zip(zori["RegionName"])
    zori = zori[zori["ZIP"].ne("")].drop_duplicates("ZIP").set_index("ZIP")
    zori[month_columns] = zori[month_columns].apply(pd.to_numeric, errors="coerce")
    return zori, month_columns


@st.cache_data(persist="disk", max_entries=3, show_spinner=False)
def load_zip_boundaries(path, modified_at):
    """Load generalized Census ZCTA polygons used for the ZIP rent overlay."""
    del modified_at
    if not os.path.exists(path):
        return {"type": "FeatureCollection", "features": []}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"type": "FeatureCollection", "features": []}
    if payload.get("type") != "FeatureCollection":
        return {"type": "FeatureCollection", "features": []}
    return payload


@st.cache_data(persist="disk", max_entries=3)
def load_rent_records(path, modified_at=None):
    """Load notice-level rent amounts for a no-bucket distribution view."""
    needed = {"full_address", "Date Received", "Rent Owed", "Zip"}
    records = pd.read_csv(
        path,
        usecols=lambda column: column in needed,
        low_memory=False,
    )
    if not needed.issubset(records.columns):
        return pd.DataFrame(columns=["full_address", "Date_Filed", "Rent_Owed", "Zip"])

    records["Date_Filed"] = pd.to_datetime(
        records["Date Received"], errors="coerce", format="%m-%d-%Y"
    )
    records["Rent_Owed"] = pd.to_numeric(
        records["Rent Owed"]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )
    records["Zip"] = normalize_zip(records["Zip"])
    return records[["full_address", "Date_Filed", "Rent_Owed", "Zip"]].dropna(
        subset=["full_address", "Date_Filed", "Rent_Owed"]
    )

# ==============================
# DATA LOADING — SUBDIV / NEIGHBORHOODS
# ==============================
@st.cache_data
def load_subdivisions():
    """Load County_Subdivisions.shp and ensure EPSG:4326, with Subdiv_Name."""
    if not os.path.exists(SUBDIV_SHP):
        st.warning("Subdivision shapefile not found.")
        return gpd.GeoDataFrame()

    gdf = gpd.read_file(SUBDIV_SHP)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    if "Subdiv_Name" not in gdf.columns:
        if "NAME" in gdf.columns:
            gdf = gdf.rename(columns={"NAME": "Subdiv_Name"})
        else:
            gdf["Subdiv_Name"] = gdf.index.astype(str)

    gdf["Subdiv_Name"] = gdf["Subdiv_Name"].astype(str)
    return gdf

@st.cache_data
def load_neighborhoods():
    """Load LA neighborhood shapefile, transform to EPSG:4326, with 'name' field."""
    if not os.path.exists(NBHD_SHP):
        st.warning("Neighborhood shapefile not found.")
        return gpd.GeoDataFrame()

    gdf = gpd.read_file(NBHD_SHP)

    # Match Shiny: original often 3310, then transform
    if gdf.crs is None:
        gdf.set_crs(epsg=3310, inplace=True)
    gdf = gdf.to_crs(epsg=4326)

    if "name" not in gdf.columns:
        if "NAME" in gdf.columns:
            gdf = gdf.rename(columns={"NAME": "name"})
        else:
            gdf["name"] = "nbhd_" + gdf.index.astype(str)

    gdf["name"] = gdf["name"].astype(str)
    return gdf

@st.cache_data
def compute_subdiv_minus_neighborhood():
    """Rough analog of Shiny's subdiv_minus_neighborhood."""
    subdiv = load_subdivisions()
    nbhd = load_neighborhoods()

    if subdiv.empty:
        return subdiv
    if nbhd.empty:
        return subdiv

    try:
        nbhd_union = nbhd.geometry.union_all()
        nbhd_union_gdf = gpd.GeoDataFrame(geometry=[nbhd_union], crs=subdiv.crs)
        diff = gpd.overlay(subdiv, nbhd_union_gdf, how="difference")
        if "Subdiv_Name" not in diff.columns and "Subdiv_Name" in subdiv.columns:
            diff = gpd.sjoin(
                diff,
                subdiv[["Subdiv_Name", "geometry"]],
                how="left",
                predicate="intersects",
            )
        return diff
    except Exception as e:
        st.warning(f"Could not compute subdiv minus neighborhoods: {e}")
        return subdiv

def filter_by_area(ev_df, area_choice, subdiv, nbhd, subdiv_minus):
    """Filter eviction DataFrame by selected neighborhood or subdivision."""
    if ev_df.empty:
        return ev_df

    if (
        area_choice is None
        or area_choice == ""
        or area_choice.startswith("All Areas")
    ):
        return ev_df

    ev_gdf = gpd.GeoDataFrame(
        ev_df.copy(),
        geometry=gpd.points_from_xy(ev_df["long"], ev_df["lat"]),
        crs="EPSG:4326",
    )

    # Neighborhood match
    if area_choice in nbhd["name"].values:
        poly = nbhd[nbhd["name"] == area_choice]
        joined = gpd.sjoin(
            ev_gdf, poly[["name", "geometry"]], how="inner", predicate="within"
        )
        return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))

    # Subdivision (excl neighborhoods) match
    prefix = "Subdiv (excl. neighborhoods): "
    if area_choice.startswith(prefix):
        # Polygon subtraction is unnecessary for the default map.
        subdiv_minus = compute_subdiv_minus_neighborhood()
        subdiv_name = area_choice.replace(prefix, "")
        poly = subdiv_minus[subdiv_minus["Subdiv_Name"] == subdiv_name]
        if poly.empty:
            poly = load_subdivisions()
            poly = poly[poly["Subdiv_Name"] == subdiv_name]
        if poly.empty:
            return ev_df
        joined = gpd.sjoin(
            ev_gdf, poly[["Subdiv_Name", "geometry"]], how="inner", predicate="within"
        )
        return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))

    return ev_df

# ==============================
# DATA LOADING — CPI (FRED)
# ==============================
@st.cache_data(ttl=3600)
def load_cpi(api_key=""):
    return fetch_rent_cpi(api_key)

# ==============================
# DATA LOADING — CENSUS
# ==============================
@st.cache_data
def load_neighborhood_census():
    """
    Optional: load pre-aggregated census metrics for neighborhoods.
    """
    if not os.path.exists(NEIGHBORHOOD_CENSUS_CSV):
        return pd.DataFrame()

    df = pd.read_csv(NEIGHBORHOOD_CENSUS_CSV)
    if "name" not in df.columns:
        st.warning("neighborhood_census.csv must have a 'name' column.")
        return pd.DataFrame()
    return df

# ==============================
# STREAMLIT APP LAYOUT
# ==============================
st.set_page_config(
    page_title="EVICTOR BOOK ALPHA",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("EVICTOR BOOK ALPHA")
st.caption("Los Angeles reported eviction notices · 3D map, rent comparisons, address profiles and research tools")

st.markdown(
    """
    <style>
    .creator-badge {
        position: fixed;
        left: 12px;
        bottom: 12px;
        z-index: 9999;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.9);  /* tweak if you want darker */
        color: #0f172a;                         /* text color */
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    :focus-visible { outline: 3px solid #ffbf47 !important; outline-offset: 2px; }
    @media (max-width: 760px) {
        .creator-badge { display: none; }
        h1 { font-size: 1.8rem !important; }
        [data-testid="stMetric"] { min-width: 150px; }
    }
    @media (prefers-reduced-motion: reduce) {
        * { scroll-behavior: auto !important; transition: none !important; }
    }
    </style>

    <div class="creator-badge">
        Created by <b>Dylan Guthrie</b>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Make the sidebar header (EvictorBookLA) larger and bolder */
    section[data-testid="stSidebar"] h2 {
        font-size: 2.0rem !important;   /* tweak this value */
        font-weight: 800 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def _auto_refresh_if_due():
    """Refresh from LAHD at most once per configured interval."""
    try:
        interval_hours = float(os.getenv("LAHD_AUTO_REFRESH_HOURS", "24"))
    except ValueError:
        interval_hours = 24
    if interval_hours <= 0:
        return None
    refreshed_at = None
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text())
            refreshed_at = pd.to_datetime(metadata.get("checked_at") or metadata.get("refreshed_at"), utc=True)
        except Exception:
            refreshed_at = None
    now = pd.Timestamp.now(tz="UTC")
    if refreshed_at is None or (now - refreshed_at).total_seconds() >= interval_hours * 3600:
        with st.spinner("Checking LAHD for updated eviction notices…"):
            result = refresh_lahd_data(Path(EVICTION_CSV), dry_run=False, geocode=False)
            if result.get("status") == "updated":
                st.cache_data.clear()
            return result
    return None


@st.fragment(run_every="1h")
def refresh_status():
    # Never put a potentially 180-second source download on the first render.
    now = time.monotonic()
    if "_lahd_next_check" not in st.session_state:
        st.session_state["_lahd_next_check"] = now + 3600
    if now < st.session_state["_lahd_next_check"]:
        st.caption("Using the saved LAHD dataset. Automatic update checks begin after one hour; manual refresh is available in Data admin.")
        return
    st.session_state["_lahd_next_check"] = now + 3600
    try:
        if _auto_refresh_if_due() is not None:
            st.rerun()
    except Exception as refresh_error:
        st.warning(f"Automatic LAHD refresh failed; using the last validated local dataset. {refresh_error}")
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())
        if metadata.get("status") == "retained_newer_local_data":
            st.info(metadata.get("message"))


refresh_status()

evictions = load_evictions(EVICTION_CSV, os.path.getmtime(EVICTION_CSV))
if not evictions.empty:
    st.caption(f"Loaded mapped notice history: {evictions['Date_Filed'].min():%b %d, %Y} through {evictions['Date_Filed'].max():%b %d, %Y}. These are observed data dates, not a claim that today's reporting is complete.")
if os.path.exists(ZORI_ZIP_CSV):
    zori_zip, zori_month_columns = load_zori_zip(
        ZORI_ZIP_CSV,
        os.path.getmtime(ZORI_ZIP_CSV),
    )
else:
    zori_zip, zori_month_columns = pd.DataFrame(), []
if os.path.exists(ZCTA_GEOJSON):
    zip_boundaries = load_zip_boundaries(
        ZCTA_GEOJSON,
        os.path.getmtime(ZCTA_GEOJSON),
    )
else:
    zip_boundaries = {"type": "FeatureCollection", "features": []}
subdiv = load_subdivisions()
nbhd = load_neighborhoods()
# Calculate exclusion geometry only when that area filter is selected.
subdiv_minus = subdiv
if st.sidebar.checkbox("Load optional FRED rent index", value=False,
                       help="External data download; leave off for faster startup."):
    cpi_df, cpi_source, cpi_notes = load_cpi(FRED_API_KEY)
else:
    cpi_df = pd.DataFrame(columns=["date", "value"])
    cpi_source = "Not requested"
    cpi_notes = ["Enable ‘Load optional FRED rent index’ in the sidebar to load the national CPI comparison."]
census_df = load_neighborhood_census()

# ==============================
# BUILD NEIGHBORHOOD CHOROPLETH GEOJSON
# ==============================
nbhd_choro_json = "null"
nbhd_demo = gpd.GeoDataFrame()

if not nbhd.empty and not census_df.empty:
    # Join neighborhood geometries with census metrics
    nbhd_demo = nbhd.merge(census_df, on="name", how="left")

    # Optional: CPI-adjusted median income (rough “real income index”)
    if not cpi_df.empty and "median_income" in nbhd_demo.columns:
        latest_cpi = cpi_df["value"].iloc[-1]
        if pd.notna(latest_cpi) and latest_cpi != 0:
            nbhd_demo["median_income_cpi_adj"] = nbhd_demo["median_income"] / latest_cpi

    # To be safe, fill NaNs for numeric choropleth fields
    for col in ["median_income", "median_income_cpi_adj", "rent_burden_pct"]:
        if col in nbhd_demo.columns:
            nbhd_demo[col] = pd.to_numeric(nbhd_demo[col], errors="coerce").fillna(0.0)

    # --- NEW: total evictions per neighborhood (using full eviction dataset) ---
    try:
        if not evictions.empty:
            # Eviction points as GeoDataFrame
            ev_gdf = gpd.GeoDataFrame(
                evictions.copy(),
                geometry=gpd.points_from_xy(evictions["long"], evictions["lat"]),
                crs="EPSG:4326",
            )

            # Neighborhoods are already EPSG:4326 from load_neighborhoods()
            nbhd_4326 = nbhd.to_crs(epsg=4326)

            joined = gpd.sjoin(
                ev_gdf,
                nbhd_4326[["name", "geometry"]],
                how="inner",
                predicate="within",
            )

            # Sum Eviction_Count per neighborhood
            ev_agg = (
                joined.groupby("name", as_index=False)["Eviction_Count"]
                .sum()
                .rename(columns={"Eviction_Count": "evictions_total"})
            )

            # Attach to nbhd_demo (so features have both ACS + evictions_total)
            nbhd_demo = nbhd_demo.merge(ev_agg, on="name", how="left")
            nbhd_demo["evictions_total"] = (
                nbhd_demo["evictions_total"].fillna(0).astype(float)
            )
    except Exception as e:
        st.warning(f"Could not compute total evictions per neighborhood: {e}")

    # Convert to GeoJSON string for Mapbox
    nbhd_choro_json = nbhd_demo.to_json()

# ------------------------------
# INTERACTIVE CHART FILTER STATE
# ------------------------------
if "trend_filter" not in st.session_state:
    st.session_state["trend_filter"] = None  # (start_ts, end_ts) for a month
if "rent_filter" not in st.session_state:
    st.session_state["rent_filter"] = None   # (min_rent, max_rent) for histogram selection

def clear_chart_filters():
    st.session_state["trend_filter"] = None
    st.session_state["rent_filter"] = None

st.caption(
    "Data collected from the Los Angeles Housing Department Eviction notices filed from the City of Los Angeles. Retrieved from https://housing.lacity.gov/residents/renters/eviction-notices-filed. "

    "Scroll, pan, and zoom. Click points to select data nodes and draw polygons to select multiple data nodes and place them in an array. "
    "The following map is still a work in progress. Future Content: a methodology tab, more inferential statistics, upgraded UI, more up-to-date eviction data, and further information on corporate ownership of units."
    "Note: there is still a large margin of error in some statistics, which should not be taken wholeheartedly prior to the creation of the methodology tab. AI was used for assistance in coding."

)

if evictions.empty:
    st.stop()

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.image(
    "la_logo.png",
    width=30,
)
st.sidebar.header("EVICTOR BOOK ALPHA")

shared_view = {}
try:
    encoded_view = st.query_params.get("view")
    if encoded_view:
        shared_view = json.loads(base64.b64decode(encoded_view).decode("utf-8"))
except Exception:
    shared_view = {}

min_date = evictions["Date_Filed"].min()
max_date = evictions["Date_Filed"].max()

date_range = st.sidebar.date_input(
    "Eviction Date Range",
    value=(
        pd.to_datetime(shared_view.get("start_date", min_date)).date(),
        pd.to_datetime(shared_view.get("end_date", max_date)).date(),
    ),
    min_value=min_date,
    max_value=max_date,
)

# During range selection Streamlit briefly returns a one-item tuple after the
# first click. Treat that as a valid single-day range until the second date is
# chosen instead of trying to unpack two values.
selected_dates = list(date_range) if isinstance(date_range, (tuple, list)) else [date_range]
if len(selected_dates) >= 2:
    start_date, end_date = selected_dates[0], selected_dates[1]
elif len(selected_dates) == 1 and selected_dates[0] is not None:
    start_date = end_date = selected_dates[0]
else:
    start_date, end_date = min_date.date(), max_date.date()

min_evict_count = st.sidebar.number_input(
    "Minimum Evictions at Address",
    min_value=1,
    max_value=200,
    value=1,
    step=1,
)

# Area dropdown (neighborhoods + subdivs)
area_choices = ["All Areas"]
if not nbhd.empty:
    area_choices += sorted(nbhd["name"].unique().tolist())
if not subdiv_minus.empty and "Subdiv_Name" in subdiv_minus.columns:
    area_choices += [
        f"Subdiv (excl. neighborhoods): {n}"
        for n in sorted(subdiv_minus["Subdiv_Name"].dropna().unique().tolist())
    ]

selected_area = st.sidebar.selectbox(
    "Filter by Area (Neighborhood / Subdivision)",
    options=area_choices,
    index=area_choices.index(shared_view.get("area")) if shared_view.get("area") in area_choices else 0,
)

zip_values = normalize_zip(evictions["Zip"])
zip_options = sorted(zip_values[zip_values.ne("")].unique())
selected_zori_month = None
zori_by_zip = pd.Series(dtype="float64")

with st.sidebar.expander("ZIP and Zillow rent overlay", expanded=True):
    if zori_month_columns and not zori_zip.empty:
        available_months = [
            month
            for month in zori_month_columns
            if zori_zip.reindex(zip_options)[month].notna().any()
        ]
        requested_month = shared_view.get("zori_month")
        default_month = requested_month if requested_month in available_months else available_months[-1]
        selected_zori_month = st.selectbox(
            "ZORI month",
            available_months,
            index=available_months.index(default_month),
            format_func=lambda month: pd.to_datetime(month).strftime("%B %Y"),
            help="Zillow Observed Rent Index (ZORI) is a smoothed estimate of typical market asking rent.",
        )
        zori_by_zip = zori_zip[selected_zori_month].reindex(zip_options)
        covered_zori = zori_by_zip.dropna()
        st.caption(
            f"{covered_zori.size} of {len(zip_options)} mapped ZIP codes have Zillow data for "
            f"{pd.to_datetime(selected_zori_month).strftime('%B %Y')}."
        )
        st.caption(
            "The month controls the ZIP overlay. ZORI is a market asking-rent index, "
            "not the rent paid by a particular tenant."
        )
    else:
        st.info("Zillow ZIP rent data is unavailable; ZIP filtering still works.")

    selected_zips = st.multiselect(
        "ZIP codes",
        zip_options,
        default=[x for x in shared_view.get("zips", []) if x in zip_options],
        format_func=lambda zip_code: (
            f"{zip_code} — ${zori_by_zip.get(zip_code):,.0f} ZORI"
            if zip_code in zori_by_zip.index and pd.notna(zori_by_zip.get(zip_code))
            else f"{zip_code} — no ZORI data"
        ),
    )

with st.sidebar.expander("Advanced filters", expanded=False):
    available_cause_categories = set(evictions["Cause_Category"].dropna())
    cause_options = [category for category in CAUSE_CATEGORY_ORDER if category in available_cause_categories]
    selected_causes = st.multiselect(
        "Cause category",
        cause_options,
        default=[x for x in shared_view.get("causes", []) if x in cause_options],
    )
    district_values = evictions["Council_District"].dropna().astype(str).str.strip()
    available_districts = set(district_values)
    district_options = [str(number) for number in range(1, 16) if str(number) in available_districts]
    selected_districts = st.multiselect("Council district", district_options, default=[x for x in shared_view.get("districts", []) if x in district_options])
    apn_query = st.text_input("APN", value=shared_view.get("apn", ""), placeholder="e.g. 6132001016")
    address_query = st.text_input("Address contains", value=shared_view.get("address", ""))
    repeat_min = st.number_input("Minimum repeat filings", 1, 500, int(shared_view.get("minimum_repeat_filings", 1)))
    positive_rent = evictions.loc[evictions["Total_Rent_Owed"] > 0, "Total_Rent_Owed"]
    rent_cap = int(max(1, positive_rent.max())) if not positive_rent.empty else 1
    rent_range = st.slider(
        "Reported rent owed",
        0,
        rent_cap,
        (0, rent_cap),
        format="$%d",
    )

st.sidebar.caption("Counts represent notices, not confirmed court-ordered removals.")

st.sidebar.button("Clear Chart-Based Filters", on_click=clear_chart_filters)

st.sidebar.markdown(
    """
    <style>
    /* Make sidebar container a positioning context */
    [data-testid="stSidebar"] > div:first-child {
        position: relative;
    }

    .sidebar-creator-badge {
        position: absolute;
        left: 5px;
        bottom: -80px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.9);
        color: #0f172a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12);
        z-index: 999;
        white-space: nowrap;
    }
    </style>

    <div class="sidebar-creator-badge">
        Created by <b>Dylan Guthrie</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
# FILTER DATA IN PYTHON
# ==============================
mask_date = (evictions["Date_Filed"] >= pd.to_datetime(start_date)) & (
    evictions["Date_Filed"] <= pd.to_datetime(end_date)
)
mask_count = evictions["Eviction_Count"] >= min_evict_count

ev_filtered = evictions.loc[mask_date & mask_count].copy()
if selected_zori_month is not None:
    ev_filtered["ZORI_Rent"] = ev_filtered["Zip"].map(zori_by_zip)
else:
    ev_filtered["ZORI_Rent"] = np.nan
ev_filtered = filter_by_area(ev_filtered, selected_area, subdiv, nbhd, subdiv_minus)

if selected_causes:
    ev_filtered = ev_filtered[ev_filtered["Cause_Category"].isin(selected_causes)]
if selected_districts:
    ev_filtered = ev_filtered[ev_filtered["Council_District"].astype(str).isin(selected_districts)]
if selected_zips:
    ev_filtered = ev_filtered[ev_filtered["Zip"].isin(selected_zips)]
if apn_query.strip():
    ev_filtered = ev_filtered[ev_filtered["Apn"].str.contains(apn_query.strip(), case=False, regex=False)]
if address_query.strip():
    ev_filtered = ev_filtered[ev_filtered["full_address"].str.contains(address_query.strip(), case=False, regex=False)]
ev_filtered = ev_filtered[ev_filtered["repeat_count"] >= repeat_min]
ev_filtered = ev_filtered[
    ev_filtered["Total_Rent_Owed"].between(rent_range[0], rent_range[1])
    | ((rent_range[0] == 0) & (ev_filtered["Total_Rent_Owed"] == 0))
]

# ---- Extra interactive filters from charts ----
trend_filter = st.session_state.get("trend_filter")
if trend_filter is not None:
    t_start, t_end = trend_filter
    ev_filtered = ev_filtered[
        (ev_filtered["Date_Filed"] >= t_start)
        & (ev_filtered["Date_Filed"] < t_end)
    ]

rent_filter = st.session_state.get("rent_filter")
if rent_filter is not None:
    r_min, r_max = rent_filter
    if "Total_Rent_Owed" in ev_filtered.columns:
        ev_filtered = ev_filtered[
            (ev_filtered["Total_Rent_Owed"] >= r_min)
            & (ev_filtered["Total_Rent_Owed"] <= r_max)
        ]

if ev_filtered.empty:
    st.warning("No evictions in this date range / filter / area.")
    st.stop()

# Stable address ID for JS ↔ Python linkage. All filing-date rows at the
# same address share an ID, so polygon selections still return every row.
ev_filtered = ev_filtered.reset_index(drop=True).copy()
ev_filtered["map_id"] = pd.factorize(ev_filtered["full_address"], sort=True)[0]

# ==============================
# READ SELECTION FROM URL (?sel=...)
# ==============================
try:
    if hasattr(st, "query_params"):  # Streamlit ≥ 1.30
        qp = st.query_params
        sel_param = qp.get("sel", None)
    else:
        qp = st.experimental_get_query_params()
        sel_param_list = qp.get("sel", [None])
        sel_param = sel_param_list[0] if sel_param_list else None
except Exception:
    sel_param = None

selected_ids = []
if sel_param:
    if isinstance(sel_param, list):
        sel_str = sel_param[0]
    else:
        sel_str = sel_param
    try:
        selected_ids = [int(x) for x in sel_str.split(",") if x.strip() != ""]
    except ValueError:
        selected_ids = []

if selected_ids:
    ev_selected = ev_filtered[ev_filtered["map_id"].isin(selected_ids)].copy()
else:
    ev_selected = pd.DataFrame(columns=ev_filtered.columns)

# ==============================
# PREP DATA FOR JS
# ==============================
ev_js = ev_filtered.copy()
ev_js["Date_Filed_str"] = ev_js["Date_Filed"].dt.strftime("%m-%d-%Y")

# Render one point per address instead of one point per address-date. This
# cuts the default map workload by roughly two thirds while keeping all rows
# available to the charts, downloads, and polygon-selected Python results.
monthly_address = (
    ev_js.assign(month_key=ev_js["Date_Filed"].dt.strftime("%Y-%m"))
    .groupby(["map_id", "month_key"], as_index=False)["Eviction_Count"]
    .sum()
)
monthly_address["month_part"] = (
    monthly_address["month_key"] + ":" + monthly_address["Eviction_Count"].astype(int).astype(str)
)
monthly_lookup = monthly_address.groupby("map_id")["month_part"].agg(",".join)

map_data = (
    ev_js.groupby(["map_id", "full_address"], as_index=False, sort=False)
    .agg(
        date_min=("Date_Filed", "min"),
        date_max=("Date_Filed", "max"),
        Eviction_Count=("Eviction_Count", "sum"),
        Cause=("Cause", lambda values: ", ".join(sorted({v for v in values if v}))),
        Council_District=("Council_District", "first"),
        Total_Rent_Owed=("Total_Rent_Owed", "sum"),
        Apn=("Apn", lambda values: ", ".join(sorted({v for v in values if v}))),
        repeat_count=("Date_Filed", "nunique"),
        Zip=("Zip", "first"),
        ZORI_Rent=("ZORI_Rent", "first"),
        lat=("lat", "first"),
        long=("long", "first"),
    )
)
date_min = map_data["date_min"].dt.strftime("%m-%d-%Y")
date_max = map_data["date_max"].dt.strftime("%m-%d-%Y")
map_data["Date_Filed_str"] = np.where(date_min.eq(date_max), date_min, date_min + " to " + date_max)
map_data["month_counts"] = map_data["map_id"].map(monthly_lookup).fillna("")

map_columns = [
    "map_id",
    "full_address",
    "Date_Filed_str",
    "Eviction_Count",
    "Cause",
    "Council_District",
    "Total_Rent_Owed",
    "Apn",
    "repeat_count",
    "Zip",
    "ZORI_Rent",
    "lat",
    "long",
    "month_counts",
]
for column in ["full_address", "Date_Filed_str", "Cause", "Council_District", "Apn", "Zip"]:
    map_data[column] = map_data[column].fillna("").astype(str)
for column in ["map_id", "Eviction_Count", "Total_Rent_Owed", "repeat_count", "lat", "long"]:
    map_data[column] = pd.to_numeric(map_data[column], errors="coerce").fillna(0)
map_data["ZORI_Rent"] = pd.to_numeric(map_data["ZORI_Rent"], errors="coerce").round(2)

# Compact row arrays avoid repeating twelve field names for every point and
# prevent a second full copy of the records in the browser.
records = (
    map_data[map_columns]
    .astype(object)
    .where(pd.notna(map_data[map_columns]), None)
    .values.tolist()
)
evictions_json = json.dumps(
    records,
    separators=(",", ":"),
    ensure_ascii=False,
).replace("</", "<\\/")

# Join the selected Zillow month to official Census ZCTA polygons. The
# geometry stays fixed while the month-specific rent values update on rerun.
zip_overlay = copy.deepcopy(zip_boundaries)
selected_zip_set = set(selected_zips)
for feature in zip_overlay.get("features", []):
    properties = feature.setdefault("properties", {})
    zip_code = str(properties.get("GEOID") or properties.get("ZCTA5") or "").zfill(5)
    properties["zip"] = zip_code
    properties["selected"] = zip_code in selected_zip_set
    zori_value = zori_by_zip.get(zip_code) if selected_zori_month is not None else np.nan
    if pd.notna(zori_value):
        properties["zori_rent"] = round(float(zori_value), 2)
    else:
        properties.pop("zori_rent", None)
zori_zip_json = json.dumps(
    zip_overlay,
    separators=(",", ":"),
    ensure_ascii=False,
).replace("</", "<\\/")

# ==============================
# EMBEDDED HTML: MAPBOX GL JS + DRAW
# ==============================
html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>EvictorBookLA Atlas</title>
<meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />

<!-- Mapbox GL core (v2.15.0 – stable with Draw and 3D buildings) -->
<link href="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css" rel="stylesheet" />
<script src="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js"></script>

<!-- Mapbox GL Draw (polygon selection) -->
<link rel="stylesheet" href="https://api.mapbox.com/mapbox-gl-js/plugins/mapbox-gl-draw/v1.5.0/mapbox-gl-draw.css" type="text/css" />
<script src="https://api.mapbox.com/mapbox-gl-js/plugins/mapbox-gl-draw/v1.5.0/mapbox-gl-draw.js"></script>

<!-- Turf.js (point-in-polygon) -->
<script src="https://cdn.jsdelivr.net/npm/@turf/turf@6/turf.min.js"></script>

<style>
  body {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  #map {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 100%;
  }
  .mapboxgl-popup {
    max-width: 300px;
    font: 12px/1.4 'Helvetica Neue', Arial, Helvetica, sans-serif;
  }
  #map-status {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1;
    color: #334155;
    background: #eef3f7;
    font-size: 14px;
  }

  /* Address search bar  */
  #search-container {
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 3;
    background: rgba(255, 255, 255, 0.96);
    border-radius: 4px;
    box-shadow: 0 0 6px rgba(0,0,0,0.25);
    padding: 4px 8px;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  #address-input {
    width: 260px;
    font-size: 12px;
    padding: 3px 5px;
    border-radius: 3px;
    border: 1px solid #bbb;
  }
  #address-search-btn {
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 3px;
    border: 1px solid #888;
    background: #f3f3f3;
    cursor: pointer;
    white-space: nowrap;
  }
  #address-search-btn:hover {
    background: #e2e2e2;
  }

  /* Choropleth overlay controls */
  #choropleth-controls {
    position: absolute;
    top: 60px;
    left: 10px;
    z-index: 3;
    background: rgba(255, 255, 255, 0.96);
    border-radius: 4px;
    box-shadow: 0 0 6px rgba(0,0,0,0.25);
    padding: 4px 8px;
    font-size: 11px;
  }
  #choropleth-select {
    margin-left: 4px;
    font-size: 11px;
    max-width: 260px;
  }
  #overlay-legend {
    display: none;
    margin-top: 6px;
    min-width: 210px;
    color: #334155;
  }
  #overlay-legend-gradient {
    height: 8px;
    margin: 3px 0 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, #eff3ff, #6baed6, #08306b);
  }
  #overlay-legend-labels {
    display: flex;
    justify-content: space-between;
  }

  /* Selection info panel */
  #selection-panel {
    position: absolute;
    bottom: 25px;
    right: 10px;
    max-width: 320px;
    max-height: 45%;
    overflow-y: auto;
    background: rgba(255, 255, 255, 0.96);
    border-radius: 6px;
    box-shadow: 0 0 8px rgba(0,0,0,0.25);
    padding: 10px 12px;
    font-size: 12px;
    z-index: 2;
  }
  #selection-panel h4 {
    margin: 0 0 4px 0;
    font-size: 13px;
  }
  #selection-panel small {
    color: #555;
  }
  #selection-list {
    margin: 6px 0 0 0;
    padding-left: 18px;
  }
  #download-selected-btn {
    margin-top: 6px;
    padding: 3px 8px;
    font-size: 11px;
    border-radius: 4px;
    border: 1px solid #ccc;
    background: #f3f3f3;
    cursor: pointer;
  }
  #download-selected-btn:hover {
    background: #e6e6e6;
  }

  /* Neighborhood hover info (bottom-left) */
  #nbhd-info-panel {
    position: absolute;
    bottom: 10px;
    left: 10px;
    max-width: 260px;
    background: rgba(255, 255, 255, 0.96);
    border-radius: 4px;
    box-shadow: 0 0 6px rgba(0,0,0,0.25);
    padding: 6px 8px;
    font-size: 11px;
    z-index: 3;
  }
  #nbhd-info-panel b {
    font-size: 12px;
  }
</style>
</head>
<body>
<div id="map"></div>
<div id="map-status">Loading eviction map…</div>

<!-- Address search UI -->
<div id="search-container">
  <input id="address-input" type="text" placeholder="Search address (from dataset)..." />
  <button id="address-search-btn" type="button">Go</button>
</div>

<!-- Choropleth overlay controls -->
<div id="choropleth-controls">
  <label for="choropleth-select">Overlay:</label>
  <select id="choropleth-select">
    <option value="zori_rent" selected>Zillow asking rent by ZIP (ZORI_MONTH_PLACEHOLDER)</option>
    <option value="none">No overlay</option>
    <option value="evictions_total">Total evictions (all years)</option>
    <option value="median_income">Median income</option>
    <option value="rent_burden_pct">Reported rent burden (definition unverified)</option>
  </select>
  <div id="overlay-legend" aria-live="polite">
    <div id="overlay-legend-title"></div>
    <div id="overlay-legend-gradient"></div>
    <div id="overlay-legend-labels"><span id="overlay-legend-min"></span><span id="overlay-legend-max"></span></div>
  </div>
  <br/>
  <label for="map-mode-select">Map:</label>
  <select id="map-mode-select">
    <option value="clusters" selected>Clusters</option>
    <option value="points">Individual points</option>
    <option value="heatmap">Density heatmap</option>
  </select>
  <br/>
  <button id="animation-btn" type="button" aria-label="Play monthly animation">▶ Animate</button>
  <select id="animation-speed" aria-label="Animation speed"><option value="1500">Slow</option><option value="900" selected>Normal</option><option value="400">Fast</option></select>
  <span id="animation-label">All dates</span>
</div>


<div id="selection-panel">
  <h4>Selection</h4>
  <small>
    Draw a polygon with the <b>polygon tool</b> (top-left) to encapture data nodes.
  </small>
  <div id="selection-summary" style="margin-top:6px;">
    No selection yet.
  </div>
  <button id="download-selected-btn" type="button">
    ⬇ Download Selected as CSV
  </button>
  <button id="download-polygons-btn" type="button">⬇ Polygons GeoJSON</button>
  <ol id="selection-list"></ol>
</div>

<!-- Neighborhood hover info panel (bottom-left) -->
<div id="nbhd-info-panel">
  <div id="nbhd-info-content">
    Hover over a shaded ZIP code to view Zillow typical asking rent.
  </div>
</div>

<script>
  window.addEventListener("error", () => {
    const status = document.getElementById("map-status");
    if (status) status.textContent = "Map initialization failed. Check blocked scripts, Mapbox configuration, or browser WebGL support. Notice analytics remain available below.";
  });
  mapboxgl.accessToken = "MAPBOX_TOKEN_PLACEHOLDER";

  // Compact eviction rows from Streamlit. Keeping these as arrays avoids
  // retaining a second full object copy of the map dataset.
  const evictionRows = EVIC_DATA_PLACEHOLDER;
  const IDX = {
    id: 0, address: 1, date: 2, count: 3, cause: 4, district: 5,
    rent: 6, apn: 7, repeat: 8, zip: 9, zori: 10, lat: 11, lng: 12, months: 13
  };
  const zoriMonthLabel = "ZORI_MONTH_PLACEHOLDER";

  // Neighborhood choropleth data (census + CPI-adjusted income, etc.)
  const nbhdChoro = NBHD_CHORO_PLACEHOLDER;
  // Census ZCTA polygons joined to the selected Zillow ZORI month.
  const zipZoriChoro = ZIP_ZORI_CHORO_PLACEHOLDER;

  // Build GeoJSON with slight jitter for identical coordinates.
const jitterTracker = {};
const evictionFeatures = evictionRows.map((row) => {
  const baseLng = Number(row[IDX.lng]);
  const baseLat = Number(row[IDX.lat]);

  // Key rounded to ~meter level to group "same" coordinates
  const key = `${baseLng.toFixed(5)}|${baseLat.toFixed(5)}`;
  const count = (jitterTracker[key] || 0);
  jitterTracker[key] = count + 1;

  // First point stays in place; others spiral out slightly
  let lng = baseLng;
  let lat = baseLat;
  if (count > 0) {
    const angle = (count * 2 * Math.PI) / 24;     // 8 petals around
    const radius = 0.00000225 * count;              // ~10–15m per step
    lng = baseLng + Math.cos(angle) * radius;
    lat = baseLat + Math.sin(angle) * radius;
  }

  return {
    "type": "Feature",
    "properties": {
      id: row[IDX.id],
      full_address: row[IDX.address],
      date_filed: row[IDX.date],
      eviction_count: row[IDX.count],
      cause: row[IDX.cause],
      council_district: row[IDX.district],
      total_rent_owed: row[IDX.rent],
      apn: row[IDX.apn],
      repeat_count: row[IDX.repeat],
      zip: row[IDX.zip],
      zori_rent: row[IDX.zori],
      month_counts: row[IDX.months]
    },
    "geometry": {
      "type": "Point",
      "coordinates": [lng, lat]
    }
  };
});

const evictionGeojson = {
  "type": "FeatureCollection",
  "features": evictionFeatures
};
// The compact transport rows are no longer needed after GeoJSON creation.
evictionRows.length = 0;


  // Center the map above Los Angeles (fixed default)
  const map = new mapboxgl.Map({
    container: "map",
    style: "mapbox://styles/mapbox/streets-v12",
    center: [-118.2437, 34.0522],  // Los Angeles (lng, lat)
    zoom: 11,
    pitch: 45,
    bearing: -17.6,
    antialias: true
  });
  map.on("error", (event) => {
    const status = document.getElementById("map-status");
    if (status) {
      status.style.display = "block";
      status.textContent = "Map could not load. Check the Mapbox public token, its allowed URLs, and network access. Notice charts remain available below.";
    }
    console.error("Mapbox map error", event.error && event.error.message);
  });
  setTimeout(() => {
    if (!map.isStyleLoaded()) {
      const status = document.getElementById("map-status");
      if (status) status.textContent = "Map is taking longer than expected. Check Mapbox token configuration or blocked map requests; analytics are available below.";
    }
  }, 20000);

  // ==========================
  // Mapbox Draw control
  // ==========================
  let draw = null;
  try {
    if (typeof MapboxDraw !== "undefined") {
      const drawStyles = [
        {id:"gl-draw-polygon-fill-inactive",type:"fill",filter:["all",["==","active","false"],["==","$type","Polygon"],["!=","mode","static"]],paint:{"fill-color":"#1E90FF","fill-outline-color":"#1E90FF","fill-opacity":0.18}},
        {id:"gl-draw-polygon-fill-active",type:"fill",filter:["all",["==","active","true"],["==","$type","Polygon"]],paint:{"fill-color":"#1E90FF","fill-outline-color":"#1E90FF","fill-opacity":0.25}},
        {id:"gl-draw-polygon-midpoint",type:"circle",filter:["all",["==","$type","Point"],["==","meta","midpoint"]],paint:{"circle-radius":3,"circle-color":"#1E90FF"}},
        {id:"gl-draw-polygon-stroke-inactive",type:"line",filter:["all",["==","active","false"],["==","$type","Polygon"],["!=","mode","static"]],layout:{"line-cap":"round","line-join":"round"},paint:{"line-color":"#1E90FF","line-width":2}},
        {id:"gl-draw-polygon-stroke-active",type:"line",filter:["all",["==","active","true"],["==","$type","Polygon"]],layout:{"line-cap":"round","line-join":"round"},paint:{"line-color":"#1E90FF","line-width":2}},
        {id:"gl-draw-line-inactive",type:"line",filter:["all",["==","active","false"],["==","$type","LineString"],["!=","mode","static"]],layout:{"line-cap":"round","line-join":"round"},paint:{"line-color":"#1E90FF","line-width":2}},
        {id:"gl-draw-line-active",type:"line",filter:["all",["==","$type","LineString"],["==","active","true"]],layout:{"line-cap":"round","line-join":"round"},paint:{"line-color":"#1E90FF","line-width":2}},
        {id:"gl-draw-polygon-and-line-vertex-stroke-inactive",type:"circle",filter:["all",["==","meta","vertex"],["==","$type","Point"],["!=","mode","static"]],paint:{"circle-radius":5,"circle-color":"#ffffff"}},
        {id:"gl-draw-polygon-and-line-vertex-inactive",type:"circle",filter:["all",["==","meta","vertex"],["==","$type","Point"],["!=","mode","static"]],paint:{"circle-radius":3,"circle-color":"#1E90FF"}}
      ];
      draw = new MapboxDraw({
        displayControlsDefault: false,
        controls: {
          polygon: true,
          trash: true
        },
        styles: drawStyles,
        defaultMode: "simple_select"
      });
      map.addControl(draw, "top-left");
    }
  } catch (e) {
    console.error("MapboxDraw failed to initialize:", e);
  }

  map.addControl(new mapboxgl.NavigationControl(), "top-right");

  const mapStatus = document.getElementById("map-status");
  const loadTimeout = window.setTimeout(() => {
    if (mapStatus) {
      mapStatus.textContent = "The basemap is taking longer than expected. Check the internet connection and reload.";
    }
  }, 15000);
  window.addEventListener("error", (event) => {
    if (mapStatus) {
      mapStatus.style.display = "flex";
      mapStatus.textContent = "Map error: " + (event.message || "unknown error");
    }
  });

  // Keep last selection in memory so we can download as CSV
  let lastSelectedFeatures = [];

  // Helper: update selection panel and keep selected features
  function updateSelectionPanel(selectedFeatures) {
    lastSelectedFeatures = selectedFeatures || [];

    const summaryDiv = document.getElementById("selection-summary");
    const listEl = document.getElementById("selection-list");
    listEl.innerHTML = "";

    if (!lastSelectedFeatures.length) {
      summaryDiv.innerHTML = "No selection yet. Draw a polygon to select nodes.";
      return;
    }

    const nodeCount = lastSelectedFeatures.length;
    const unitCount = lastSelectedFeatures.reduce(
      (sum, f) => sum + (Number(f.properties.eviction_count) || 0),
      0
    );
    const rents = lastSelectedFeatures.map(
      f => Number(f.properties.total_rent_owed) || 0
    );
    const totalRent = rents.reduce((a,b) => a + b, 0);
    const avgRent = rents.length ? totalRent / rents.length : 0;
    const maxRent = rents.length ? Math.max(...rents) : 0;

    summaryDiv.innerHTML = `
      <b>Selected nodes:</b> ${nodeCount.toLocaleString()}<br/>
      <b>Notice records:</b> ${unitCount.toLocaleString()}<br/>
      <b>Total rent owed (approx):</b> $${totalRent.toLocaleString("en-US", {
        maximumFractionDigits: 0
      })}<br/>
      <b>Average rent owed per node:</b> $${avgRent.toLocaleString("en-US", {
        maximumFractionDigits: 0
      })}<br/>
      <b>Max rent owed (node):</b> $${maxRent.toLocaleString("en-US", {
        maximumFractionDigits: 0
      })}<br/>
      <small>Showing up to 50 unique addresses below.</small>
    `;

    const uniqueAddresses = [];
    const seen = new Set();
    for (const f of lastSelectedFeatures) {
      const addr = f.properties.full_address || "(no address)";
      if (!seen.has(addr)) {
        seen.add(addr);
        uniqueAddresses.push(addr);
      }
    }

    uniqueAddresses.slice(0, 50).forEach(addr => {
      const li = document.createElement("li");
      li.textContent = addr;
      listEl.appendChild(li);
    });
  }

  // Send selected IDs back to Streamlit via URL (?sel=...)
  function sendSelectionToPython(idArray) {
    try {
      const parentLoc = window.parent.location;
      const params = new URLSearchParams(parentLoc.search);

      if (!idArray || !idArray.length) {
        params.delete("sel");
      } else {
        params.set("sel", idArray.join(","));
      }

      const newSearch = params.toString();
      const currentFull =
        parentLoc.pathname + parentLoc.search + parentLoc.hash;
      const newFull =
        parentLoc.pathname + (newSearch ? "?" + newSearch : "") + parentLoc.hash;

      if (newFull !== currentFull) {
        parentLoc.href = newFull;
      }
    } catch (err) {
      console.error("Could not update parent URL with selection:", err);
    }
  }

  // Compute selection whenever polygon is created/updated
  function handleDrawChange(e) {
    if (!draw) {
      return;
    }
    const data = draw.getAll();
    if (!data || !data.features || data.features.length === 0) {
      updateSelectionPanel([]);
      sendSelectionToPython([]);
      return;
    }

    const polygons = data.features.filter(f =>
      f.geometry && (f.geometry.type === "Polygon" || f.geometry.type === "MultiPolygon")
    );
    const selectedFeatures = evictionGeojson.features.filter(f =>
      polygons.some(polygon => turf.booleanPointInPolygon(f, polygon))
    );

    updateSelectionPanel(selectedFeatures);

    const selectedIds = selectedFeatures.map(f => f.properties.id);
    sendSelectionToPython(selectedIds);
  }

  // Download selected features as CSV
  function downloadSelectedAsCSV() {
    if (!lastSelectedFeatures.length) {
      alert("No selected evictions to download. Draw a polygon first.");
      return;
    }

    const header = [
      "full_address",
      "date_filed",
      "eviction_count",
      "cause",
      "council_district",
      "total_rent_owed",
      "apn",
      "zip",
      "zori_month",
      "zori_typical_asking_rent",
      "longitude",
      "latitude"
    ];

    const rows = lastSelectedFeatures.map(f => {
      const p = f.properties;
      const coords = f.geometry && f.geometry.coordinates
        ? f.geometry.coordinates
        : [null, null];
      return [
        p.full_address || "",
        p.date_filed || "",
        p.eviction_count || "",
        p.cause || "",
        p.council_district || "",
        p.total_rent_owed || "",
        p.apn || "",
        p.zip || "",
        zoriMonthLabel || "",
        p.zori_rent == null ? "" : p.zori_rent,
        coords[0] != null ? coords[0] : "",
        coords[1] != null ? coords[1] : ""
      ].map(v => {
        const s = String(v).replace(/"/g, '""');
        return `"${s}"`;
      }).join(",");
    });

    const csvContent = header.join(",") + "\\n" + rows.join("\\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "evictions_selected_polygon.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  document.getElementById("download-selected-btn")
    .addEventListener("click", downloadSelectedAsCSV);

  document.getElementById("download-polygons-btn").addEventListener("click", () => {
    if (!draw || !draw.getAll().features.length) {
      alert("Draw one or more polygons first.");
      return;
    }
    const blob = new Blob([JSON.stringify(draw.getAll(), null, 2)], {type: "application/geo+json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "evictorbookla_selection_polygons.geojson"; a.click();
    URL.revokeObjectURL(url);
  });

  // Address search (within eviction dataset)
  function flyToAddressFromInput() {
    const input = document.getElementById("address-input");
    const queryRaw = input.value || "";
    const query = queryRaw.trim().toLowerCase();
    if (!query) {
      return;
    }

    let match = evictionGeojson.features.find(f =>
      (f.properties.full_address || "").toLowerCase() === query
    );
    if (!match) {
      match = evictionGeojson.features.find(f =>
        (f.properties.full_address || "").toLowerCase().includes(query)
      );
    }

    if (!match) {
      alert("No matching address found in current dataset/filters.");
      return;
    }

    const coords = match.geometry.coordinates.slice();
    const props = match.properties;

    map.easeTo({
      center: coords,
      zoom: 17,
      duration: 800
    });

    const rent = Number(props.total_rent_owed || 0);
    const rentFormatted = rent.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
    const hasZori = props.zori_rent !== null && props.zori_rent !== "" && Number.isFinite(Number(props.zori_rent));
    const zoriFormatted = hasZori
      ? `$${Number(props.zori_rent).toLocaleString("en-US", {maximumFractionDigits: 0})} (${zoriMonthLabel})`
      : "N/A";

    const html = `
      <div style="font-size:12px;line-height:1.4">
        <strong>Address:</strong> ${props.full_address}<br/>
        <strong>Filing period:</strong> ${props.date_filed}<br/>
        <strong>Tenants Evicted:</strong> ${props.eviction_count}<br/>
        <strong>Cause:</strong> ${props.cause || "N/A"}<br/>
        <strong>Council District:</strong> ${props.council_district || "N/A"}<br/>
        <strong>ZIP:</strong> ${props.zip || "N/A"}<br/>
        <strong>Zillow typical asking rent:</strong> ${zoriFormatted}<br/>
        <strong>Total Rent Owed:</strong> $${rentFormatted}<br/>
        <strong>APN:</strong> ${props.apn || "N/A"}
      </div>
    `;

    new mapboxgl.Popup()
      .setLngLat(coords)
      .setHTML(html)
      .addTo(map);
  }

  document.getElementById("address-search-btn")
    .addEventListener("click", flyToAddressFromInput);

  document.getElementById("address-input")
    .addEventListener("keyup", (e) => {
      if (e.key === "Enter") {
        flyToAddressFromInput();
      }
    });

  // ==========================
  // Map load + layers
  // ==========================
  map.on("load", () => {
    window.clearTimeout(loadTimeout);
    if (mapStatus) mapStatus.textContent = "Loading eviction clusters…";
    const nbhdInfoContent = document.getElementById("nbhd-info-content");

    function setNbhdInfo(html) {
      if (nbhdInfoContent) {
        nbhdInfoContent.innerHTML = html;
      }
    }

    function defaultOverlayMessage() {
      const selectedOverlay = document.getElementById("choropleth-select").value;
      return selectedOverlay === "zori_rent"
        ? "Hover over a shaded ZIP code to view Zillow typical asking rent."
        : "Hover over a shaded neighborhood to view its overlay values.";
    }

    function bindNeighborhoodHover(layerId) {
      if (!map.getLayer(layerId)) return;
      if (!nbhdChoro || !nbhdChoro.features) return;

      map.on("mousemove", layerId, (e) => {
        if (!e.features || !e.features.length) return;
        const f = e.features[0];
        const p = f.properties || {};

        const name = p.name || "Unknown neighborhood";
        const total = Number(p.evictions_total);
        const inc   = Number(p.median_income);
        const rb    = Number(p.rent_burden_pct);

        const totalStr = Number.isFinite(total)
          ? total.toLocaleString("en-US")
          : "N/A";
        const incStr = Number.isFinite(inc)
          ? "$" + inc.toLocaleString("en-US", { maximumFractionDigits: 0 })
          : "N/A";
        const rbStr = Number.isFinite(rb)
          ? rb.toFixed(1) + "%"
          : "N/A";

        setNbhdInfo(
          `<b>${name}</b><br/>` +
          `Total evictions (all years): ${totalStr}<br/>` +
          `Median income: ${incStr}<br/>` +
          `Reported rent burden (definition unverified): ${rbStr}`
        );
      });

      map.on("mouseleave", layerId, () => {
        setNbhdInfo(defaultOverlayMessage());
      });
    }

    function bindZipHover(layerId) {
      if (!map.getLayer(layerId)) return;
      map.on("mousemove", layerId, (e) => {
        if (!e.features || !e.features.length) return;
        const p = e.features[0].properties || {};
        const rent = Number(p.zori_rent);
        const rentLabel = Number.isFinite(rent)
          ? "$" + rent.toLocaleString("en-US", {maximumFractionDigits: 0})
          : "No Zillow observation";
        setNbhdInfo(
          `<b>ZIP ${p.zip || p.GEOID || "Unknown"}</b><br/>` +
          `Zillow typical asking rent: ${rentLabel}<br/>` +
          `Month: ${zoriMonthLabel}<br/>` +
          `<span style="color:#64748b">Census ZCTA boundary; not a USPS delivery boundary.</span>`
        );
      });
      map.on("mouseleave", layerId, () => setNbhdInfo(defaultOverlayMessage()));
      map.on("mouseenter", layerId, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", layerId, () => { map.getCanvas().style.cursor = ""; });
    }


    // 3D BUILDINGS LAYER (only when the selected base style exposes one)
    const layers = map.getStyle().layers || [];
    const labelLayer = layers.find(
      (layer) => layer.type === "symbol" && layer.layout && layer.layout["text-field"]
    );
    const labelLayerId = labelLayer ? labelLayer.id : undefined;
    const existingExtrusions = layers.some(layer => layer.type === "fill-extrusion");
    const buildingLayer = layers.find(layer => layer["source-layer"] === "building" && layer.source);
    if (!existingExtrusions && buildingLayer) {
      map.addLayer({
        id: "add-3d-buildings",
        source: buildingLayer.source,
        "source-layer": "building",
        type: "fill-extrusion",
        minzoom: 15,
        paint: {
          "fill-extrusion-color": "#b8b8b8",
          "fill-extrusion-height": ["coalesce", ["get", "render_height"], ["get", "height"], 0],
          "fill-extrusion-base": ["coalesce", ["get", "render_min_height"], ["get", "min_height"], 0],
          "fill-extrusion-opacity": 0.6
        }
      }, labelLayerId);
    }


    // EVICTIONS: CLUSTERED SOURCE
    map.addSource("evictions", {
      type: "geojson",
      data: evictionGeojson,
      cluster: true,
      clusterMaxZoom: 14,
      clusterRadius: 40
    });
    // Layer for clusters
    map.addLayer({
      id: "clusters",
      type: "circle",
      source: "evictions",
      filter: ["has", "point_count"],
      paint: {
        "circle-color": [
          "step",
          ["get", "point_count"],
          "#deebf7",  10,   // very light blue - few points
      "#9ecae1",  50,   // light blue
      "#6baed6",  100,  // medium blue
      "#3182bd",  200,  // darker blue
      "#08519c"
        ],
        "circle-radius": [
          "step",
          ["get", "point_count"],
          6,   10,
          9,   50,
          12,  100,
          15
        ],
        "circle-stroke-color": "#fff",
        "circle-stroke-width": 1
      }
    });

    // Count labels on clusters
    map.addLayer({
      id: "cluster-count",
      type: "symbol",
      source: "evictions",
      filter: ["has", "point_count"],
      layout: {
        "text-field": ["get", "point_count_abbreviated"],
        "text-font": ["DIN Offc Pro Medium", "Arial Unicode MS Bold"],
        "text-size": 12
      },
      paint: {
    "text-color": "#ffffff",          // white text
    "text-halo-color": "#08306b",     // dark blue halo for contrast
    "text-halo-width": 1.5,           // thicker halo
    "text-halo-blur": 0.5
  }
    });

    // Unclustered individual eviction points
    // Unclustered individual eviction points (yellow → red by Eviction_Count)
map.addLayer({
  id: "unclustered-point",
  type: "circle",
  source: "evictions",
  filter: ["!", ["has", "point_count"]],
  paint: {
    "circle-color": [
      "interpolate",
      ["linear"],
      ["get", "eviction_count"],
  1,  "#deebf7",
  3,  "#2171b5",
  5,  "#6baed6",
  10, "#fc9272",
  20, "#b10026"
],
    "circle-radius": [
      "interpolate",
      ["linear"],
      ["get", "eviction_count"],
      1,  6,
      2,  7,
      3,  8,
      5,  9,
      10, 11,
      20, 13
    ],
    "circle-stroke-width": 0.8,
    "circle-stroke-color": "#08306b"
  }
});

    // The unclustered source is expensive for a large dataset. Create it only
    // when the user explicitly asks for points or a heatmap.
    function ensureRawLayers() {
      if (!map.getSource("evictions-raw")) {
        map.addSource("evictions-raw", {type: "geojson", data: evictionGeojson});
      }
      if (!map.getLayer("all-points")) {
        map.addLayer({
          id: "all-points",
          type: "circle",
          source: "evictions-raw",
          layout: {visibility: "none"},
          paint: {
            "circle-color": ["interpolate", ["linear"], ["get", "eviction_count"], 1, "#6baed6", 10, "#fb6a4a", 25, "#99000d"],
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 8, 2, 16, 8],
            "circle-opacity": 0.75,
            "circle-stroke-color": "#fff",
            "circle-stroke-width": 0.5
          }
        });
      }
      if (!map.getLayer("eviction-heatmap")) {
        map.addLayer({
          id: "eviction-heatmap",
          type: "heatmap",
          source: "evictions-raw",
          layout: {visibility: "none"},
          paint: {
            "heatmap-weight": ["interpolate", ["linear"], ["get", "eviction_count"], 0, 0, 20, 1],
            "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 8, 1, 15, 3],
            "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"], 0, "rgba(0,0,255,0)", 0.2, "#deebf7", 0.5, "#6baed6", 0.8, "#fb6a4a", 1, "#99000d"],
            "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 8, 12, 15, 30],
            "heatmap-opacity": 0.82
          }
        });
      }
    }

    const modeSelect = document.getElementById("map-mode-select");
    function setMapMode(mode) {
      if (mode !== "clusters") ensureRawLayers();
      const clusterVisibility = mode === "clusters" ? "visible" : "none";
      ["clusters", "cluster-count", "unclustered-point"].forEach(id => {
        if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", clusterVisibility);
      });
      if (map.getLayer("all-points")) map.setLayoutProperty("all-points", "visibility", mode === "points" ? "visible" : "none");
      if (map.getLayer("eviction-heatmap")) map.setLayoutProperty("eviction-heatmap", "visibility", mode === "heatmap" ? "visible" : "none");
    }
    if (modeSelect) modeSelect.addEventListener("change", e => setMapMode(e.target.value));

    const months = [...new Set(
      evictionGeojson.features.flatMap(f =>
        String(f.properties.month_counts || "")
          .split(",")
          .map(part => part.split(":")[0])
          .filter(Boolean)
      )
    )].sort();
    let animationTimer = null;
    let animationIndex = 0;
    const animationBtn = document.getElementById("animation-btn");
    const animationLabel = document.getElementById("animation-label");
    function showAnimationMonth(monthKey) {
      const monthFeatures = evictionGeojson.features.flatMap(f => {
        const entry = String(f.properties.month_counts || "")
          .split(",")
          .find(part => part.startsWith(monthKey + ":"));
        if (!entry) return [];
        return [{
          ...f,
          properties: {...f.properties, eviction_count: Number(entry.split(":")[1]) || 0}
        }];
      });
      const data = {type: "FeatureCollection", features: monthFeatures};
      map.getSource("evictions").setData(data);
      const rawSource = map.getSource("evictions-raw");
      if (rawSource) rawSource.setData(data);
      animationLabel.textContent = monthKey;
    }
    function stopAnimation(reset) {
      if (animationTimer) clearInterval(animationTimer);
      animationTimer = null;
      animationBtn.textContent = "▶ Animate";
      if (reset) {
        map.getSource("evictions").setData(evictionGeojson);
        const rawSource = map.getSource("evictions-raw");
        if (rawSource) rawSource.setData(evictionGeojson);
        animationLabel.textContent = "All dates";
      }
    }
    if (animationBtn) animationBtn.addEventListener("click", () => {
      if (animationTimer) { stopAnimation(true); return; }
      animationBtn.textContent = "■ Stop";
      animationIndex = 0;
      if (months.length) showAnimationMonth(months[0]);
      const speed = Number(document.getElementById("animation-speed").value || 900);
      animationTimer = setInterval(() => {
        animationIndex += 1;
        if (animationIndex >= months.length) { stopAnimation(true); return; }
        showAnimationMonth(months[animationIndex]);
      }, speed);
    });


    // Cluster click: zoom in
    map.on("click", "clusters", (e) => {
      const features = map.queryRenderedFeatures(e.point, {
        layers: ["clusters"]
      });
      const clusterId = features[0].properties.cluster_id;
      map.getSource("evictions").getClusterExpansionZoom(
        clusterId,
        (err, zoom) => {
          if (err) return;
          map.easeTo({
            center: features[0].geometry.coordinates,
            zoom: zoom
          });
        }
      );
    });

    // Popup on unclustered point
    map.on("click", "unclustered-point", (e) => {
      const props = e.features[0].properties;
      const coords = e.features[0].geometry.coordinates.slice();

      const rent = Number(props.total_rent_owed || 0);
      const rentFormatted = rent.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      });
      const hasZori = props.zori_rent !== null && props.zori_rent !== "" && Number.isFinite(Number(props.zori_rent));
      const zoriFormatted = hasZori
        ? `$${Number(props.zori_rent).toLocaleString("en-US", {maximumFractionDigits: 0})} (${zoriMonthLabel})`
        : "N/A";

      const html = `
        <div style="font-size:12px;line-height:1.4">
          <strong>Address:</strong> ${props.full_address}<br/>
          <strong>Filing period:</strong> ${props.date_filed}<br/>
          <strong>Number of Tenants Evicted:</strong> ${props.eviction_count}<br/>
          <strong>Cause:</strong> ${props.cause || "N/A"}<br/>
          <strong>Council District:</strong> ${props.council_district || "N/A"}<br/>
          <strong>ZIP:</strong> ${props.zip || "N/A"}<br/>
          <strong>Zillow typical asking rent:</strong> ${zoriFormatted}<br/>
          <strong>Total Rent Owed:</strong> $${rentFormatted}<br/>
          <strong>APN:</strong> ${props.apn || "N/A"}
          <br/><strong>Address filing dates:</strong> ${props.repeat_count || 1}
        </div>
      `;

      new mapboxgl.Popup()
        .setLngLat(coords)
        .setHTML(html)
        .addTo(map);
    });

    // Hover cursor changes
    map.on("mouseenter", "clusters", () => {
      map.getCanvas().style.cursor = "pointer";
    });
    map.on("mouseleave", "clusters", () => {
      map.getCanvas().style.cursor = "";
    });
    map.on("mouseenter", "unclustered-point", () => {
      map.getCanvas().style.cursor = "pointer";
    });
    map.on("mouseleave", "unclustered-point", () => {
      map.getCanvas().style.cursor = "";
    });

    // Draw events  selection update
    if (draw) {
      map.on("draw.create", handleDrawChange);
      map.on("draw.update", handleDrawChange);
      map.on("draw.delete", () => {
        updateSelectionPanel([]);
        sendSelectionToPython([]);
      });
    }

    // Polygon styling to blue
    const blue = "#1E90FF";
    const fillLayers = [
      "gl-draw-polygon-fill-inactive",
      "gl-draw-polygon-fill-active"
    ];
    const lineLayers = [
      "gl-draw-polygon-stroke-inactive",
      "gl-draw-polygon-stroke-active"
    ];

    fillLayers.forEach(id => {
      if (map.getLayer(id)) {
        map.setPaintProperty(id, "fill-color", blue);
        map.setPaintProperty(id, "fill-opacity", 0.25);
      }
    });
    lineLayers.forEach(id => {
      if (map.getLayer(id)) {
        map.setPaintProperty(id, "line-color", blue);
        map.setPaintProperty(id, "line-width", 2);
      }
    });

    // ==========================
    // Neighborhood and ZIP choropleth overlays
    // ==========================
    if (nbhdChoro && nbhdChoro.type === "FeatureCollection" && nbhdChoro.features && nbhdChoro.features.length) {
      map.addSource("nbhd-choropleth-src", {
        type: "geojson",
        data: nbhdChoro
      });
    }
    if (zipZoriChoro && zipZoriChoro.type === "FeatureCollection" && zipZoriChoro.features && zipZoriChoro.features.length) {
      map.addSource("zip-zori-src", {
        type: "geojson",
        data: zipZoriChoro
      });
    }

    const legend = document.getElementById("overlay-legend");
    const legendTitle = document.getElementById("overlay-legend-title");
    const legendMin = document.getElementById("overlay-legend-min");
    const legendMax = document.getElementById("overlay-legend-max");

    function percentile(sortedValues, proportion) {
      if (!sortedValues.length) return NaN;
      const index = Math.max(0, Math.min(sortedValues.length - 1, Math.round((sortedValues.length - 1) * proportion)));
      return sortedValues[index];
    }

    function formatOverlayValue(value, propertyName) {
      if (propertyName === "zori_rent" || propertyName === "median_income") {
        return "$" + Math.round(value).toLocaleString("en-US");
      }
      if (propertyName === "rent_burden_pct") return value.toFixed(1) + "%";
      return Math.round(value).toLocaleString("en-US");
    }

    function updateChoropleth(propertyName) {
      if (map.getLayer("zip-zori-outline")) map.removeLayer("zip-zori-outline");
      if (map.getLayer("nbhd-choropleth-layer")) {
        map.removeLayer("nbhd-choropleth-layer");
      }
      if (legend) legend.style.display = "none";
      if (!propertyName || propertyName === "none") {
        setNbhdInfo("Choose an overlay to inspect neighborhood or ZIP values.");
        return;
      }

      const isZori = propertyName === "zori_rent";
      const sourceId = isZori ? "zip-zori-src" : "nbhd-choropleth-src";
      if (!map.getSource(sourceId)) return;
      const sourceData = isZori ? zipZoriChoro : nbhdChoro;
      const features = (sourceData && sourceData.features) ? sourceData.features : [];
      const values = features
        .map(f => Number(f.properties[propertyName]))
        .filter(v => !isNaN(v) && isFinite(v));

      if (!values.length) return;
      values.sort((a, b) => a - b);

      // Zillow ZIP values have a long high-rent tail. Use a robust visual
      // range while preserving exact values in the hover details.
      const min = isZori ? percentile(values, 0.10) : values[0];
      const max = isZori ? percentile(values, 0.90) : values[values.length - 1];
      if (!isFinite(min) || !isFinite(max) || min === max) return;
      const mid = isZori ? percentile(values, 0.50) : (min + max) / 2;

      map.addLayer(
        {
          id: "nbhd-choropleth-layer",
          type: "fill",
          source: sourceId,
          paint: {
            "fill-color": [
              "case",
              ["has", propertyName],
              [
                "interpolate",
                ["linear"],
                ["to-number", ["get", propertyName]],
                min, "#eff3ff",
                mid, "#6baed6",
                max, "#08306b"
              ],
              "#cbd5e1"
            ],
            "fill-opacity": ["case", ["has", propertyName], 0.58, 0.18],
            "fill-outline-color": "#555555"
          }
        },
        "clusters"
      );

      if (isZori) {
        map.addLayer({
          id: "zip-zori-outline",
          type: "line",
          source: sourceId,
          paint: {
            "line-color": ["case", ["==", ["get", "selected"], true], "#f97316", "#475569"],
            "line-width": ["case", ["==", ["get", "selected"], true], 3, 0.8],
            "line-opacity": 0.9
          }
        }, "clusters");
        bindZipHover("nbhd-choropleth-layer");
        setNbhdInfo(defaultOverlayMessage());
      } else {
        bindNeighborhoodHover("nbhd-choropleth-layer");
        setNbhdInfo(defaultOverlayMessage());
      }

      if (legend) {
        legend.style.display = "block";
        legendTitle.textContent = isZori
          ? `Typical asking rent — ${zoriMonthLabel} (10th–90th percentile color range)`
          : document.getElementById("choropleth-select").selectedOptions[0].textContent;
        legendMin.textContent = formatOverlayValue(min, propertyName);
        legendMax.textContent = formatOverlayValue(max, propertyName);
      }
    }

    const choroSelect = document.getElementById("choropleth-select");
    if (choroSelect) {
      choroSelect.addEventListener("change", (e) => {
        updateChoropleth(e.target.value);
      });
      updateChoropleth(choroSelect.value);
    }
    if (mapStatus) mapStatus.textContent = "Rendering eviction clusters…";
    const finishMapLoad = () => {
      const sourceFeatures = map.querySourceFeatures("evictions");
      if (sourceFeatures.length > 0 || evictionFeatures.length === 0) {
        if (mapStatus) mapStatus.style.display = "none";
        map.off("idle", finishMapLoad);
      }
    };
    map.on("idle", finishMapLoad);
  });
</script>
</body>
</html>
"""

# Inject JSON + config into template
zori_month_label = (
    pd.to_datetime(selected_zori_month).strftime("%B %Y")
    if selected_zori_month is not None
    else "Unavailable"
)
html_final = (
    html_template
    .replace("EVIC_DATA_PLACEHOLDER", evictions_json)
    .replace("MAPBOX_TOKEN_PLACEHOLDER", MAPBOX_TOKEN)
    .replace("NBHD_CHORO_PLACEHOLDER", nbhd_choro_json)
    .replace("ZIP_ZORI_CHORO_PLACEHOLDER", zori_zip_json)
    .replace("ZORI_MONTH_PLACEHOLDER", zori_month_label)
)

# ==============================
# LAYOUT: MAP FULL-WIDTH + ANALYTICS BELOW
# ==============================
st.markdown("### 3D Eviction Map")
if selected_zori_month is not None:
    st.caption(
        f"The default ZIP overlay shows Zillow ZORI for {zori_month_label}; selected ZIPs have an orange outline. "
        "ZORI estimates typical market asking rent and is not a unit-level contract rent. ZIP areas use Census ZCTAs."
    )
if not MAPBOX_TOKEN:
    st.error("The ALPHA map needs a Mapbox public token. In Streamlit Cloud → App settings → Secrets, set MAPBOX_TOKEN to your public token (pk.…). All notice charts and downloads remain available below.")
else:
    embed_html(html_final, height=720)

st.markdown("---")
st.markdown("### Summary & Download")

# --- Full filtered metrics ---
sum_cols = st.columns(4)
total_units = int(ev_filtered["Eviction_Count"].sum())
total_addresses = ev_filtered["full_address"].nunique()
total_rent = float(ev_filtered["Total_Rent_Owed"].sum())

with sum_cols[0]:
    st.metric("Notice records (filtered)", f"{total_units:,}")
with sum_cols[1]:
    st.metric("Unique Address–Date Nodes", f"{len(ev_filtered):,}")
with sum_cols[2]:
    st.metric("Approx. Total Rent Owed", f"${total_rent:,.0f}")
with sum_cols[3]:
    csv_bytes = ev_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download CSV (filtered)",
        data=csv_bytes,
        file_name=f"evictions_filtered_{start_date}_to_{end_date}.csv",
        mime="text/csv",
    )

# --- Selection metrics (if any) ---
if not ev_selected.empty:
    st.markdown("#### Polygon Selection (subset of filtered)")
    sel_cols = st.columns(3)
    sel_units = int(ev_selected["Eviction_Count"].sum())
    sel_nodes = len(ev_selected)
    sel_rent = float(ev_selected["Total_Rent_Owed"].sum())

    with sel_cols[0]:
        st.metric("Selected Units", f"{sel_units:,}")
    with sel_cols[1]:
        st.metric("Selected Evictions", f"{sel_nodes:,}")
    with sel_cols[2]:
        st.metric("Selected Rent Owed", f"${sel_rent:,.0f}")
else:
    st.caption(
        "Polygon-based stats are shown in the selection panel on the map (bottom-right)."
    )

st.write(f"**Area filter:** {selected_area}")
st.markdown("---")

# ==============================
# CALLBACKS FOR INTERACTIVE CHART FILTERING
# ==============================
def _update_trend_filter():
    event = st.session_state.get("trend_chart")
    # No selection -> clear filter
    if not event or "selection" not in event or not event["selection"].get("points"):
        st.session_state["trend_filter"] = None
        return

    x_val = event["selection"]["points"][0]["x"]

    month_ts = pd.to_datetime(x_val)
    month_ts = month_ts.replace(day=1)

    next_month_ts = month_ts + pd.offsets.MonthBegin(1)

    st.session_state["trend_filter"] = (month_ts, next_month_ts)

def _update_rent_filter():
    event = st.session_state.get("rent_hist_chart")
    if not event or "selection" not in event or not event["selection"].get("points"):
        st.session_state["rent_filter"] = None
        return

    xs = [p["x"] for p in event["selection"]["points"]]
    if not xs:
        st.session_state["rent_filter"] = None
        return

    r_min = float(min(xs))
    r_max = float(max(xs))
    st.session_state["rent_filter"] = (r_min, r_max)

# Reuse the same notice-level population for the rent comparison and the
# individualized rent-owed distribution. Address-date matching carries all
# active map filters down to the original LAHD notice rows.
rent_records = load_rent_records(EVICTION_CSV, os.path.getmtime(EVICTION_CSV))
selected_nodes = ev_filtered[["full_address", "Date_Filed"]].drop_duplicates()
df_rent_records = rent_records.merge(
    selected_nodes,
    on=["full_address", "Date_Filed"],
    how="inner",
)

# ==============================
# ANALYTICS TABS
# ==============================
tab_trend, tab_cpi, tab_hist, tab_causes, tab_demo = st.tabs(
    [
        " Eviction Trend",
        " Rent Comparison",
        " Rent Owed Distribution",
        " Eviction Causes",
        " Demographics (Census Data)",
    ], on_change="rerun", key="alpha_analytics_tabs",
)

# --- Eviction Trend (monthly) ---
with tab_trend:
    if tab_trend.open:
        st.subheader("Monthly Evictions")
        st.caption("Counts are LAHD notice records received, not completed evictions or unique tenants. This chart follows map filters and includes only geocoded records. See Methods for all-source monthly coverage.")
        st.caption("Use mouse to click on a data point to view evictions of that month on the map. Click 'clear chart based filters' on the left to reset data.")
        df_trend = ev_filtered.copy()
        df_trend["month"] = df_trend["Date_Filed"].dt.to_period("M").dt.to_timestamp()
        trend = (
            df_trend.groupby("month", as_index=False)["Eviction_Count"]
            .sum()
            .rename(columns={"Eviction_Count": "evictions"})
            .sort_values("month")
        )

        if trend.empty:
            st.info("No data in this range.")
        else:
            fig = px.line(
                trend,
                x="month",
                y="evictions",
                markers=True,
                labels={"month": "Month", "evictions": "Notice records"},
            )

            fig.update_traces(
                mode="lines+markers+text",
                marker=dict(size=15),
                text=trend["evictions"],
                textposition="top center",
            )
            fig.update_layout(
                height=430,
                margin=dict(l=40, r=20, t=60, b=40),
                yaxis_title="Notice records",
                yaxis_tickformat=",.0f",
            )
            st.plotly_chart(
                fig,
                width="stretch",
                key="trend_chart",
                on_select=_update_trend_filter,
                selection_mode="points",
            )
            if st.session_state.get("trend_filter") is not None:
                st.caption("Eviction trend filter is active (use 'Clear Chart-Based Filters' in the sidebar to reset).")

# --- Zillow asking rent vs. LAHD reported rent owed ---
with tab_cpi:
    if tab_cpi.open:
        st.subheader("Monthly asking rent and reported rent owed")
        st.caption(
            "Zillow ZORI is the unweighted average typical asking rent across ZIP codes represented "
            "by the current filters. LAHD rent owed is the average positive balance reported on each "
            "matched notice. A notice balance may cover multiple months, so the two lines are contextually "
            "related but are not equivalent rent measures."
        )

        positive_owed_records = df_rent_records[df_rent_records["Rent_Owed"] > 0].copy()
        eligible_zips = sorted(
            zip_code for zip_code in ev_filtered["Zip"].dropna().astype(str).unique()
            if zip_code
        )

        if ev_filtered.empty:
            st.info("No eviction records match the current filters.")
        elif not eligible_zips or zori_zip.empty or not zori_month_columns:
            st.info("No Zillow ZIP observations match the current filtered eviction records.")
        else:
            comparison_start = ev_filtered["Date_Filed"].min().to_period("M").to_timestamp()
            comparison_end = ev_filtered["Date_Filed"].max().to_period("M").to_timestamp()
            comparison_months = pd.date_range(comparison_start, comparison_end, freq="MS")

            zori_matrix = zori_zip.reindex(eligible_zips)[zori_month_columns].T
            zori_matrix.index = pd.to_datetime(zori_matrix.index).to_period("M").to_timestamp()
            zori_matrix = zori_matrix.loc[
                (zori_matrix.index >= comparison_start)
                & (zori_matrix.index <= comparison_end)
            ]
            zori_monthly = pd.DataFrame(
                {
                    "month": zori_matrix.index,
                    "average_zori": zori_matrix.mean(axis=1, skipna=True).values,
                    "zori_zip_count": zori_matrix.count(axis=1).values,
                }
            )

            if positive_owed_records.empty:
                owed_monthly = pd.DataFrame(columns=["month", "average_rent_owed", "notice_count"])
            else:
                positive_owed_records["month"] = (
                    positive_owed_records["Date_Filed"].dt.to_period("M").dt.to_timestamp()
                )
                owed_monthly = (
                    positive_owed_records.groupby("month", as_index=False)
                    .agg(
                        average_rent_owed=("Rent_Owed", "mean"),
                        notice_count=("Rent_Owed", "size"),
                    )
                )

            rent_comparison = pd.DataFrame({"month": comparison_months})
            rent_comparison = rent_comparison.merge(zori_monthly, on="month", how="left")
            rent_comparison = rent_comparison.merge(owed_monthly, on="month", how="left")
            rent_comparison["zori_zip_count"] = rent_comparison["zori_zip_count"].fillna(0).astype(int)
            rent_comparison["notice_count"] = rent_comparison["notice_count"].fillna(0).astype(int)

            if rent_comparison[["average_zori", "average_rent_owed"]].notna().sum().sum() == 0:
                st.info("No Zillow or positive rent-owed observations fall in this filtered period.")
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=rent_comparison["month"],
                        y=rent_comparison["average_zori"],
                        mode="lines+markers",
                        name="Average ZIP asking rent (ZORI)",
                        line={"color": "#2563eb", "width": 3},
                        marker={"size": 6},
                        customdata=rent_comparison[["zori_zip_count"]].to_numpy(),
                        hovertemplate=(
                            "%{x|%B %Y}<br>Average ZORI: $%{y:,.0f}"
                            "<br>ZIP observations: %{customdata[0]:,}<extra></extra>"
                        ),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=rent_comparison["month"],
                        y=rent_comparison["average_rent_owed"],
                        mode="lines+markers",
                        name="Average reported rent owed per notice",
                        line={"color": "#ea580c", "width": 3},
                        marker={"size": 6},
                        customdata=rent_comparison[["notice_count"]].to_numpy(),
                        hovertemplate=(
                            "%{x|%B %Y}<br>Average reported owed: $%{y:,.0f}"
                            "<br>Positive-balance notices: %{customdata[0]:,}<extra></extra>"
                        ),
                    )
                )
                fig.update_layout(
                    height=470,
                    margin=dict(l=40, r=20, t=30, b=40),
                    yaxis_title="USD",
                    yaxis_tickprefix="$",
                    yaxis_tickformat=",.0f",
                    hovermode="x unified",
                    legend={"orientation": "h", "y": 1.08, "x": 0},
                )
                st.plotly_chart(fig, width="stretch")
                st.download_button(
                    "Download monthly rent comparison",
                    rent_comparison.to_csv(index=False),
                    "zillow_zori_vs_lahd_rent_owed.csv",
                    "text/csv",
                )

        with st.expander("CPI rent-index context", expanded=False):
            st.caption(f"{FRED_SERIES_ID}: US city average rent of primary residence; index 1982–1984=100. Source: {cpi_source}. This is an index, not a dollar rent.")
            for note in cpi_notes:
                st.info(note)
            if cpi_df.empty or ev_filtered.empty:
                st.info("CPI data could not be loaded for this view.")
            else:
                date_min = ev_filtered["Date_Filed"].min()
                date_max = ev_filtered["Date_Filed"].max()
                sub_cpi = cpi_df[
                    (cpi_df["date"] >= date_min) & (cpi_df["date"] <= date_max)
                ]
                if sub_cpi.empty:
                    st.info("No CPI observations in this date range.")
                else:
                    fig_cpi = px.line(
                        sub_cpi,
                        x="date",
                        y="value",
                        markers=True,
                        labels={"date": "Date", "value": "CPI (Rent Index)"},
                    )
                    fig_cpi.update_layout(
                        height=360,
                        margin=dict(l=40, r=20, t=30, b=40),
                        yaxis_title="CPI (Rent Index)",
                        yaxis_tickformat=",.1f",
                    )
                    st.plotly_chart(fig_cpi, width="stretch")

# --- Rent Owed Distribution (95% cutoff) ---
with tab_hist:
    if tab_hist.open:
        st.subheader("Distribution of reported rent owed")
        st.caption(
            "The individualized histogram uses every LAHD notice with a positive rent-owed "
            "amount; the top 5% are excluded. LAHD does not provide a verified unit "
            "identifier, so records should not be interpreted as confirmed unique apartments."
        )

        df_rent_records = df_rent_records[df_rent_records["Rent_Owed"] > 0].copy()

        if df_rent_records.empty:
            st.info("No positive rent-owed values to display.")
        else:
            cutoff = float(df_rent_records["Rent_Owed"].quantile(0.95))
            df_rent_trimmed = df_rent_records[
                df_rent_records["Rent_Owed"] <= cutoff
            ].copy()

            metrics = st.columns(3)
            metrics[0].metric("Notice records shown", f"{len(df_rent_trimmed):,}")
            metrics[1].metric(
                "Median reported rent owed",
                f"${df_rent_trimmed['Rent_Owed'].median():,.0f}",
            )
            metrics[2].metric("95th-percentile cutoff", f"${cutoff:,.0f}")

            distribution_view = st.radio(
                "Distribution view",
                [
                    "Density curve",
                    "Individual record marks",
                    "Address-date histogram",
                ],
                horizontal=True,
            )

            if distribution_view == "Density curve":
                values = df_rent_trimmed["Rent_Owed"].to_numpy(dtype="float64")
                sample_size = len(values)
                standard_deviation = float(np.std(values, ddof=1))
                q25, q75 = np.percentile(values, [25, 75])
                robust_sigma = float((q75 - q25) / 1.349)
                positive_scales = [
                    value
                    for value in [standard_deviation, robust_sigma]
                    if np.isfinite(value) and value > 0
                ]
                scale = min(positive_scales) if positive_scales else max(cutoff / 100, 1)
                bandwidth = max(
                    0.9 * scale * sample_size ** (-1 / 5),
                    cutoff / 512,
                )

                counts, edges = np.histogram(values, bins=512, range=(0, cutoff))
                centers = (edges[:-1] + edges[1:]) / 2
                bin_width = edges[1] - edges[0]
                sigma_bins = max(bandwidth / bin_width, 1.0)
                radius = min(int(np.ceil(4 * sigma_bins)), 200)
                kernel_axis = np.arange(-radius, radius + 1)
                kernel = np.exp(-0.5 * (kernel_axis / sigma_bins) ** 2)
                kernel /= kernel.sum()
                density = np.convolve(counts, kernel, mode="same") / (
                    sample_size * bin_width
                )

                fig = go.Figure(
                    go.Scatter(
                        x=centers,
                        y=density,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#2c5aa0", width=2.5),
                        fillcolor="rgba(44, 90, 160, 0.18)",
                        hovertemplate=(
                            "<b>Reported rent owed: %{x:$,.0f}</b>"
                            "<br>Estimated density: %{y:.3e}<extra></extra>"
                        ),
                    )
                )
                median_rent = float(np.median(values))
                mean_rent = float(np.mean(values))
                fig.add_vline(
                    x=median_rent,
                    line_dash="dash",
                    line_color="#8c6d31",
                    annotation_text=f"Median ${median_rent:,.0f}",
                    annotation_position="top left",
                )
                fig.add_vline(
                    x=mean_rent,
                    line_dash="dot",
                    line_color="#756bb1",
                    annotation_text=f"Mean ${mean_rent:,.0f}",
                    annotation_position="top right",
                )
                fig.update_layout(
                    height=430,
                    margin=dict(l=75, r=25, t=45, b=65),
                    xaxis=dict(
                        title="Reported rent owed per notice record",
                        tickformat="$,.0f",
                        rangemode="tozero",
                        gridcolor="#e5e7eb",
                    ),
                    yaxis=dict(
                        title="Estimated probability density",
                        tickformat=".2e",
                        rangemode="tozero",
                        gridcolor="#e5e7eb",
                    ),
                    showlegend=False,
                    hoverlabel=dict(font_size=13),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    f"Gaussian kernel density estimate using all {sample_size:,} displayed "
                    f"notice records. Bandwidth: ${bandwidth:,.0f}, selected with a robust "
                    "Silverman rule. The same 95th-percentile cutoff and active filters apply."
                )

            elif distribution_view == "Individual record marks":
                rent_values = np.ascontiguousarray(
                    df_rent_trimmed["Rent_Owed"].to_numpy(dtype="<f4")
                )
                encoded_values = base64.b64encode(rent_values.tobytes()).decode("ascii")
                individualized_chart = """
                <div id="rent-strip" style="font-family:system-ui,-apple-system,sans-serif;color:#273244">
                  <div style="font-size:13px;margin:0 0 6px 58px">
                    Each blue mark is one LAHD notice record
                  </div>
                  <canvas id="rent-canvas" role="img"
                    aria-label="Individual reported rent owed amounts with the highest five percent excluded"
                    style="display:block;width:100%;height:330px"></canvas>
                  <div style="text-align:center;font-size:13px;margin-top:6px">
                    Reported rent owed per notice record
                  </div>
                </div>
                <script>
                (() => {
                  const canvas = document.getElementById("rent-canvas");
                  const ctx = canvas.getContext("2d");
                  const bytes = Uint8Array.from(atob("__VALUES__"), c => c.charCodeAt(0));
                  const values = new Float32Array(bytes.buffer);
                  const cutoff = __CUTOFF__;
                  const margins = {left: 58, right: 18, top: 16, bottom: 32};

                  function money(value) {
                    return "$" + Math.round(value).toLocaleString("en-US");
                  }

                  function draw() {
                    const dpr = Math.max(1, window.devicePixelRatio || 1);
                    const width = Math.max(520, canvas.clientWidth);
                    const height = 330;
                    canvas.width = Math.floor(width * dpr);
                    canvas.height = Math.floor(height * dpr);
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                    ctx.clearRect(0, 0, width, height);

                    const plotWidth = width - margins.left - margins.right;
                    const plotHeight = height - margins.top - margins.bottom;

                    ctx.strokeStyle = "#e5e7eb";
                    ctx.fillStyle = "#4b5563";
                    ctx.font = "12px system-ui, sans-serif";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "top";
                    for (let tick = 0; tick <= 4; tick += 1) {
                      const x = margins.left + plotWidth * tick / 4;
                      ctx.beginPath();
                      ctx.moveTo(x, margins.top);
                      ctx.lineTo(x, margins.top + plotHeight);
                      ctx.stroke();
                      ctx.fillText(money(cutoff * tick / 4), x, margins.top + plotHeight + 8);
                    }

                    ctx.fillStyle = "#2c5aa0";
                    ctx.globalAlpha = 0.10;
                    for (let i = 0; i < values.length; i += 1) {
                      const x = margins.left + (values[i] / cutoff) * plotWidth;
                      const hash = Math.imul(i + 1, 2654435761) >>> 0;
                      const y = margins.top + (hash / 4294967295) * plotHeight;
                      ctx.fillRect(x, y, 1.5, 1.5);
                    }
                    ctx.globalAlpha = 1;

                    ctx.strokeStyle = "#9ca3af";
                    ctx.beginPath();
                    ctx.moveTo(margins.left, margins.top + plotHeight);
                    ctx.lineTo(width - margins.right, margins.top + plotHeight);
                    ctx.stroke();
                  }

                  new ResizeObserver(draw).observe(canvas);
                  draw();
                })();
                </script>
                """
                individualized_chart = individualized_chart.replace(
                    "__VALUES__", encoded_values
                ).replace("__CUTOFF__", repr(cutoff))
                embed_html(individualized_chart, height=390)
                st.caption(
                    "Every displayed notice is drawn as its own mark. Dense, darker areas "
                    "indicate many records with similar reported rent amounts; no buckets "
                    "or cumulative transformation are used."
                )
            else:
                df_hist = ev_filtered[ev_filtered["Total_Rent_Owed"] > 0].copy()
                node_cutoff = df_hist["Total_Rent_Owed"].quantile(0.95)
                df_hist = df_hist[df_hist["Total_Rent_Owed"] <= node_cutoff]

                fig = px.histogram(
                    df_hist,
                    x="Total_Rent_Owed",
                    nbins=40,
                    labels={"Total_Rent_Owed": "Total Rent Owed ($)"},
                )
                # Bin counts belong in hover details, not overlapping text labels.
                fig.update_traces(hovertemplate="Reported rent: %{x}<br>Address-date nodes: %{y:,.0f}<extra></extra>")
                fig.update_layout(
                    height=430,
                    margin=dict(l=70, r=20, t=70, b=60),
                    xaxis_tickformat="$,.0f",
                    yaxis_title="Address-date filing nodes",
                )
                st.plotly_chart(
                    fig,
                    width="stretch",
                    key="rent_hist_chart",
                    on_select=_update_rent_filter,
                    selection_mode="box",
                )
                if st.session_state.get("rent_filter") is not None:
                    st.caption(
                        "Rent owed histogram filter is active (use Clear Chart-Based "
                        "Filters in the sidebar to reset)."
                    )

# --- Eviction Causes ---
with tab_causes:
    if tab_causes.open:
        st.subheader("Eviction causes")
        st.caption(
            "Each address-date node appears once. Records listing more than one LAHD cause "
            "are grouped as Multiple stated causes. Bars are ranked by affected tenants."
        )
        df_causes = ev_filtered.copy()

        if df_causes.empty:
            st.info("No cause records to display.")
        else:
            cause_sum = (
                df_causes.groupby("Cause_Category", as_index=False)["Eviction_Count"]
                .sum()
                .rename(columns={"Eviction_Count": "total_evictions"})
                .sort_values("total_evictions", ascending=False)
            )

            chart_total = int(cause_sum["total_evictions"].sum())
            cause_sum["share"] = cause_sum["total_evictions"] / chart_total
            cause_sum["display_label"] = cause_sum.apply(
                lambda row: f"{int(row['total_evictions']):,}  ({row['share']:.1%})",
                axis=1,
            )
            cause_sum = cause_sum.sort_values("total_evictions", ascending=True)

            # Category names already identify each bar, so color is reserved for emphasis.
            bar_colors = [
                "#08519c" if category == "Non-payment of rent" else "#6baed6"
                for category in cause_sum["Cause_Category"]
            ]

            fig = go.Figure(
                go.Bar(
                    x=cause_sum["total_evictions"],
                    y=cause_sum["Cause_Category"],
                    orientation="h",
                    text=cause_sum["display_label"],
                    textposition="outside",
                    textfont=dict(size=13, color="#172033"),
                    cliponaxis=False,
                    marker_color=bar_colors,
                    customdata=cause_sum[["share"]],
                    hovertemplate=(
                        "<b>%{y}</b><br>%{x:,.0f} affected tenants"
                        "<br>%{customdata[0]:.1%} of filtered total<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                height=500,
                margin=dict(l=190, r=135, t=20, b=60),
                bargap=0.28,
                showlegend=False,
                xaxis_title="Affected tenants",
                yaxis_title="",
                xaxis_tickformat=",.0f",
                xaxis=dict(gridcolor="#e5e7eb", tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=13, color="#273244")),
                hoverlabel=dict(font_size=13),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Filtered total: {chart_total:,} affected tenants.")

            with st.expander("View detailed LAHD source terms"):
                source_terms = (
                    df_causes[["Cause", "Eviction_Count"]]
                    .assign(Source_Term=df_causes["Cause"].map(split_cause_terms))
                    .explode("Source_Term")
                )
                source_terms = source_terms[source_terms["Source_Term"].notna()]
                source_term_summary = (
                    source_terms.groupby("Source_Term", as_index=False)["Eviction_Count"]
                    .sum()
                    .rename(columns={"Eviction_Count": "Affected tenants mentioning term"})
                    .sort_values("Affected tenants mentioning term", ascending=False)
                )
                st.caption(
                    "A record with multiple stated causes appears under each applicable source term, "
                    "so this detail table is not a part-to-whole total."
                )
                st.dataframe(source_term_summary, hide_index=True, width="stretch")

# --- Demographics (Census) ---
with tab_demo:
    if tab_demo.open:
        st.subheader("Neighborhood Demographic Based on Census Data (Irrespective of Evictions)")
        with st.expander("What do these demographic columns mean?"):
            st.markdown("`White_share`, `Black_share`, `Hispanic_share`, `Asian_share`, and `Other_share` are neighborhood population percentages: 4.7 means 4.7%, not 0.047%. They describe residents of the neighborhood, not tenants named in notices. In the supplied file the five columns sum to 100%. `Other_share` is the residual category outside the four named groups. The original Census variable crosswalk is not included, so its exact racial/ethnic membership and treatment of multiracial residents cannot be verified. Do not interpret it as other eviction causes or missing notice data.")
            st.markdown("`median_income` and `median_rent` are supplied neighborhood dollar estimates. `rent_burden_pct` is a supplied percentage; its denominator and threshold require the original source documentation. The All Areas display is an unweighted average of neighborhoods, not a population-weighted Los Angeles estimate.")

        if census_df.empty:
            st.info(
                "No census file found. To enable this panel, create "
                "`data/neighborhood_census.csv` with columns like: "
                "`name, median_income, median_rent, rent_burden_pct, "
                "White_share, Black_share, Hispanic_share, Asian_share, Other_share`."
            )
        else:
            # Build dropdown options: special "All Areas" + all individual neighborhoods
            area_options = ["All Areas (Los Angeles)"] + sorted(
                census_df["name"].unique().tolist()
            )

            # Default selection: if current map area is a neighborhood in census_df, use that
            default_index = 0
            if selected_area in census_df["name"].values:
                try:
                    default_index = area_options.index(selected_area)
                except ValueError:
                    default_index = 0

            demo_area = st.selectbox(
                "Select area for demographics",
                options=area_options,
                index=default_index,
            )

            # If user picks the special "All Areas" option, compute an aggregate row
            if demo_area == "All Areas (Los Angeles)":
                # Take simple (unweighted) averages for numeric columns
                numeric_cols = census_df.select_dtypes(include=[np.number]).columns
                agg_series = census_df[numeric_cols].mean()
                agg_series["name"] = "All Areas (Los Angeles)"
                row = agg_series
            else:
                row = census_df[census_df["name"] == demo_area].iloc[0]

            st.markdown(f"**Area:** `{row['name']}`")

            cols_demo = st.columns(1)
            with cols_demo[0]:
                if "rent_burden_pct" in row:
                    st.metric(
                        "Reported rent burden (source definition unverified)",
                        f"{row['rent_burden_pct']:.1f}%",
                    )

            race_cols = [
                "White_share",
                "Black_share",
                "Hispanic_share",
                "Asian_share",
                "Other_share",
            ]
            existing_race_cols = [c for c in race_cols if c in row.index]
            if existing_race_cols:
                race_vals = [row[c] for c in existing_race_cols]
                race_labels = [
                    c.replace("_share", "").replace("_", " ")
                    for c in existing_race_cols
                ]
                race_df = pd.DataFrame({"group": race_labels, "share": race_vals})

                race_colors = {
                    "White": "#9ecae1",
                    "Black": "#6baed6",
                    "Hispanic": "#4292c6",
                    "Asian": "#2171b5",
                    "Other": "#084594",
                }

                fig_demo = px.pie(
                    race_df,
                    names="group",
                    values="share",
                    title="Racial/Ethnic Composition of area (separate from rent and evictions data)",
                    color="group",
                    color_discrete_map=race_colors,
                )
                fig_demo.update_traces(textposition="inside", textinfo="label+percent")
                fig_demo.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=40))
                st.plotly_chart(fig_demo, width="stretch")
            else:
                st.info("No racial composition columns found in census file.")

# ==============================
# RESEARCH LAB — PROPERTY, PLACE, POLICY, AND EXPORTS
# ==============================
st.markdown("---")
st.header("Research Lab")
st.caption("Exploratory tools; correlations do not establish causation.")

(
    tab_property,
    tab_routine,
    tab_compare,
    tab_hotspots,
    tab_alerts,
    tab_forecast,
    tab_reports,
    tab_method,
    tab_admin,
) = st.tabs(
    [
        "Property & parcel",
        "Routine filers",
        "Compare areas",
        "Hotspots",
        "Alerts & policy",
        "Forecast",
        "Reports & sharing",
        "Methods & resources",
        "Data admin",
    ], on_change="rerun", key="alpha_research_tabs",
)

with tab_routine:
    if tab_routine.open:
        st.subheader("Routine and infrequent filing addresses")
        st.caption("A retrospective comparison of notice activity, inspired by the supplied chart. Addresses are proxies for properties; these data do not identify verified landlords or completed evictions.")
        routine_controls = st.columns(3)
        threshold = routine_controls[0].number_input("Routine: minimum active months", min_value=2, max_value=36, value=6,
            help="An address is routine if it has notices in at least this many distinct calendar months within the selected period. This is an adjustable research definition, not an official designation.")
        period_label = routine_controls[1].selectbox("Group notices by", ["Year", "Month"])
        measure_label = routine_controls[2].selectbox("Chart measure", ["Notice counts", "Share of notices (%)"])
        routine_summary, routine_addresses = routine_filer_summary(ev_filtered, int(threshold), "Y" if period_label == "Year" else "M")
        if routine_summary.empty:
            st.info("No notice activity matches the current filters.")
        else:
            cohort_sizes = routine_addresses["cohort"].value_counts()
            routine_counts = routine_addresses.groupby("cohort")["notices"].sum()
            cards = st.columns(3)
            cards[0].metric("Routine addresses", f"{cohort_sizes.get('Routine', 0):,}")
            cards[1].metric("Infrequent addresses", f"{cohort_sizes.get('Infrequent', 0):,}")
            cards[2].metric("Routine share of selected notices", f"{100 * routine_counts.get('Routine', 0) / routine_counts.sum():.1f}%")
            measure = "notices" if measure_label == "Notice counts" else "share_percent"
            figure = routine_chart(routine_summary, cohort_sizes, share=measure == "share_percent",
                                   yearly=period_label == "Year", area=selected_area)
            st.plotly_chart(figure, width="stretch")
            st.caption(f"Routine = notices in at least {threshold} separate months across the selected period. Cohort membership is fixed across this chart but recalculates when filters change. Share mode shows each cohort's own percentage; count mode stacks the two groups. The summary card is notice-weighted across all selected periods, not the average annual share. First/last years may be partial. Counts are affected by property size and geocoding coverage; these are not eviction rates.")
            with st.expander("Explore routine addresses and download the calculation"):
                st.dataframe(routine_addresses, hide_index=True, width="stretch")
                st.download_button("Download routine/infrequent time series", routine_summary.to_csv(index=False), "routine_filing_contribution.csv", "text/csv")
                st.download_button("Download address classifications", routine_addresses.to_csv(index=False), "routine_address_classifications.csv", "text/csv")

with tab_property:
    if tab_property.open:
        st.subheader("Address history and parcel profiles")
        address_table = address_profiles(ev_filtered)
        parcel_table = property_profiles(ev_filtered)
        profile_mode = st.radio("Profile by", ["Address", "Parcel / APN"], horizontal=True)
        if profile_mode == "Address":
            choices = address_table["full_address"].tolist()
            chosen = st.selectbox("Address", choices, index=0 if choices else None)
            if chosen:
                history = ev_filtered[ev_filtered["full_address"] == chosen].sort_values("Date_Filed")
                profile = address_table[address_table["full_address"] == chosen].iloc[0]
                cols = st.columns(4)
                cols[0].metric("Filing dates", f"{int(profile['filing_nodes']):,}")
                cols[1].metric("Notice records", f"{int(profile['affected_tenants']):,}")
                cols[2].metric("Reported rent", f"${profile['total_rent_owed']:,.0f}")
                cols[3].metric("APN(s)", profile["apns"][:24])
                timeline_group = st.radio("Timeline grouping", ["Month", "Day"], horizontal=True)
                st.plotly_chart(address_timeline(history, monthly=timeline_group == "Month"), width="stretch")
                st.caption(f"This address has recorded activity from {history['Date_Filed'].min():%b %d, %Y} to {history['Date_Filed'].max():%b %d, %Y} within your filters. A shorter address history does not establish a gap in the full dataset.")
                st.dataframe(history[["Date_Filed", "Eviction_Count", "Cause", "Total_Rent_Owed", "Apn"]], width="stretch")
        else:
            apns = parcel_table["Apn"].tolist()
            chosen_apn = st.selectbox("Parcel / APN", apns, index=0 if apns else None)
            if chosen_apn:
                parcel_rows = ev_filtered[ev_filtered["Apn"] == chosen_apn].sort_values("Date_Filed")
                st.dataframe(parcel_rows[["full_address", "Date_Filed", "Eviction_Count", "Cause", "Total_Rent_Owed"]], width="stretch")

        st.markdown("#### Optional ownership crosswalk")
        st.caption("The eviction source does not contain verified owner names. Upload a CSV with `Apn` and `owner_name`; optional `network_name` links related LLCs.")
        owner_file = st.file_uploader("Owner/APN crosswalk", type="csv", key="owner_crosswalk")
        if owner_file is not None:
            owner_map = pd.read_csv(owner_file, dtype=str)
            if {"Apn", "owner_name"}.issubset(owner_map.columns):
                ownership = ev_filtered.merge(owner_map, on="Apn", how="left")
                group_field = "network_name" if "network_name" in ownership.columns else "owner_name"
                owner_summary = (
                    ownership.dropna(subset=[group_field])
                    .groupby(group_field, as_index=False)
                    .agg(parcels=("Apn", "nunique"), addresses=("full_address", "nunique"), affected_tenants=("Eviction_Count", "sum"), rent_owed=("Total_Rent_Owed", "sum"))
                    .sort_values("affected_tenants", ascending=False)
                )
                st.dataframe(owner_summary, width="stretch")
            else:
                st.error("Crosswalk must contain `Apn` and `owner_name`.")

with tab_compare:
    if tab_compare.open:
        st.subheader("Neighborhood comparison and normalized rates")
        if nbhd.empty:
            st.info("Neighborhood geometry is unavailable.")
        else:
            points = gpd.GeoDataFrame(
                ev_filtered.copy(),
                geometry=gpd.points_from_xy(ev_filtered["long"], ev_filtered["lat"]),
                crs="EPSG:4326",
            )
            area_join = gpd.sjoin(points, nbhd[["name", "geometry"]], how="inner", predicate="within")
            area_metrics = (
                area_join.groupby("name", as_index=False)
                .agg(evictions_total=("Eviction_Count", "sum"), filing_nodes=("full_address", "size"), unique_addresses=("full_address", "nunique"), rent_owed=("Total_Rent_Owed", "sum"))
            )
            if not census_df.empty:
                area_metrics = area_metrics.merge(census_df, on="name", how="left")
            denominators_file = st.file_uploader("Optional denominators CSV (`name`, `renter_households`)", type="csv", key="denominators")
            if denominators_file is not None:
                denominators = pd.read_csv(denominators_file)
                if {"name", "renter_households"}.issubset(denominators.columns):
                    area_metrics = area_metrics.merge(denominators[["name", "renter_households"]], on="name", how="left")
                    area_metrics["notices_per_1000_renter_households"] = area_metrics["evictions_total"] / area_metrics["renter_households"] * 1000
                    rate_metric = "notices_per_1000_renter_households"
                else:
                    st.error("Denominator file needs `name` and `renter_households`.")
                    rate_metric = "evictions_total"
            else:
                rate_metric = "evictions_total"
                st.info("Upload renter-household denominators to calculate comparable rates per 1,000. Raw counts are shown until then.")
            compare_names = st.multiselect("Choose 2–6 neighborhoods", sorted(area_metrics["name"].unique()), max_selections=6)
            compare_df = area_metrics[area_metrics["name"].isin(compare_names)] if compare_names else area_metrics.nlargest(10, rate_metric)
            fig_compare = px.bar(compare_df, y="name", x=rate_metric, orientation="h",
                                 labels={"name": "Neighborhood", rate_metric: "Notice records"})
            readable_neighborhoods(fig_compare, rate=rate_metric != "evictions_total")
            st.plotly_chart(fig_compare, width="stretch")
            st.dataframe(compare_df.sort_values(rate_metric, ascending=False), width="stretch")
            corr = correlation_table(area_metrics)
            if not corr.empty:
                st.markdown("#### Demographic correlation explorer")
                st.caption("Pearson correlations are descriptive, unadjusted, and not causal.")
                with st.expander("How this correlation is calculated"):
                    st.markdown("Each observation is one neighborhood with matched, filtered notice activity. Pearson's r compares its total notice count with each numeric column. Missing pairs are omitted; the table reports the number of paired neighborhoods. At least three nonconstant pairs are required. Values range from −1 (opposite linear movement) to +1 (same-direction linear movement); 0 means no linear relationship. There is no legal or policy standard behind these scores, no significance test, no adjustment for multiple comparisons, and no control for population, rental stock, or other factors. Areas with no matched notices are absent from this comparison. Columns such as filing nodes and unique addresses overlap with the count itself and are not independent demographic explanations.")
                    st.latex(r"r=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2\sum_i(y_i-\bar{y})^2}}")
                corr_display = corr.copy()
                corr_display["metric"] = corr_display["metric"].replace({"unique_addresses": "Unique addresses with notices"})
                st.dataframe(corr_display, hide_index=True, width="stretch")
                st.caption("Unique addresses with notices counts distinct address labels in each neighborhood within the current filters—not rental units, owners, or all neighborhood addresses. Its correlation with notice count partly reflects neighborhood size and exposure. Table row numbers are not correlation values.")

with tab_hotspots:
    if tab_hotspots.open:
        st.subheader("Descriptive grid hotspots")
        precision = st.slider("Grid precision", 2, 3, 2, help="2 ≈ 1 km cells; 3 ≈ 100 m cells.")
        hotspots = grid_hotspots(ev_filtered, precision=precision)
        if not hotspots.empty:
            st.caption("Hotspot z-scores compare grid-cell counts with all cells in the filtered view; they are not spatial-significance tests.")
            fig_hot = px.scatter_map(
                hotspots,
                lat="grid_lat",
                lon="grid_long",
                size="affected_tenants",
                color="hotspot_z",
                hover_data=["unique_addresses", "filing_nodes", "total_rent_owed"],
                color_continuous_scale="RdBu_r",
                zoom=9,
                height=520,
            )
            fig_hot.update_layout(map_style="carto-positron")
            st.plotly_chart(fig_hot, width="stretch")
            st.dataframe(hotspots.head(50), width="stretch")

with tab_alerts:
    if tab_alerts.open:
        st.subheader("Trend anomalies and policy overlays")
        window = st.slider("Baseline months", 2, 12, 3)
        st.caption("Baseline = the mean of the previous selected number of calendar months, excluding the current month. A complete window is required. Alerts flag deviations of at least 2 sample standard deviations; this is a screening rule, not statistical proof or a policy standard. Gaps remain missing; incomplete or filtered months can distort comparisons.")
        anomalies = anomaly_table(ev_filtered, window=window)
        if not anomalies.empty:
            fig_alert = px.line(anomalies, x="month", y=["evictions", "baseline"], markers=True)
            flagged = anomalies[anomalies["is_alert"]]
            if not flagged.empty:
                fig_alert.add_scatter(x=flagged["month"], y=flagged["evictions"], mode="markers", marker={"size": 14, "color": "red"}, name="±2 SD alert")
            policy_file = st.file_uploader("Optional policy timeline CSV (`date`, `label`)", type="csv", key="policy_timeline")
            if policy_file is not None:
                policy = pd.read_csv(policy_file)
                if {"date", "label"}.issubset(policy.columns):
                    policy["date"] = pd.to_datetime(policy["date"], errors="coerce")
                    for event in policy.dropna(subset=["date"]).itertuples():
                        fig_alert.add_vline(x=event.date.timestamp() * 1000, line_dash="dot", annotation_text=str(event.label)[:40])
                else:
                    st.error("Policy timeline needs `date` and `label`.")
            st.plotly_chart(fig_alert, width="stretch")
            st.dataframe(anomalies.sort_values("z_score", key=abs, ascending=False).head(20), width="stretch")
        search_definition = {
            "area": selected_area,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "causes": selected_causes,
            "districts": selected_districts,
            "zips": selected_zips,
            "zori_month": selected_zori_month,
            "apn": apn_query,
            "address": address_query,
            "minimum_repeat_filings": repeat_min,
        }
        st.download_button("Save search / alert definition", json.dumps(search_definition, indent=2), "evictorbookla_saved_search.json", "application/json")
        st.caption("Automated email/SMS delivery requires a scheduler and messaging backend; this export is ready for one.")

with tab_forecast:
    if tab_forecast.open:
        st.subheader("Short-term filing forecast")
        horizon = st.slider("Forecast months", 3, 18, 6)
        forecast, diagnostics = forecast_monthly(ev_filtered, months=horizon)
        if forecast.empty:
            st.info(diagnostics.get("method", "At least four months of history are required."))
        else:
            historic = anomaly_table(ev_filtered)[["month", "evictions"]]
            fig_forecast = go.Figure()
            fig_forecast.add_scatter(x=historic["month"], y=historic["evictions"], mode="lines+markers", name="Observed")
            fig_forecast.add_scatter(x=forecast["month"], y=forecast["upper"], mode="lines", line={"width": 0}, showlegend=False)
            fig_forecast.add_scatter(x=forecast["month"], y=forecast["lower"], mode="lines", fill="tonexty", fillcolor="rgba(59,111,182,0.12)", line={"width": 0}, name="95% prediction interval (future count)")
            fig_forecast.add_scatter(x=forecast["month"], y=forecast["ci_upper_95"], mode="lines", line={"width": 0}, showlegend=False)
            fig_forecast.add_scatter(x=forecast["month"], y=forecast["ci_lower_95"], mode="lines", fill="tonexty", fillcolor="rgba(59,111,182,0.30)", line={"width": 0}, name="95% confidence interval (mean count)")
            fig_forecast.add_scatter(x=forecast["month"], y=forecast["forecast"], mode="lines+markers", name="Linear baseline",
                                    customdata=forecast[["ci_lower_95", "ci_upper_95", "lower", "upper"]],
                                    hovertemplate="%{x|%b %Y}<br>Forecast: %{y:,.0f}<br>95% mean CI: %{customdata[0]:,.0f}–%{customdata[1]:,.0f}<br>95% prediction interval: %{customdata[2]:,.0f}–%{customdata[3]:,.0f}<extra></extra>")
            st.plotly_chart(fig_forecast, width="stretch")
            st.metric("Holdout mean absolute error", f"{diagnostics['mae']:.1f} notices")
            st.warning("This is a transparent planning baseline, not a causal forecast. Sudden policy or reporting changes are not modeled.")
            st.caption("The darker 95% confidence band estimates the mean monthly count; the wider 95% prediction band allows for variation in a future monthly count. Both use OLS residual variance, forecast leverage, and a Student-t critical value with n−2 degrees of freedom. They assume a correct linear trend and independent, normally distributed errors with constant variance. Serial correlation, seasonality, incomplete months, reporting delays and policy changes can invalidate coverage. Negative values are truncated at zero; nominal 95% coverage has not been independently validated.")
            with st.expander("Monthly forecasts and intervals"):
                displayed_forecast = forecast.rename(columns={"forecast":"Forecast notices", "ci_lower_95":"95% mean CI lower", "ci_upper_95":"95% mean CI upper", "lower":"95% prediction lower", "upper":"95% prediction upper"})
                st.dataframe(displayed_forecast, hide_index=True, width="stretch")
                exported_forecast = displayed_forecast.assign(confidence_level=.95, interval_method="OLS Student-t, iid normal errors; count bounds truncated at zero")
                st.download_button("Download monthly forecasts with intervals", exported_forecast.to_csv(index=False), "monthly_notice_forecasts.csv", "text/csv")

with tab_reports:
    if tab_reports.open:
        st.subheader("Share, export, and print")
        report_bytes = report_html(ev_filtered, selected_area, f"{start_date} to {end_date}")
        col_a, col_b, col_c = st.columns(3)
        col_a.download_button("Printable report (HTML/PDF)", report_bytes, "evictorbookla_report.html", "text/html")
        col_b.download_button("Filtered GeoJSON", selection_geojson(ev_filtered), "evictorbookla_filtered.geojson", "application/geo+json")
        col_c.download_button("Filtered CSV", ev_filtered.to_csv(index=False), "evictorbookla_filtered.csv", "text/csv")
        share_state = json.dumps(search_definition)
        embed_html(
            f"""<button id='copy-view' style='padding:8px 12px'>Copy shareable view URL</button>
            <span id='copy-status' aria-live='polite'></span><script>
            document.getElementById('copy-view').onclick = async () => {{
              const url = new URL(window.parent.location.href);
              url.searchParams.set('view', btoa(unescape(encodeURIComponent(JSON.stringify({share_state})))));
              await navigator.clipboard.writeText(url.toString());
              document.getElementById('copy-status').textContent = ' Copied';
            }};</script>""",
            height=45,
        )
        st.caption("The URL stores the current filter definition. Polygon IDs remain in the existing `sel` parameter.")
        st.markdown("#### Embed")
        st.code('<iframe src="YOUR_DEPLOYED_APP_URL/?embed=true" width="100%" height="900" loading="lazy" title="EvictorBookLA Atlas map"></iframe>', language="html")

with tab_method:
    if tab_method.open:
        try:
            st.page_link("math_page.py", label="All ALPHA and BETA mathematical methods", icon="📐")
        except st.errors.StreamlitAPIException:
            # The original app.py entry point has no multipage navigation registry.
            # Keep the complete reference accessible when ALPHA runs standalone.
            with st.expander("All ALPHA and BETA mathematical methods"):
                from methods_ui import render_mathematical_methods
                render_mathematical_methods()
        st.subheader("Definitions, provenance, limitations, and tenant resources")
        with st.expander("Full metric guide and data dictionary"):
            guide = Path(__file__).with_name("ANALYTICS_GUIDE.md")
            if guide.exists():
                st.markdown(guide.read_text())
        coverage_start = evictions["Date_Filed"].min().date()
        coverage_end = evictions["Date_Filed"].max().date()
        source_rows = len(pd.read_csv(EVICTION_CSV, usecols=["row_id"]))
        zori_coverage = (
            f"{pd.to_datetime(zori_month_columns[0]).strftime('%B %Y')} to "
            f"{pd.to_datetime(zori_month_columns[-1]).strftime('%B %Y')}"
            if zori_month_columns
            else "Unavailable"
        )
        quality = pd.DataFrame(
            {
                "Item": [
                    "Eviction source",
                    "Eviction coverage",
                    "Raw eviction rows",
                    "Mapped address-date nodes",
                    "Missing coordinates after cleaning",
                    "Eviction file update",
                    "Asking-rent source",
                    "ZORI coverage",
                    "ZIP overlay geometry",
                ],
                "Value": [
                    "Los Angeles Housing Department eviction notices",
                    f"{coverage_start} to {coverage_end}",
                    f"{source_rows:,}",
                    f"{len(evictions):,}",
                    "Excluded from mapped analysis",
                    pd.Timestamp(os.path.getmtime(EVICTION_CSV), unit="s").strftime("%Y-%m-%d %H:%M"),
                    "Zillow Observed Rent Index (ZIP-level, smoothed)",
                    zori_coverage,
                    "U.S. Census Bureau 2020 generalized ZCTAs",
                ],
            }
        )
        st.dataframe(quality, hide_index=True, width="stretch")
        with st.expander("Monthly source coverage — including unmapped notices", expanded=True):
            coverage_raw = pd.read_csv(EVICTION_CSV, usecols=["Date Received", "lat", "long"])
            coverage_raw["month"] = pd.to_datetime(coverage_raw["Date Received"], format="mixed", errors="coerce").dt.to_period("M").dt.to_timestamp()
            coverage_raw["mapped"] = coverage_raw[["lat", "long"]].notna().all(axis=1)
            coverage = coverage_raw.groupby("month").agg(source_notices=("mapped", "size"), mapped_notices=("mapped", "sum"))
            if not coverage.empty:
                coverage = coverage.reindex(pd.date_range(coverage.index.min(), coverage.index.max(), freq="MS")).rename_axis("month").reset_index()
                coverage["unmapped_notices"] = coverage["source_notices"] - coverage["mapped_notices"]
                coverage["mapped_percent"] = 100 * coverage["mapped_notices"] / coverage["source_notices"]
                st.plotly_chart(px.line(coverage, x="month", y=["source_notices", "mapped_notices"], markers=True), width="stretch")
                st.dataframe(coverage, hide_index=True, width="stretch")
                st.download_button("Download monthly coverage", coverage.to_csv(index=False), "lahd_monthly_coverage.csv", "text/csv")
            st.caption("All locally imported LAHD notices by Date Received, independent of map filters. Missing coordinates reduce mapped counts; they do not remove records from this source-coverage view. An absent month is shown as missing, never invented or interpolated. The latest month may be incomplete.")
        if METADATA_PATH.exists():
            refresh_meta = json.loads(METADATA_PATH.read_text())
            st.markdown("#### Live-source status")
            status_cols = st.columns(4)
            status_cols[0].metric("Last checked", pd.to_datetime(refresh_meta.get("checked_at") or refresh_meta["refreshed_at"]).strftime("%Y-%m-%d %H:%M UTC"))
            status_cols[1].metric("Latest notice received", refresh_meta["checks"]["latest_received"])
            status_cols[2].metric("Source rows", f"{refresh_meta['checks']['rows']:,}")
            status_cols[3].metric("Mapped rows", f"{refresh_meta['checks']['mapped_percent']:.1f}%")
            st.caption(f"Published file timestamp: {refresh_meta['source_csv'].get('last_modified') or 'not supplied'}")
        st.markdown(
            """
            **Core definitions.** A filing node is one address-date combination. `Date_Filed` is LAHD's Date Received. `Eviction_Count` and legacy “Affected tenants” fields count notice records, not people or completed evictions. `Total_Rent_Owed` sums reported amounts within a node; these amounts are not independently verified. Repeat filing counts are filing dates per address. APN identifies a parcel, not a verified owner. Zillow ZORI estimates typical market asking rent for a ZIP area; it is not a tenant's contract rent.

            **Rent comparison math.** For month *t*, average asking rent is the arithmetic mean of available ZORI values across ZIP codes represented by the active eviction filters: `mean(ZORI_zip,t)`. Average rent owed is the arithmetic mean of positive LAHD `Rent Owed` values on matched notice records filed in month *t*. Missing Zillow ZIP-month observations remain missing and are not converted to zero. The ZORI aggregation is unweighted because the supplied file does not include rental-stock weights.

            **Monthly forecast uncertainty.** Ordinary least squares fits notice count to month index. For n months, residual variance is `s² = SSE / (n−2)` and future leverage is `h = 1/n + (x−mean(x))² / sum((x−mean(x))²)`. The pointwise 95% confidence interval for the mean is `forecast ± t(.975,n−2) × sqrt(s²h)`; the wider 95% prediction interval for a future count is `forecast ± t(.975,n−2) × sqrt(s²(1+h))`. Bounds are truncated at zero. Coverage assumes independent, normal, constant-variance errors and an adequate linear trend; real-world 95% coverage has not been verified. See BETA's Methods for separate address-probability intervals.

            **Map method.** The asking-rent overlay joins the selected ZORI month to 2020 Census ZIP Code Tabulation Area (ZCTA) polygons by five-digit code. The color scale uses the 10th to 90th percentile of visible ZIP values so high-rent outliers do not flatten the rest of the map; hover details retain the exact value. ZCTAs approximate ZIP service areas and are not USPS delivery-route boundaries.

            **Important limitations.** Notices are not court judgments or proof that a household physically moved. Address geocoding, duplicate records, missing notices, boundary assignment, and changing reporting practices can affect results. Owner analysis requires an external verified APN crosswalk. Neighborhood rates require renter-household denominators.

            **Data sources.** [Zillow Research ZORI data](https://www.zillow.com/research/data/) · [Census ZCTA definitions](https://www.census.gov/programs-surveys/geography/guidance/geo-areas/zctas.html) · [Census TIGERweb boundary service](https://tigerweb.geo.census.gov/arcgis/rest/services/Generalized_TAB2020/PUMA_TAD_TAZ_UGA_ZCTA/MapServer/3)

            **Tenant resources.** [LAHD renter resources](https://housing.lacity.gov/residents/renters) · [Stay Housed LA](https://www.stayhousedla.org/) · [211 LA](https://211la.org/)
            """
        )

with tab_admin:
    if tab_admin.open:
        st.subheader("LAHD live-data refresh")
        st.markdown(f"Source: [LAHD published eviction CSV]({SOURCE_CSV})")
        st.caption("While this dashboard is open, an hourly timer checks whether the 24-hour refresh interval has elapsed. CSV and Excel source files are accepted. Closing the app stops these checks. Set `LAHD_AUTO_REFRESH_HOURS=0` to disable or another number to change the interval.")
        refresh_cols = st.columns(2)
        if refresh_cols[0].button("Check source now", type="secondary"):
            try:
                with st.spinner("Downloading and validating LAHD data…"):
                    check_result = refresh_lahd_data(Path(EVICTION_CSV), dry_run=True, geocode=False)
                if check_result.get("status") == "retained_newer_local_data":
                    st.info(check_result["message"])
                else:
                    st.success("Source passed validation; no local files were changed.")
                st.json(check_result)
            except Exception as error:
                st.error(f"Source check failed; saved data retained. {error}")
        if refresh_cols[1].button("Update from LAHD now", type="primary"):
            try:
                with st.spinner("Updating from LAHD and preserving a rollback copy…"):
                    update_result = refresh_lahd_data(Path(EVICTION_CSV), dry_run=False, geocode=False)
                st.cache_data.clear()
                if update_result.get("status") == "retained_newer_local_data":
                    st.info(update_result["message"])
                else:
                    st.success(f"Updated to {update_result['checks']['latest_received']} with {update_result['checks']['rows']:,} rows.")
                st.rerun()
            except Exception as error:
                st.error(f"Refresh failed; saved data retained. {error}")
        st.info("The direct LAHD CSV does not include coordinates. Existing address/APN coordinates and the geocode cache are reused; unmatched records remain in the source file but are excluded from map-only analysis.")

        st.markdown("#### Validate a manually downloaded dataset")
        upload = st.file_uploader("Upload CSV for validation", type="csv", key="data_refresh")
        if upload is not None:
            candidate = pd.read_csv(upload)
            required = {"full_address", "lat", "long", "Date Received", "Cause", "Apn", "Council District", "Rent Owed"}
            missing = sorted(required - set(candidate.columns))
            checks = {
                "Rows": len(candidate),
                "Columns": len(candidate.columns),
                "Missing required columns": ", ".join(missing) or "None",
                "Missing coordinates": int(candidate[["lat", "long"]].isna().any(axis=1).sum()) if {"lat", "long"}.issubset(candidate.columns) else "N/A",
                "Duplicate rows": int(candidate.duplicated().sum()),
            }
            st.json(checks)
            if not missing:
                st.success("Schema validation passed. Review the preview before replacing the production file.")
                st.dataframe(candidate.head(100), width="stretch")
            else:
                st.error("Schema validation failed; the production file was not changed.")
