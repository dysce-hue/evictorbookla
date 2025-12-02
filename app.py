import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

# ==============================
# CONFIG
# ==============================
MAPBOX_TOKEN = "pk.eyJ1IjoiZHlsYW5nMjEzIiwiYSI6ImNtYnR1dXY5bTA3ZTcybXExbjYzYWRndHEifQ.ZCO0qTR3GITmJYJn0glaxg"
DATA_DIR = "data"
EVICTION_CSV = os.path.join(DATA_DIR, "cleaned_evictions_final.csv")

# Shapefiles (same as Shiny)
SUBDIV_SHP = os.path.join(DATA_DIR, "County_Subdivisions.shp")
NBHD_SHP = os.path.join(DATA_DIR, "8494cd42-db48-4af1-a215-a2c8f61e96a22020328-1-621do0.x5yiu.shp")

# Optional pre-aggregated census file
NEIGHBORHOOD_CENSUS_CSV = os.path.join(DATA_DIR, "neighborhood_census.csv")

# FRED CPI config 
FRED_API_KEY = "e96136ab46ba35fca84c640ca9873d86"
FRED_SERIES_ID = "CUSR0000SA0L2"  # rent of primary residence

# ==============================
# DATA LOADING — EVICTIONS
# ==============================
@st.cache_data
def load_evictions(path):
    if not os.path.exists(path):
        st.error(f"Eviction CSV not found at: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    expected_cols = [
        "Council District", "Rent Owed", "Date Received",
        "full_address", "long", "lat", "Cause", "Apn"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns in CSV: {missing}")

    df = df.copy()

    # Coerce stringish columns & handle NAs
    for col in ["Council District", "full_address", "Cause", "Apn"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

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
            df["Date Received"], errors="coerce", format="%m-%d-%Y"
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
    agg["Apn"] = agg["Apn"].fillna("")

    return agg

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
        nbhd_union = nbhd.unary_union
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
@st.cache_data
def load_cpi():
    """Load CPI data from FRED, same series as Shiny (rent of primary residence)."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": FRED_SERIES_ID,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "2015-01-01",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]].dropna()
    except Exception as e:
        st.warning(f"Could not load CPI data from FRED: {e}")
        return pd.DataFrame(columns=["date", "value"])

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
    page_title="EvictorBookLA 3D (WIP)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .creator-badge {
        position: fixed;
        left: 12px;
        bottom: -10px;
        z-index: 9999;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.9);  /* tweak if you want darker */
        color: #0f172a;                         /* text color */
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
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

evictions = load_evictions(EVICTION_CSV)
subdiv = load_subdivisions()
nbhd = load_neighborhoods()
subdiv_minus = compute_subdiv_minus_neighborhood()
cpi_df = load_cpi()
census_df = load_neighborhood_census()

# ==============================
# BUILD NEIGHBORHOOD CHOROPLETH GEOJSON
# ==============================
nbhd_choro_json = "null"

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

st.markdown(
    "<h1 style='font-size: 2.5rem; font-weight: 700;'>EvictorBookLA — 3D Map (WIP)</h1>",
    unsafe_allow_html=True,
)
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
    width=80,         
    use_column_width=False
)
st.sidebar.header("EvictorBookLA")




min_date = evictions["Date_Filed"].min()
max_date = evictions["Date_Filed"].max()

date_range = st.sidebar.date_input(
    "Eviction Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

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
    index=0,
)

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
        left: 10px;
        bottom: 10px;
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
ev_filtered = filter_by_area(ev_filtered, selected_area, subdiv, nbhd, subdiv_minus)

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

# Stable ID for JS ↔ Python linkage
ev_filtered = ev_filtered.reset_index(drop=True).copy()
ev_filtered["js_id"] = ev_filtered.index.astype(int)

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
    ev_selected = ev_filtered[ev_filtered["js_id"].isin(selected_ids)].copy()
else:
    ev_selected = pd.DataFrame(columns=ev_filtered.columns)

# ==============================
# PREP DATA FOR JS
# ==============================
ev_js = ev_filtered.copy()
ev_js["Date_Filed_str"] = ev_js["Date_Filed"].dt.strftime("%m-%d-%Y")

records = ev_js[
    [
        "js_id",
        "full_address",
        "Date_Filed_str",
        "Eviction_Count",
        "Cause",
        "Council_District",
        "Total_Rent_Owed",
        "Apn",
        "lat",
        "long",
    ]
].to_dict(orient="records")

evictions_json = json.dumps(records)

# --- Neighborhood GeoJSON for base overlay ---
if not nbhd.empty:
    nbhd_export = nbhd[["name", "geometry"]].copy()
    nbhd_geojson = nbhd_export.to_json()
else:
    nbhd_geojson = json.dumps({"type": "FeatureCollection", "features": []})

# ==============================
# EMBEDDED HTML: MAPBOX GL JS + DRAW
# ==============================
html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>EvictorBookLA 3D</title>
<meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />

<!-- Mapbox GL core (v2.15.0 – stable with Draw) -->
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

<!-- Address search UI -->
<div id="search-container">
  <input id="address-input" type="text" placeholder="Search address (from dataset)..." />
  <button id="address-search-btn" type="button">Go</button>
</div>

<!-- Choropleth overlay controls -->
<div id="choropleth-controls">
  <label for="choropleth-select">Overlay:</label>
  <select id="choropleth-select">
    <option value="none" selected>No overlay</option>
    <option value="evictions_total">Total evictions (all years)</option>
    <option value="median_income">Median income</option>
    <option value="rent_burden_pct">Severe rent burden ≥50%</option>
  </select>
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
  <ol id="selection-list"></ol>
</div>

<!-- Neighborhood hover info panel (bottom-left) -->
<div id="nbhd-info-panel">
  <div id="nbhd-info-content">
    Hover over a shaded neighborhood to view total evictions, median income, and severe rent burden.
  </div>
</div>

<script>
  mapboxgl.accessToken = "MAPBOX_TOKEN_PLACEHOLDER";

  // Eviction data from Streamlit
  const evictions = EVIC_DATA_PLACEHOLDER;

  // Neighborhood choropleth data (census + CPI-adjusted income, etc.)
  const nbhdChoro = NBHD_CHORO_PLACEHOLDER;

  // Neighborhood polygons from Streamlit (basic outline)
  const neighborhoods = NBHD_DATA_PLACEHOLDER;

  // Build GeoJSON with clustering enabled
  // Build GeoJSON with slight jitter for identical coordinates
const jitterTracker = {};
const evictionFeatures = evictions.map((e) => {
  const baseLng = e.long;
  const baseLat = e.lat;

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
      id: e.js_id,
      full_address: e.full_address,
      date_filed: e.Date_Filed_str,
      eviction_count: e.Eviction_Count,
      cause: e.Cause,
      council_district: e.Council_District,
      total_rent_owed: e.Total_Rent_Owed,
      apn: e.Apn
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

  // ==========================
  // Mapbox Draw control
  // ==========================
  let draw = null;
  try {
    if (typeof MapboxDraw !== "undefined") {
      draw = new MapboxDraw({
        displayControlsDefault: false,
        controls: {
          polygon: true,
          trash: true
        },
        defaultMode: "simple_select"
      });
      map.addControl(draw, "top-left");
    }
  } catch (e) {
    console.error("MapboxDraw failed to initialize:", e);
  }

  map.addControl(new mapboxgl.NavigationControl(), "top-right");

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
      <b>Approx Tenants Evicted (sum of Tenant Evictions):</b> ${unitCount.toLocaleString()}<br/>
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

    const polygon = data.features[data.features.length - 1];

    const selectedFeatures = evictionGeojson.features.filter(f =>
      turf.booleanPointInPolygon(f, polygon)
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

    const html = `
      <div style="font-size:12px;line-height:1.4">
        <strong>Address:</strong> ${props.full_address}<br/>
        <strong>Date Filed:</strong> ${props.date_filed}<br/>
        <strong>Tenants Evicted:</strong> ${props.eviction_count}<br/>
        <strong>Cause:</strong> ${props.cause || "N/A"}<br/>
        <strong>Council District:</strong> ${props.council_district || "N/A"}<br/>
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
    const nbhdInfoContent = document.getElementById("nbhd-info-content");

    function setNbhdInfo(html) {
      if (nbhdInfoContent) {
        nbhdInfoContent.innerHTML = html;
      }
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
          `Severe rent burden (≥50%): ${rbStr}`
        );
      });

      map.on("mouseleave", layerId, () => {
        setNbhdInfo(
          "Hover over a shaded neighborhood to view total evictions, median income, and severe rent burden."
        );
      });
    }


    // 3D BUILDINGS LAYER
    const layers = map.getStyle().layers;
    const labelLayer = layers.find(
      (layer) => layer.type === "symbol" && layer.layout && layer.layout["text-field"]
    );
    const labelLayerId = labelLayer ? labelLayer.id : undefined;

    map.addLayer(
      {
        "id": "add-3d-buildings",
        "source": "composite",
        "source-layer": "building",
        "filter": ["==", "extrude", "true"],
        "type": "fill-extrusion",
        "minzoom": 15,
        "paint": {
          "fill-extrusion-color": "#aaa",
          "fill-extrusion-height": [
            "interpolate",
            ["linear"],
            ["zoom"],
            15, 0,
            15.05, ["get", "height"]
          ],
          "fill-extrusion-base": [
            "interpolate",
            ["linear"],
            ["zoom"],
            15, 0,
            15.05, ["get", "min_height"]
          ],
          "fill-extrusion-opacity": 0.6
        }
      },
      labelLayerId
    );

        // NEIGHBORHOOD OUTLINE OVERLAY (start hidden)
    if (neighborhoods && neighborhoods.type === "FeatureCollection") {
      map.addSource("neighborhoods", {
        type: "geojson",
        data: neighborhoods
      });

      map.addLayer({
        id: "neighborhood-fill",
        type: "fill",
        source: "neighborhoods",
        paint: {
          "fill-color": "#3182bd",
          "fill-opacity": 0.12
        },
        layout: {
          "visibility": "none"
        }
      });

      map.addLayer({
        id: "neighborhood-outline",
        type: "line",
        source: "neighborhoods",
        paint: {
          "line-color": "#08519c",
          "line-width": 2.5
        },
        layout: {
          "visibility": "none"
        }
      });
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

      const html = `
        <div style="font-size:12px;line-height:1.4">
          <strong>Address:</strong> ${props.full_address}<br/>
          <strong>Date Filed:</strong> ${props.date_filed}<br/>
          <strong>Number of Tenants Evicted:</strong> ${props.eviction_count}<br/>
          <strong>Cause:</strong> ${props.cause || "N/A"}<br/>
          <strong>Council District:</strong> ${props.council_district || "N/A"}<br/>
          <strong>Total Rent Owed:</strong> $${rentFormatted}<br/>
          <strong>APN:</strong> ${props.apn || "N/A"}
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
    // Neighborhood choropleth overlays
    // ==========================
    if (nbhdChoro && nbhdChoro.type === "FeatureCollection" && nbhdChoro.features && nbhdChoro.features.length) {
      map.addSource("nbhd-choropleth-src", {
        type: "geojson",
        data: nbhdChoro
      });
    }

    function updateChoropleth(propertyName) {
      const src = map.getSource("nbhd-choropleth-src");
      if (!src) return;

      // Remove previous layer if present
      if (map.getLayer("nbhd-choropleth-layer")) {
        map.removeLayer("nbhd-choropleth-layer");
      }

      if (!propertyName || propertyName === "none") return;

      const features = (nbhdChoro && nbhdChoro.features) ? nbhdChoro.features : [];
      const values = features
        .map(f => Number(f.properties[propertyName]))
        .filter(v => !isNaN(v) && isFinite(v));

      if (!values.length) return;

      const min = Math.min(...values);
      const max = Math.max(...values);
      if (!isFinite(min) || !isFinite(max) || min === max) return;

      const mid = (min + max) / 2;

      map.addLayer(
        {
          id: "nbhd-choropleth-layer",
          type: "fill",
          source: "nbhd-choropleth-src",
          paint: {
            "fill-color": [
              "interpolate",
              ["linear"],
              ["get", propertyName],
              min, "#eff3ff",   // very light blue
              mid, "#6baed6",   // medium blue
              max, "#08306b"  
            ],
            "fill-opacity": 0.55,
            "fill-outline-color": "#555555"
          }
        },
        "clusters"  // draw beneath eviction points
      );

      // Enable hover info for this layer
      bindNeighborhoodHover("nbhd-choropleth-layer");
    }

    const choroSelect = document.getElementById("choropleth-select");
    if (choroSelect) {
      choroSelect.addEventListener("change", (e) => {
        updateChoropleth(e.target.value);
      });
    }
  });
</script>
</body>
</html>
"""

# Inject JSON + config into template
html_final = (
    html_template
    .replace("EVIC_DATA_PLACEHOLDER", evictions_json)
    .replace("MAPBOX_TOKEN_PLACEHOLDER", MAPBOX_TOKEN)
    .replace("NBHD_CHORO_PLACEHOLDER", nbhd_choro_json)
    .replace("NBHD_DATA_PLACEHOLDER", nbhd_geojson)
)

# ==============================
# LAYOUT: MAP FULL-WIDTH + ANALYTICS BELOW
# ==============================
st.markdown("### 3D Eviction Map")
components.html(html_final, height=720, scrolling=False)

st.markdown("---")
st.markdown("### Summary & Download")

# --- Full filtered metrics ---
sum_cols = st.columns(4)
total_units = int(ev_filtered["Eviction_Count"].sum())
total_addresses = ev_filtered["full_address"].nunique()
total_rent = float(ev_filtered["Total_Rent_Owed"].sum())

with sum_cols[0]:
    st.metric("Total Evicted Tenants (filtered)", f"{total_units:,}")
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

# ==============================
# ANALYTICS TABS 
# ==============================
tab_trend, tab_cpi, tab_hist, tab_causes, tab_demo = st.tabs(
    [
        " Eviction Trend",
        " CPI & Rent Index",
        " Rent Owed Histogram",
        " Eviction Causes",
        " Demographics (Census Data)",
    ]
)

# --- Eviction Trend (monthly) ---
with tab_trend:
    st.subheader("Monthly Evictions")
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
            labels={"month": "Month", "evictions": "Number of Evictions"},
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
            yaxis_title="Number of Evictions",
            yaxis_tickformat=",.0f",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="trend_chart",
            on_select=_update_trend_filter,
            selection_mode="points",
        )
        if st.session_state.get("trend_filter") is not None:
            st.caption("Eviction trend filter is active (use 'Clear Chart-Based Filters' in the sidebar to reset).")

# --- CPI & Rent Index (FRED) ---
with tab_cpi:
    st.subheader("CPI (Rent Index) over eviction date range")
    if cpi_df.empty:
        st.info("CPI data could not be loaded from FRED.")
    else:
        date_min = ev_filtered["Date_Filed"].min()
        date_max = ev_filtered["Date_Filed"].max()
        sub_cpi = cpi_df[
            (cpi_df["date"] >= date_min) & (cpi_df["date"] <= date_max)
        ]
        if sub_cpi.empty:
            st.info("No CPI observations in this date range.")
        else:
            fig = px.line(
                sub_cpi,
                x="date",
                y="value",
                markers=True,
                labels={"date": "Date", "value": "CPI (Rent Index)"},
            )
            
            fig.update_traces(
                mode="lines+markers+text",
                text=sub_cpi["value"].round(1),
                textposition="top center",
            )
            fig.update_layout(
                height=430,
                margin=dict(l=40, r=20, t=60, b=40),
                yaxis_title="CPI (Rent Index)",
                yaxis_tickformat=",.1f",
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Rent Owed Histogram (95% cutoff) ---
with tab_hist:
    st.subheader("Distribution of Rent Owed (excluding top 5% outliers)")
    st.caption("Use Box Select in top right of the graph to filter for selected units")
    df_hist = ev_filtered.copy()
    df_hist = df_hist[df_hist["Total_Rent_Owed"] > 0]

    if df_hist.empty:
        st.info("No positive rent-owed values to display.")
    else:
        cutoff = df_hist["Total_Rent_Owed"].quantile(0.95)
        df_hist = df_hist[df_hist["Total_Rent_Owed"] <= cutoff]

        fig = px.histogram(
            df_hist,
            x="Total_Rent_Owed",
            nbins=40,
            labels={"Total_Rent_Owed": "Total Rent Owed ($)"},
        )
        
        fig.update_traces(
            texttemplate="%{y}",
            textposition="outside",
        )
        fig.update_layout(
            height=430,
            margin=dict(l=70, r=20, t=70, b=60),
            xaxis_tickformat="$,.0f",
            yaxis_title="Number of Separate Eviction Addresses",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="rent_hist_chart",
            on_select=_update_rent_filter,
            selection_mode="box",
        )
        if st.session_state.get("rent_filter") is not None:
            st.caption("Rent owed histogram filter is active (use 'Clear Chart-Based Filters' in the sidebar to reset).")

# --- Eviction Causes Pie ---
with tab_causes:
    st.subheader("Eviction Cause Breakdown (filtered set)")
    df_causes = ev_filtered.copy()
    df_causes = df_causes[
        (df_causes["Cause"].notna()) & (df_causes["Cause"].str.strip() != "")
    ]

    if df_causes.empty:
        st.info("No non-empty Cause values to display.")
    else:
        cause_sum = (
            df_causes.groupby("Cause", as_index=False)["Eviction_Count"]
            .sum()
            .rename(columns={"Eviction_Count": "total_evictions"})
            .sort_values("total_evictions", ascending=False)
        )

        top_n = 9
        if len(cause_sum) > top_n:
            top = cause_sum.iloc[:top_n].copy()
            other = cause_sum.iloc[top_n:].copy()
            other_row = pd.DataFrame(
                {
                    "Cause": ["Other"],
                    "total_evictions": [other["total_evictions"].sum()],
                }
            )
            cause_sum = pd.concat([top, other_row], ignore_index=True)

        fig = px.pie(
            cause_sum,
            names="Cause",
            values="total_evictions",
            hole=0.3,
        )
        fig.update_traces(textposition="inside", textinfo="label+percent")
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)

# --- Demographics (Census) ---
with tab_demo:
    st.subheader("Neighborhood Demographic Based on Census Data (Irrespective of Evictions)")

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
                    "Severely Rent-Burdened (≥50%) *paying over 50% of income on rent*",
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
            st.plotly_chart(fig_demo, use_container_width=True)
        else:
            st.info("No racial composition columns found in census file.")
