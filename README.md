# EVICTOR BOOK ALPHA and EVICTOR BOOK BETA

A pair of local, interactive Los Angeles eviction-notice applications that share one validated data file.

- **EVICTOR BOOK ALPHA** (`app.py`) preserves the primary 3D mapping and research experience.
- **EVICTOR BOOK BETA** (`app_beta.py`) is a separate policy-visualization workspace for time patterns, concentration, cause composition, recurrence, persistent hotspots, hotspot address inspection, a transparent building-risk portfolio, an experimental self-updating 90-day notice-recurrence forecast, complete mathematical methods, and data quality.

## Included features

- Cluster, individual-point, heatmap, 3D-building, and neighborhood choropleth map modes
- Monthly map animation, address search, editable multi-polygon selection, and CSV/GeoJSON export
- Date, cause, council district, ZIP, APN, address, repeat-filing, rent, and neighborhood filters
- Address timelines, parcel profiles, and optional owner/LLC network crosswalks
- Neighborhood comparison with optional rates per 1,000 renter households
- Demographic correlation explorer, grid hotspots, anomaly alerts, policy-event overlays, and a backtested forecast baseline
- Saved-search definitions, shareable filter state, printable HTML/PDF reports, and embeddable Streamlit views
- English interface, accessible focus styles, reduced-motion handling, and mobile layout improvements
- Data provenance, limitations, tenant-resource links, and refresh-file schema validation

Optional enrichment templates live in `data_templates/`. These inputs are never silently treated as source data: owner names, policy events, and renter-household denominators must be supplied and reviewed by the user.

The Beta forecast is intentionally limited to addresses already observed during the preceding 365 days. It uses chronological holdout testing, compares its ridge-logistic candidate with a recent-history baseline, and rebuilds when the validated source CSV changes. It remains a shadow-mode research and outreach aid; it does not predict completed evictions or establish wrongdoing.

## Local setup

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Configure the required Mapbox token and optional FRED API key using environment variables:

   ```bash
   export MAPBOX_TOKEN="your-mapbox-public-token"
   export FRED_API_KEY="your-fred-api-key"
   ```

   Streamlit secrets are also supported. Create `.streamlit/secrets.toml` locally:

   ```toml
   MAPBOX_TOKEN = "your-mapbox-public-token"
   FRED_API_KEY = "your-fred-api-key"
   ```

4. Run the combined site:

   ```bash
   .venv/bin/streamlit run site.py --server.address 127.0.0.1 --server.port 8501
   ```

Open http://127.0.0.1:8501/ and use the top navigation to switch between ALPHA's 3D map and BETA's policy tools. BETA is also available at /policy on the same server. Its address inspector opens the ALPHA map on this same site with the selected address and period.

The original entry point remains supported: `streamlit run app.py` runs ALPHA alone, with the complete mathematical reference inside its Methods tab. Use `site.py` for the combined ALPHA/BETA navigation and the dedicated `/methods` page.

The map requires `MAPBOX_TOKEN`. FRED uses the national rent-of-primary-residence index `CUSR0000SEHA`. If a valid 32-character `FRED_API_KEY` is unavailable or rejected, the app tries FRED's public CSV. Error messages never include the credential or authenticated URL.

## Live LAHD data refresh

Startup uses the saved local dataset immediately. The first automatic LAHD check is deferred for one hour, so a slow external download does not block the initial page. Manual refresh remains in Data admin. The optional FRED download is enabled separately in the sidebar.

## Loading performance

Notice aggregates, notice-level rent records, Zillow data and ZIP boundaries use bounded disk caches keyed by their source file's modification time. Tabs render their contents only when selected. Subdivision-minus-neighborhood geometry is calculated only for the corresponding area filter. These changes reduce runtime work, not dependency installation; Community Cloud may discard disk caches during rebuilding or reprovisioning. A cold deployment still needs to populate them.

While the app is open, an hourly timer checks whether the default 24-hour refresh interval has elapsed. The importer detects CSV or XLSX content (LAHD's .csv URL now sometimes returns an Excel workbook) and validates schema, dates, row counts, coordinate coverage, and historical-month retention before replacement. The prior validated file is retained as `data/cleaned_evictions_final.previous.csv` for rollback, and source timestamps plus QA results are recorded in `data/refresh_metadata.json`. File modification times invalidate the dashboard's cached data. Closing the app stops the timer.

Run the refresh independently with:

```bash
.venv/bin/python data_refresh.py --dry-run   # validate only
.venv/bin/python data_refresh.py --no-geocode
.venv/bin/python data_refresh.py             # include new-address Census geocoding
```

Set `LAHD_AUTO_REFRESH_HOURS=0` to disable automatic checks, or set another number to change the interval. LAHD's direct CSV does not contain coordinates, so the pipeline reuses verified address/APN coordinates and a persistent incremental geocode cache. Unmatched records remain in the source file but are excluded from map-only analysis.

Never commit `.streamlit/secrets.toml` or `.env`. If a key has previously been committed, revoke and replace it at the provider.
