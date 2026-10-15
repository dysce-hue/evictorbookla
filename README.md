# EVICTOR BOOK ALPHA and EVICTOR BOOK BETA

A pair of local, interactive Los Angeles eviction-notice applications that share one validated data file.

- **EVICTOR BOOK ALPHA** (`app.py`) preserves the primary 3D mapping and research experience.
- **EVICTOR BOOK BETA** (`app_beta.py`) is a separate policy-visualization workspace for time patterns, concentration, cause composition, recurrence, persistent hotspots, hotspot address inspection, a transparent building-risk portfolio, an experimental self-updating 90-day notice-recurrence forecast, complete mathematical methods, and data quality.

## Included features

BETA's **Model data & validation** view provides source links, building crosswalk/dated-attribute CSV templates, point-in-time import review and a prospective evaluation guide. Non-dry-run LAHD refreshes archive exact downloaded payloads under `data/source_archives/`, gzip-compressed and deduplicated by SHA-256, with capture observations in SQLite. The local archive is ignored by Git and must be backed up to durable storage. Dry runs do not archive. A local baseline was observed on October 15, 2026; it does not establish historical publication dates. With multiple changed captures, the UI can measure first-observed minus received-date delays, excluding left-censored baseline identities. Record corrections and missing stable notice IDs limit this proxy. These tools do **not** yet add building attributes or a delay buffer to the production predictor.

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

The Beta forecast is limited to addresses observed during the preceding 365 days. It compares calibrated XGBoost, enhanced ridge and a recent-history baseline using up to three rolling earlier validation periods and separate calibration snapshots. XGBoost requires lower mean Brier and no reduction in mean average precision or top-decile precision versus eligible ridge. The latest period is evaluation-only in this procedure, but was examined during prior development; prospective validation remains necessary. Source, origin or model-version changes rebuild the forecast. It is a research/supportive-outreach aid predicting reported notices, not completed evictions or wrongdoing. Approximate 95% intervals exclude model-selection and reporting/backfill errors.

XGBoost tests seven bounded configurations, including a 20-feature activity-enriched challenger, and two monotone calibration objectives. Five additional event-history features cover prior-month counts, 30-day acceleration, recent-week share, active weeks and latest-date gaps. The enrichment is promoted only on earlier validation with lower Brier and no decrease in AP/top-decile precision versus eligible 15-feature XGBoost. Two CPU threads limit resource use; live intervals retain 30 whole-address bootstrap refits. On macOS XGBoost requires OpenMP (`brew install libomp`); `.venv/lib/libomp.dylib` is supported. Do not commit environments/native libraries.

### October 15, 2026 ranking/data experiment

`fetch_property_context.py` retrieved 30,536 official county parcels and 60,599 matching LAHD enforcement-case rows after approval to query APNs. Current parcel units/year-built fields and source cases are visible in the address inspector; they are not verified building/rental-unit or historical publication-time data. `--rent-only` captured an actual 256-ZIP LA County ZORI vintage through August 2026. These current extracts are context/prospective evidence, excluded from retrospective fitting; no ownership-change data was obtained. Only 440 source notice rows contain explicit unit markers, without verified building/unit linkage. Do not infer unit-level repeat eviction rates.

The activity-enriched model passed earlier gates: latest retrospective Brier ≈.122624 vs .122736 for previous XGBoost, AP ≈.646327 vs .646015. Top-decile precision was unchanged. In the Outreach capacity tab, capacities are selected from earlier periods only; the top-100 choice averaged 99% earlier precision and observed 100/100 later, but reached only 2.8% of positive addresses. Previous XGBoost/ridge also achieved 100/100 in that later group; this is not evidence of a new 100%-accurate model. Top-500 precision fell from 96.6% to 96.2%. Latest outcomes were already inspected during development; prospective validation is pending. Approximate Wilson intervals ignore dependence and selection. Export a future protocol before outcomes; no scheduler is created.

`evaluate_ranking.py` reproduces experiment tables in `data/ranking_evaluation/`. Source provenance/checksums live in `data/property_context/`. A fresh LAHD download ended July 19, 2026—older than saved September 28 data—so it was archived but rejected for replacement. Older-ending snapshots are excluded from observed-delay estimates. There is still no estimated reporting-maturity buffer.

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

The original entry point remains supported: `streamlit run app.py` starts ALPHA with the complete mathematical reference inside its Methods tab. Use `site.py` for the combined top navigation and the dedicated `/methods` page.

The original `app.py` deployment also discovers `pages/1_policy.py`, exposing BETA and its 90-day predictor as the sidebar's **policy** page (`/policy`). `site.py` retains the combined top navigation. A reproducible one-period XGBoost evaluation is recorded in `data/model_evaluation_xgboost_2026-10-14.json`; it is a dated audit snapshot, not a live result.

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
