"""Source-backed acquisition and prospective-validation tools for BETA."""
from pathlib import Path
import pandas as pd
import streamlit as st
from model_data import capture_inventory, reporting_delay_report, point_in_time_attributes

ARCHIVE = Path("data/source_archives")
ATTRIBUTE_COLUMNS = ["building_id", "effective_at", "available_at", "source_url", "rental_units", "year_built", "rso_status", "violations_open", "ownership_change_date"]
CROSSWALK_COLUMNS = ["full_address", "building_id", "apn", "unit_id", "match_status", "available_at", "source_url"]

GUIDE = """# Model data and prospective validation

## What is active
The predictor compares its original 15 notice-history features with five additional activity features. Optional building attributes are context/review-only until verified historical coverage and leakage-safe evaluation justify promotion. No reporting-delay buffer has yet been estimated or applied. No claim of 95% precision follows from this page.

## Reporting delays
Non-dry-run LAHD refreshes now archive exact downloaded bytes in compressed, checksum-addressed files and log each observation in SQLite. Identical downloads share one payload. Even a stale download retained only for inspection is archived; the production source still passes the existing replacement checks. Dry runs do not write archives. The initial local baseline is an observation made today, not proof of historical availability.

Compare successive archived extracts. A record's first-seen date minus Date Received is an observed availability-delay proxy. The initial capture is left-censored and excluded. This is not the time between tenant service and agency receipt. Polling frequency, agency backfills and record corrections affect it; full-field fingerprints are not stable notice IDs, and exact duplicate notices cannot be distinguished. Request a stable notice identifier and publication/revision timestamps from LAHD. Archive missing/deleted records too; never interpret disappearance as withdrawal or a resolved case.

After multiple captures spanning several reporting cycles, inspect delay distributions by received month and revisions. Choose a conservative, documented outcome-maturity buffer using earlier data only. Train/evaluate only outcomes whose 90-day window plus this buffer has elapsed. A measured 95th percentile is not a guarantee of completeness. Do not retroactively invent availability timestamps. Cloud local disk can disappear on reboot/redeploy; back up archives to durable controlled storage. Compression and hash deduplication reduce storage, but unique snapshots accumulate. No scheduled job is created by this feature; archives are added when an existing refresh runs.

## Verified building identification
Use the LA County Assessor portal to investigate APNs, site addresses and property characteristics. Request a dated, machine-readable extract and data dictionary if a suitable public bulk export is unavailable. APNs identify parcels, not necessarily buildings; a parcel can contain multiple buildings, and parcel numbers can change. Keep building_id, parcel/APN, unit_id and source notice ID as separate fields.

Normalize case, whitespace, directionals and suffixes without deleting meaningful unit or street-number information. Use exact authoritative matches first. Review fuzzy/spatial candidates manually; do not merge them automatically. Keep unmatched and conflicting records, match method, source, effective dates and availability dates. A current lookup does not prove the same building characteristics were known historically. Compare raw and matched counts and forbid join multiplication. Repeat filings are not necessarily duplicate records.

## Rental units and time-matched attributes
Request dated building-level total residential units, rental units, year built and parcel/building relationships from the Assessor and LAHD. LAHD's Rent Registry describes unit-level reporting and possible public-record disclosure with exemptions/redactions; it is not an unrestricted public bulk dataset. Request building-level counts and dated RSO status, not tenant names/contact details. Do not equate Assessor residential units with rental units or occupied renter households. Verify split parcels, mixed-use properties, exemptions, demolitions and conversions.

For each forecast origin use both effective_at <= origin and available_at <= origin. Unknown values remain missing, not zero or one. Preserve raw notice counts alongside notices per 100 verified rental units (100 × notices / rental_units); this is notice intensity, not the proportion of tenants evicted. Compute building age as origin year minus verified year built and reject impossible years.

Obtain RSO status through LAHD/ZIMAS and dated records; do not infer it solely from age. Obtain dated inspection/violation events and closure history through LAHD/LADBS records or a public-records request; complaints are not confirmed violations. Ownership-change dates require dated transaction/recording evidence; similar owner names do not establish a common owner. Aggregate only events known by the forecast origin. For ZIP ZORI, retain release/vintage dates and use only releases available at origin; current revised historical values are not a genuine point-in-time vintage. Compute 12-month change as ZORI_t / ZORI_(t-12) - 1, preserving unavailable months. ZORI measures asking rents, not actual rent paid.

## Future-period evaluation
Existing rolling validation uses non-overlapping 90-day windows, not exact calendar quarters. Its latest evaluation was already examined during development. Pre-register a new future 90-day origin, feature definitions, eligibility, data versions, calibration procedure and outreach capacity before outcomes arrive. At that origin save predictions, 95% intervals, building IDs, source checksum and model version. Never overwrite this frozen forecast after outcomes become visible. Evaluate after the 90 days plus a documented reporting-maturity buffer. Keep a later untouched period outside all tuning.

Report Brier error, constant-prevalence and existing-model baselines, AP, precision AND recall at fixed outreach counts (e.g. 100 and 500), calibration, sample sizes and missingness. Compare models on the same eligible addresses and periods. Bootstrap entire addresses for paired metric differences; use time-block sensitivity checks when several periods exist. Repeat by building size and neighborhood where sample sizes support interpretation. Do not choose a threshold on the evaluation outcomes or claim 95% precision from a tiny selected sample. If future-period results influence a change, they become development evidence; reserve another untouched future period.

## Source entry points
- LA County Assessor: https://portal.assessor.lacounty.gov/
- LAHD Rent Registry: https://housing.lacity.gov/rental-property-owners/rent-registry
- LAHD RSO/ZIMAS instructions: https://housing.lacity.gov/rental-property-owners/rso-property-search
- LAHD service directory: https://housing.lacity.gov/online-services
- LADBS service directory: https://dbs.lacity.gov/
- Zillow research downloads: https://www.zillow.com/research/data/
- Zillow ZORI methodology: https://www.zillow.com/research/methodology-zori-repeat-rent-27092/
"""


def render_model_data():
    st.subheader("Model data & validation")
    st.info("Official property context is now downloaded. It is not used in retrospective training: verified historical availability and building/rental-unit links are still required. Notice-history enrichment is tested separately in the predictor.")
    st.caption("Current forecasts still use the latest extract; reporting-delay-adjusted training is not yet active. More capture history is needed to estimate a maturity buffer.")
    st.caption("This view uses the full source/archive inventory; sidebar address, cause and district filters do not alter it.")
    from property_context_ui import render_context_summary
    render_context_summary()
    st.download_button("Download acquisition and validation guide", GUIDE, "model_data_guide.md", "text/markdown")
    inventory = capture_inventory(ARCHIVE)
    st.markdown("### Reporting-delay archives")
    st.write(f"Capture observations: {len(inventory):,} · Distinct payloads: {inventory.sha256.nunique():,}")
    if inventory.empty:
        st.warning("No captures yet. The next non-dry-run LAHD refresh will start an observed baseline. Past availability cannot be reconstructed from one current extract.")
    else:
        st.dataframe(inventory.rename(columns={"captured_at":"Captured (UTC)", "sha256":"SHA-256", "source":"Source", "bytes":"Original bytes"}), hide_index=True, width="stretch")
    st.caption("Exact payloads are compressed and deduplicated. Archives are local and excluded from Git; back them up to durable storage. No new schedule has been created.")
    if inventory.sha256.nunique() >= 2 and st.button("Measure observed reporting delays"):
        try:
            with st.spinner("Comparing archived source records…"):
                report = reporting_delay_report(ARCHIVE)
        except (ValueError, OSError) as error:
            st.error(f"Archive comparison unavailable: {error}. Preserve the archive and check its integrity; do not treat this as zero delay.")
            report = None
        if report is not None and report.attrs.get("excluded_stale_snapshots"):
            st.warning(f"Excluded {report.attrs['excluded_stale_snapshots']} older-ending source snapshot(s) from delay estimates. They remain archived; an older-ending extract cannot establish current reporting completeness.")
        if report is None:
            pass
        elif report.empty:
            st.info("No non-baseline, newly observed record identities with valid nonnegative delays yet.")
        else:
            st.write(f"{len(report):,} newly observed record identities; median delay {report.observed_delay_days.median():.0f} days; 95th percentile {report.observed_delay_days.quantile(.95):.0f} days.")
            st.download_button("Download observed-delay audit", report.to_csv(index=False), "observed_reporting_delays.csv", "text/csv")
        st.warning("First-seen minus Date Received is an observation-delay proxy, not a measured publication timestamp. Initial records are excluded; corrections may look new. Do not automatically use this percentile as a completeness guarantee.")

    st.markdown("### Building identity, units and dated attributes")
    for label, columns, name in [("Verified address crosswalk", CROSSWALK_COLUMNS, "building_crosswalk.csv"), ("Dated building attributes", ATTRIBUTE_COLUMNS, "building_attributes.csv")]:
        st.download_button(f"Download {label.lower()} template", pd.DataFrame(columns=columns).to_csv(index=False), name, "text/csv")
    st.markdown("""Start with [LA County Assessor property lookup](https://portal.assessor.lacounty.gov/), [LAHD Rent Registry](https://housing.lacity.gov/rental-property-owners/rent-registry), [RSO/ZIMAS lookup instructions](https://housing.lacity.gov/rental-property-owners/rso-property-search), [LAHD services](https://housing.lacity.gov/online-services), and [Zillow source data](https://www.zillow.com/research/data/).

Request dated machine-readable records and a data dictionary when bulk history is not public. Parcel ≠ building ≠ unit. Residential units ≠ rental units. Request aggregate building counts, not private tenant information. The downloadable guide explains matching, violations, ownership changes and ZIP rent changes.""")
    with st.expander("Review a dated building-attributes file (does not retrain model)"):
        uploaded = st.file_uploader("Building attributes CSV", type="csv", key="building_attributes_review")
        origin = st.date_input("Forecast origin (00:00 UTC)", key="attributes_origin")
        if uploaded is not None:
            try:
                attributes = pd.read_csv(uploaded, dtype={"building_id":str})
                if attributes.empty:
                    st.info("The template has no records. Add verified source rows before reviewing.")
                else:
                    ids = attributes[["building_id"]].drop_duplicates()
                    reviewed = point_in_time_attributes(ids, attributes, pd.Timestamp(origin, tz="UTC"))
                    st.dataframe(reviewed, hide_index=True, width="stretch")
                    st.caption("Only records effective and available by the cutoff appear. Missing matches remain missing. Validation checks structure, not truth or building-link accuracy; nothing is saved or added to forecasts.")
            except (ValueError, KeyError, pd.errors.ParserError) as error:
                st.error(f"Import rejected: {error}")
    st.markdown("### Hold back a genuinely future evaluation")
    st.write("Existing rolling tests are retrospective. Before a future origin, freeze the protocol; at that origin save forecasts and source/model versions. Evaluate only after 90 days plus a justified reporting-delay buffer. Do not tune on that evaluation. Keep a later period untouched.")
    with st.expander("Full methods and acquisition checklist"):
        st.markdown(GUIDE)
