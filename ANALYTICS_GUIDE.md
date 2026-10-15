# EVICTOR BOOK: interpreting the current metrics

The combined site's **Mathematical methods** navigation page (`/methods`) contains the shared code-grounded reference for ALPHA and BETA, including calibrated XGBoost, ridge logistic regression, monthly OLS and their approximate uncertainty methods. It distinguishes forecasting from descriptive scores and display encodings, and offers a Markdown download of all formulas. ALPHA's Methods & resources and BETA's Mathematical methods view point to the same reference, maintained in `methods_ui.py`.

## Model-data readiness

BETA's **Model data & validation** view contains a downloadable acquisition/validation guide, building crosswalk and dated-attribute templates, and an import reviewer. The reviewer requires known building IDs, source references, effective dates and availability dates. It excludes versions not known at origin, rejects duplicate versions and invalid unit counts, and preserves unmatched IDs. This is structural review, not verification of source truth or a new model feature. APNs, buildings and units must remain separate; total residential units are not automatically rental units.

`data_refresh.py` archives exact source payloads on non-dry-run downloads before the production replacement checks, including a stale source retained only for comparison. `model_data.py` compresses/deduplicates payloads by SHA-256 and logs every capture in SQLite. The October 15, 2026 local baseline is observed availability only, not the original publication date. With successive changed extracts, first-observed minus received-date delays can be measured; first-capture identities are left-censored and excluded. Corrections can appear as new identities without stable source IDs. No maturity buffer is yet estimated/applied, no scheduler is added, and cloud local disks need durable backup. Archives are excluded from Git.

The current predictor still uses the same 15 notice-history features. New building attributes, ZIP-rent vintages, reporting-delay buffers, and truly prospective evaluation remain acquisition/development work—not demonstrated improvements. The guide specifies chronological availability gates, notice counts alongside per-unit intensity, and freezing forecasts before a future 90-day window plus maturity buffer. More retrospective tuning cannot create an untouched future test or establish 95% precision.

## Counts and dates

The source is the Los Angeles Housing Department (LAHD), not the separate housing authority. A source row represents a reported notice record. It does not establish a completed eviction, a unique household, or a number of people displaced. The importer currently removes exact duplicate rows; identical notices cannot always be distinguished without a stable source identifier.

`Date_Filed` is the app's name for LAHD's **Date Received**. `Notice Date` is a separate source field. An address-date node groups records sharing an address and received date. `Eviction_Count` counts records in that node. `Total_Rent_Owed` sums reported dollars in that node; it is not monthly contract rent. `repeat_count` in the aggregated app counts distinct address-date nodes. `Apn` identifies a parcel, not a verified owner.

## Correlation

The app spatially matches filtered geocoded notice records to neighborhoods, totals notices within each neighborhood, and joins the supplied neighborhood Census CSV by name. Each neighborhood is one observation. Pearson's r compares each numeric column with `evictions_total`, using complete pairs only. The displayed sample size is the number of paired neighborhoods. At least three pairs with variation are required as a basic computational guard; that is not an evidentiary standard.

r ranges from −1 to +1: sign indicates direction and magnitude measures linear association. The app does not run a significance test, adjust for multiple comparisons, control for confounders, account for spatial dependence, or establish causation. Neighborhoods with no matched notice activity are absent. Raw counts are affected by neighborhood size and rental stock. Filing-node and address counts overlap mathematically with notice counts. Neighborhood associations do not establish characteristics of individual tenants.

## Demographic columns

`White_share`, `Black_share`, `Hispanic_share`, `Asian_share`, and `Other_share` are percentage-scale fields. A value of 4.7 means 4.7%. The five values sum to approximately 100% in every row of the supplied 114-neighborhood file. `Other_share` is the residual outside the four named categories; the original Census variable mapping is absent, so exact membership and race/ethnicity handling are unverified. It is not an eviction cause or missing-data indicator.

`median_income` and `median_rent` contain supplied dollar estimates. Their vintage, aggregation, and denominator require the original data construction documentation. `rent_burden_pct` is a percentage whose exact threshold/denominator also requires that documentation. Averaging these neighborhood figures without weights does not produce a citywide population estimate or citywide median.

## Trend anomalies

For month t and selected window W, baseline = average of the W preceding monthly notice counts, excluding t. Baseline SD is their sample standard deviation (denominator W−1). z = (current count − baseline) / baseline SD. An alert appears when |z| ≥ 2. Percentage change is 100 × (current / baseline − 1).

A full W-month history is now required. Missing months stay missing, and zero SD or zero baseline prevents the corresponding division. The two-SD rule is a descriptive screening threshold, not a validated significance test or a legal standard. Filters, incomplete months, source revisions, and geocoding coverage can create apparent anomalies.

## ALPHA forecast

Ordinary least squares fits count = intercept + slope × month index. The final 1–3 months (min(3, max(1, floor(n/4)))) are withheld to calculate mean absolute error (MAE) from an earlier-data fit. MAE is the average absolute difference between predictions and observed counts, measured in notices. The final displayed model is refitted to all available months; future negative counts are clipped to zero.

Two nominal 95% bands are displayed and exported. The confidence interval estimates the mean monthly count under the fitted linear trend; the wider prediction interval estimates a future observed monthly count. With n months, residual variance is SSE / (n−2), leverage at future index x is h = 1/n + (x−mean(x))² / Σ(x−mean(x))², and the critical value is Student-t(.975, n−2). Mean bounds are fitted count ± t × sqrt(s²h); prediction bounds are fitted count ± t × sqrt(s²(1+h)). Count bounds are truncated at zero. These are pointwise intervals, not simultaneous coverage of the entire forecast path.

At least four months are required; missing calendar months withhold the forecast. Classical OLS coverage assumes an adequate linear trend and independent, normal, constant-variance errors. Actual 95% coverage has not been validated. This model does not adjust for seasonality, reporting delay, partial months, or policy changes; these can invalidate the nominal coverage.

## Routine filing addresses

The Routine filers tab uses fixed retrospective cohorts within the active filters. By default an address is routine when notices occur in at least six distinct calendar months in the selected period. The threshold is adjustable and is a project definition, not a validated or official label. Multiple records in the same month count as one active month for classification, but all records contribute to notice volume.

For each year or month, the chart sums notices separately for routine and infrequent addresses. The orange lower area is routine volume; the purple area adds infrequent volume, so the upper boundary is the combined total. Percentage mode divides each cohort's period count by all selected notices in the same period. It is share of notices, not notices per rental household. Cohorts recalculate when filters change, and partial years should not be compared as full years. Addresses are not verified landlords, and building size is not controlled. This retrospective grouping must not be treated as a prospective prediction.

## BETA prediction

BETA estimates whether an address with a notice in the preceding 365 days receives another reported notice in the next 90 days. Version temporal-xgboost-v5-rolling-brier-ci95 compares calibrated XGBoost, full-feature ridge and a two-feature baseline. Fifteen historical features include counts, filing dates, recency, momentum, nonpayment/rent-reporting shares and reported dollars—not actual rent paid, completed evictions or verified ownership.

Snapshots are 90 days apart with non-overlapping outcomes. At least five mature snapshots are required. Up to three earlier validation origins reserve their immediately preceding mature snapshot for calibration and train on older snapshots. Equal-weight mean validation scores select candidates. Ridge penalties are .5/10/100, baseline 2. Six XGBoost configurations test depth 2–4, L2 10/30, 160 rounds at .05 or 320 at .03, child weight 10/30, and optional 365-day half-life training weights. Each tests log-loss or Brier sigmoid calibration with nonnegative slope and .001 slope-squared penalty. XGBoost must lower mean Brier without reducing mean AP or top-10% precision versus eligible ridge. Top-decile ties receive fractional inclusion. Short histories use the legacy ridge-only fallback.

Evaluation models refit on older snapshots; the preceding period calibrates XGBoost, with outcomes known by the latest origin. Latest-period outcomes do not choose parameters or fit calibration in this procedure. However this period was inspected during previous development, so these are retrospective development results, not a fresh prospective test. Live XGBoost reserves the latest mature snapshot for calibration; ridge uses all mature snapshots. Calibration plots are diagnostics only. Wilson 95% intervals describe observed bin rates. Source modification time, version and origin invalidate cached output.

Ridge probability intervals use address-clustered sandwich covariance and a Student-t critical value. XGBoost intervals use 30 bootstrap refits, resampling entire address clusters jointly across training and calibration. Both the booster and sigmoid calibration are refitted in each sample. The bootstrap calibrated-logit standard deviation supplies a normal 95% margin around the fitted logit; at least 20 successful refits are required. Bounds appear in charts, tables, inspectors and downloads. This is a computationally bounded approximation, not proof of established 95% coverage.

These are pointwise, approximate model-conditional intervals for fitted probabilities, not prediction intervals for a future binary event. Feature transformations, model selection, and ridge penalty are treated as fixed. Shrinkage bias, cross-address dependence, reporting errors, changing policies, and selection uncertainty are not fully captured. Independent 95% coverage has not been established. A narrow interval does not demonstrate that an address will receive a notice or that a tenant will be displaced.

Historical ridge-only scores from earlier versions must not be presented as results for this XGBoost pipeline. Current model-specific independent test scores and selection parameters appear in BETA's Model performance tab and its downloadable comparison.

This is an experimental recurrence estimate, not a probability that a tenant will be physically evicted. Historical data is reconstructed from today's extract: original publication timestamps and later backfills are unavailable. Source gaps or reporting delays can therefore bias even correctly separated backtests. No rental-unit denominator, verified parcel/ownership identity, or independent fairness validation is available. Do not use it for tenant screening or punitive decisions.

## Data priorities for improving forecasts

These are recommended additions, not currently integrated model features. Additional data must demonstrate improvement across rolling chronological tests; more columns do not guarantee better predictions.

1. **Building size and stable identity:** [LA County Assessor parcel rolls](https://www.arcgis.com/home/item.html?id=70d93266f45a4080a97b285a471493cd&sublayer=0) include AIN, property use, units, year built, and roll year. Match notices to verified parcels and use the roll known at the prediction origin. Do not double-count units across roll years or confuse parcel identity with verified ownership. Unit counts can help separate property size from unusually frequent notices.
2. **Actual rents and dated changes:** [LAHD Rent Registry](https://housing.lacity.gov/rental-property-owners/rent-registry) collects annual unit rents and tenancy information. Seek privacy-redacted records with APN, reporting year, rent amounts, and dated rent increases. Availability is subject to public-records exemptions. The portal's CSV is an owner submission template, not a public citywide rent dataset. Contract rent would permit a more meaningful arrears-to-monthly-rent measure than ZIP asking rent.
3. **Neighborhood denominators and context:** [Census ACS 5-year API](https://www.census.gov/data/developers/data-sets/acs-5year.html) provides tract/ZCTA housing estimates. Candidate tables include B25003 (tenure), B25070 (gross rent as a percentage of household income), and B25064 (median gross rent). Use released vintages and retain margins of error. Area estimates are not individual tenant characteristics.
4. **Different, verified outcomes:** [LA Superior Court case access information](https://www.lacourt.org/paos/v2public/FAQ) is a starting point for researching unlawful-detainer filings and case documents. Access limits and fees apply; an unrestricted bulk dataset is not assumed. Court filings, judgments, and actual removals are different targets. Verified outcomes could support a separate model; they must not be substituted silently for the current any-LAHD-notice target.
5. **Reporting history:** Archive successive official LAHD extracts with collection timestamps and stable notice IDs if available. This makes it possible to quantify backfills and reporting delay, and to reconstruct what was actually knowable at each historical forecast origin. This is important for honest validation even before adding new predictors.

Validate candidates on multiple mature 90-day periods using Brier error, average precision, calibration, and geographic subgroup checks. Monthly prediction bands need out-of-time empirical coverage checks. Exclude information first learned after the forecast origin, even when a record's event date is earlier.

## Repair and coverage verification

The September 30, 2026 import accepted LAHD's Excel payload served from its .csv URL: 309,010 normalized records, latest received date September 28, 2026, 258,130 mapped (83.53%), and 50,880 unmapped. The net change from the prior file was +8,842 records; source revisions mean that is not necessarily the number of newly filed notices. All 12 months of 2024 are present, totaling 86,512 records. A monthly all-source coverage table in Methods distinguishes source coverage from mapped coverage.

FRED rent CPI now uses CUSR0000SEHA (national US city average, seasonally adjusted, 1982–1984=100), with a public CSV fallback. The key supplied for this repair had 31 characters; FRED requires 32. No missing character was guessed, and that incomplete key was not stored.

On October 1 the same official download URL returned a CSV with a latest received date of July 19, despite a newer file-modification timestamp. The refresh guard retains the September 28 local dataset, records the check separately from the successful import, and retries on the normal interval. A file's publication timestamp is not proof of newer event coverage.

Sources: [LAHD notices](https://housing.lacity.gov/residents/renters/eviction-notices-filed), [FRED rent series](https://fred.stlouisfed.org/series/CUSR0000SEHA), [FRED key format](https://fred.stlouisfed.org/docs/api/api_key.html). Local evidence: `data/refresh_metadata.json`, `data/cleaned_evictions_final.csv`, `data/neighborhood_census.csv`, `feature_engineering.py`, and `policy_metrics.py`.
