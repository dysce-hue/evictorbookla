# EVICTOR BOOK: interpreting the current metrics

The combined site's **Mathematical methods** navigation page (`/methods`) now contains the shared code-grounded reference for both ALPHA and BETA. It covers 16 method groups, identifies forecasts separately from descriptive scores and display encodings, and offers a Markdown download of all formulas. ALPHA's Methods & resources and BETA's Mathematical methods view point to the same reference, maintained in `methods_ui.py`.

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

BETA separately estimates whether an address with a notice in the preceding 365 days receives another notice in the next 90 days. It uses ridge logistic regression on recent counts, distinct filing dates, recency, momentum, nonpayment share, rent-reporting share, and reported dollars. It compares Brier score (mean squared probability error, lower better) and average precision (ranking performance, higher better) against a smaller recent-history model.

Version temporal-ridge-v2 adds active months / 13 (a 365-day window can span 13 calendar months), distinct filing days / notices, 90-day / annual notice share, squared recency, and recency × log(1 + 90-day notices). Snapshots are 90 days apart with non-overlapping outcome windows. The penultimate snapshot selects ridge penalty from 0.5, 10, 100 and compares against the two-feature baseline (penalty 2). The enhanced model is selected only if validation Brier improves and average precision does not worsen. Otherwise the baseline is actually served. Average precision treats tied scores as one threshold.

The last mature snapshot is an independent chronological test of that frozen selection, with training outcomes ending no later than the test origin. Test calibration is diagnostic only; live probabilities are no longer transformed using test labels. Wilson 95% intervals describe observed test-group rates. The selected model is then refitted on all mature snapshots for live scoring. File modification time, model version and selected end date invalidate the forecast cache.

Version temporal-ridge-v3-ci95 adds an approximate 95% confidence interval for every live address probability, shown as chart whiskers, table columns, inspector values, and CSV fields. Repeated training snapshots are clustered by address. Let H = XᵀWX + λD (intercept unpenalized), U_g = Σ within address g of x_i(y_i−p_i), and V = cH⁻¹(Σ_g U_gU_gᵀ)H⁻¹, where c = G/(G−1) × (n−1)/(n−k). An address's bounds are sigmoid(η ± t(.975,G−1) × sqrt(xᵀVx)). If there are fewer than two address clusters or insufficient rows, intervals are unavailable rather than fabricated.

These are pointwise, approximate model-conditional intervals for fitted probabilities, not prediction intervals for a future binary event. Feature transformations, model selection, and ridge penalty are treated as fixed. Shrinkage bias, cross-address dependence, reporting errors, changing policies, and selection uncertainty are not fully captured. Independent 95% coverage has not been established. A narrow interval does not demonstrate that an address will receive a notice or that a tenant will be displaced.

Verified October 2, 2026 on the local source through September 28: 10 snapshots, 174,484 address-snapshots, 16,069 final-test addresses (July 1–September 28), 16,035 live eligible addresses. Selected enhanced model: Brier 0.123918 vs baseline 0.137830; average precision 0.641157 vs 0.567905. These are one-period retrospective results, not prospective accuracy or a significance test. Current live window: September 29–December 27, 2026.

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
