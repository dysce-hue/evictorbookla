# From recurrence prediction to a defensible property-month study

## Implemented October 16, 2026

The separate **Property-month pilot** uses 21,211 parcels from an official selected-area RSO layer, not all LA rentals or all RSO properties. Its 954,495 APN-month rows span 45 origins. Of these parcels, 14,734 have no observed notice in the saved extract. This adds a real zero-observed comparison group within the selected cohort, but not verified true negatives or citywide representativeness.

The panel preserves exact normalized APN matches, past-only 30/90/180/365-day notice counts and recency, extract-boundary flags, mature observed-notice labels, and separate unknown court stages. Current RSO units are explicitly current context. Historical membership is unverified. Every training label is null and every training-eligibility flag is false. No new model is fitted; the existing address-recurrence predictor is unchanged.

Reproduce with `fetch_rso_panel.py`; use `--build-only` to rebuild from the saved inventory without another download. Source provenance/checksums, the compressed inventory, Parquet panel, monthly summary and audit are under `data/property_panel/`.

## Required before a citywide model

1. **Complete dated universe:** obtain a documented citywide inventory covering RSO and non-RSO rentals, including zero-notice properties, entry/exit dates and known coverage exclusions. Do not substitute the existing notice-linked parcel download, selected-area pilot, or age-based guesses for this universe.
2. **Historical denominators:** verify parcel-to-building-to-unit relationships and dated rental-unit counts, conversions, demolitions and exemptions. Keep occupied renter households distinct from residential units and registered RSO units. Do not divide old notices by today's unit count and call it a historic eviction rate.
3. **Publication-time history:** continue durable notice-source capture and identify stable notice IDs, corrections and reporting maturity. An older-ending source was retained only for audit, not used as evidence of current completeness. Original availability cannot be reconstructed simply from event dates.
4. **Separate stage evidence:** acquire legally accessible, documented court/outcome records with case/event IDs, event dates and availability dates. Link notice, UD filing, judgment, writ and execution only with verified evidence. Allow unmatched, unresolved, reversed and non-progressing cases. Missing outcomes remain unknown. LAHD administrative enforcement cases are not UD filings. Do not send or publish tenant information without appropriate authorization and safeguards.
5. **Historical owners and networks:** obtain dated legal ownership and transaction records. Store entity IDs, effective/availability intervals, legal-owner versus parent-network relationships and match method. Similar names or shared addresses are candidate links—not established common ownership. Ultimate networks require verification. Network features must exclude future acquisitions and future outcomes.

## Matching acceptance rules

Keep `exact_apn`, verified address/unit, parcel-contained coordinate, reviewed fuzzy candidate and unmatched/ambiguous categories separate. Exact APN resolves a parcel only. Never manufacture scores such as .95 to imply a measured matching probability. Estimate error rates from labeled manual review before using probabilistic match confidence. Reject join multiplication and conflicting denominators; report unmatched coverage and sensitivity to excluding uncertain links.

## Modeling and validation after those requirements are met

The new target must be explicit: citywide notice incidence is not recurrence among previously noticed addresses, and neither is completed eviction. Define stage-specific at-risk populations and follow-up completeness before fitting conditional stage models. Include counts alongside legitimate per-unit measures; never label notice intensity as a tenant eviction probability.

Monthly origins with 90-day outcomes overlap. Purge training rows whose outcome windows extend beyond validation/calibration origins, and add a justified reporting buffer. Preserve a genuinely later untouched period; consider owner/spatial holdouts to test generalization beyond known networks. Freeze predictions, versions, source checksums and the capacity rule before outcomes arrive. Evaluate calibration, Brier error, precision AND recall at fixed capacities, prevalence baselines, sample sizes, coverage, and dependence-aware uncertainty. Compare the same population and periods. Do not tune on the final outcomes and continue calling them untouched.

An abstention/review band can be evaluated on earlier data, but arbitrary .20/.70 thresholds are not established policy standards. High estimated probability is not the same as high statistical confidence. The current experiments do not establish 95% prospective precision. The narrower top-100 historical lists have high observed precision but low recall, and simpler models performed similarly.

Prediction and descriptive panels do not establish causal policy effects. A separate identification strategy, intervention timing, comparison group, assumptions and diagnostics are required for causal claims.
