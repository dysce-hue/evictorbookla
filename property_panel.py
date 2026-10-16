"""Stage-aware, APN-month pilot. Not a citywide rental-universe model."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def normalize_apn(values):
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    text = text.str.replace(r"[-\s]", "", regex=True)
    return text.where(text.str.fullmatch(r"\d{10}",na=False) & text.ne("0000000000").fillna(False))


def build_panel(notices, inventory, observed_at, start=None, end=None):
    """Current inventory defines a descriptive cohort, not past exposure.

    All prior features use event dates <= origin. Availability before historical
    origins and rental-universe membership are not established, so training
    labels remain null. Zero observed notices is not proof of no eviction.
    """
    raw = inventory.copy()
    raw["apn"] = normalize_apn(raw.APN)
    invalid_inventory = int(raw.apn.isna().sum())
    raw = raw.dropna(subset=["apn"])
    if raw.empty:
        raise ValueError("No valid inventory APNs")
    raw["units"] = pd.to_numeric(raw.RSO_Units,errors="coerce")
    raw.loc[(raw.units <= 0) | (raw.units % 1 != 0),"units"] = float("nan")
    grouped = raw.groupby("apn")
    properties = grouped.agg(source_rows=("APN","size"), unit_versions=("units","nunique"), rso_units_current=("units","first")).reset_index()
    properties.loc[properties.unit_versions.ne(1),"rso_units_current"] = float("nan")
    properties["unit_count_ambiguous"] = properties.unit_versions.gt(1)
    work = notices.copy()
    work["apn"] = normalize_apn(work.Apn)
    work["received_date"] = pd.to_datetime(work["Date Received"],format="mixed",errors="coerce").dt.normalize()
    dated = work.received_date.dropna()
    if dated.empty:
        raise ValueError("No valid notice dates")
    first,last = dated.min(),dated.max()
    first_origin = pd.Timestamp(start or first).to_period("M").end_time.normalize()
    last_origin = pd.Timestamp(end or last).to_period("M").end_time.normalize()
    origins = pd.date_range(first_origin,last_origin,freq="ME")
    captured = pd.Timestamp(observed_at)
    if captured.tzinfo is None:
        raise ValueError("Inventory observation timestamp requires timezone")
    inventory_day = captured.tz_convert("America/Los_Angeles").tz_localize(None).normalize()
    matched = work[work.apn.isin(properties.apn) & work.received_date.notna()]
    frames = []
    for origin in origins:
        part = properties.copy()
        part["origin"] = origin
        month_start = origin.to_period("M").start_time
        counts = matched[matched.received_date.between(month_start,origin)].groupby("apn").size()
        part["observed_notices_month"] = part.apn.map(counts).fillna(0).astype(int)
        for window in [30,90,180,365]:
            prior = matched[matched.received_date.between(origin-pd.Timedelta(days=window-1),origin)]
            part[f"prior_notices_{window}"] = part.apn.map(prior.groupby("apn").size()).fillna(0).astype(int)
        past = matched[matched.received_date.le(origin)].groupby("apn").received_date.max()
        part["days_since_last_notice"] = (origin-part.apn.map(past)).dt.days.astype("Int64")
        part["history_365_inside_extract"] = origin-pd.Timedelta(days=364) >= first
        part["month_inside_extract"] = month_start >= first and origin <= last
        part["outcome_window_inside_extract"] = origin+pd.Timedelta(days=90) <= last
        future = matched[matched.received_date.gt(origin) & matched.received_date.le(origin+pd.Timedelta(days=90))].groupby("apn").size()
        labels = part.apn.map(future).fillna(0).gt(0).astype("Int64")
        part["observed_notice_next_90d"] = labels.where(part.outcome_window_inside_extract)
        # Current source vintage cannot establish historical membership or availability.
        part["inventory_observed_by_origin"] = origin >= inventory_day
        part["historical_membership_verified"] = False
        part["training_eligible"] = False
        part["training_target_notice_next_90d"] = pd.Series(pd.NA,index=part.index,dtype="Int64")
        for stage in ["ud_filing", "judgment", "writ", "lockout"]:
            part[f"{stage}_next_90d"] = pd.Series(pd.NA,index=part.index,dtype="Int64")
        part["owner_network_id"] = pd.NA
        part["match_method"] = "exact_apn_not_building_verification"
        frames.append(part)
    if not frames:
        raise ValueError("No monthly origins in requested range")
    panel = pd.concat(frames,ignore_index=True)
    if panel.duplicated(["apn","origin"]).any():
        raise ValueError("Panel key duplicated")
    ever = set(matched.apn)
    audit = dict(grain="APN × month-end; parcel, not verified building or unit",inventory_source_rows=len(inventory),
        inventory_invalid_apn_rows=invalid_inventory,universe_apns=len(properties),
        universe_apns_with_observed_notice=len(set(properties.apn)&ever),
        universe_apns_without_observed_notice=len(set(properties.apn)-ever),
        notice_rows=len(work),notice_invalid_apn_rows=int(work.apn.isna().sum()),
        notice_rows_matched_to_subset=len(matched),notice_rows_outside_subset=int((work.apn.notna() & ~work.apn.isin(properties.apn)).sum()),
        notice_invalid_date_rows=int(work.received_date.isna().sum()),
        inventory_missing_unit_apns=int(properties.rso_units_current.isna().sum()),
        ambiguous_unit_apns=int(properties.unit_count_ambiguous.sum()), panel_rows=len(panel),monthly_origins=len(origins),
        source_first_received=first.date().isoformat(),source_last_received=last.date().isoformat(),
        inventory_observed_at=captured.isoformat(),training_eligible_rows=0,
        missing_stages=["UD filing","judgment","writ","lockout","historical owner/network"],
        limitations=["Selected-area RSO subset, not all LA rentals or all RSO properties.",
            "Current inventory cohort creates survivorship/selection risks when projected backward.",
            "Observed zeros are not verified true negatives; extract completeness and historic membership are unverified.",
            "Current RSO unit counts are not historic exposure denominators.",
            "Notice event dates reconstruct history; original publication-time availability remains unknown.",
            "No new model is trained on this pilot; existing recurrence predictor is unchanged."])
    return panel,audit


def save_panel(notice_path, inventory_path, observed_at, output):
    notices = pd.read_csv(notice_path,dtype={"Apn":str},low_memory=False)
    inventory = pd.read_csv(inventory_path,dtype={"APN":str})
    panel,audit = build_panel(notices,inventory,observed_at)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    panel.to_parquet(output/"parcel_month_pilot.parquet",index=False)
    summary = panel.groupby("origin").agg(parcels=("apn","nunique"),observed_notices=("observed_notices_month","sum"),zero_observed_notice_parcels=("observed_notices_month",lambda s:int(s.eq(0).sum())),training_eligible=("training_eligible","sum"),month_inside_extract=("month_inside_extract","first")).reset_index()
    summary.to_csv(output/"monthly_summary.csv",index=False)
    audit["notice_source_sha256"] = hashlib.sha256(Path(notice_path).read_bytes()).hexdigest()
    audit["inventory_sha256"] = hashlib.sha256(Path(inventory_path).read_bytes()).hexdigest()
    (output/"audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
