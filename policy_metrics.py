"""Policy-oriented measures for the separate EVICTOR BOOK BETA app."""

from __future__ import annotations

import numpy as np
import pandas as pd


FORECAST_HORIZON_DAYS = 90
FORECAST_LOOKBACK_DAYS = 365
FORECAST_MODEL_VERSION = "temporal-xgboost-v4-calibrated-ci95"

FORECAST_FEATURES = [
    "log_notices_30",
    "log_notices_90",
    "log_notices_365",
    "log_filing_dates_90",
    "log_filing_dates_365",
    "recency",
    "momentum",
    "nonpayment_share_365",
    "rent_reporting_share_365",
    "log_rent_owed_365",
    "active_month_share",
    "filing_days_per_notice",
    "recent_activity_share",
    "recency_squared",
    "recency_frequency",
]

FORECAST_FEATURE_LABELS = {
    "log_notices_30": "Notices in the last 30 days",
    "log_notices_90": "Notices in the last 90 days",
    "log_notices_365": "Notices in the last 365 days",
    "log_filing_dates_90": "Distinct filing dates in 90 days",
    "log_filing_dates_365": "Distinct filing dates in 365 days",
    "recency": "More recent filing activity",
    "momentum": "90-day activity acceleration",
    "nonpayment_share_365": "Non-payment share",
    "rent_reporting_share_365": "Rent-owed reporting coverage",
    "log_rent_owed_365": "Reported rent owed in 365 days",
    "active_month_share": "Consistency across active months",
    "filing_days_per_notice": "Spread across distinct filing days",
    "recent_activity_share": "Share of annual notices in 90 days",
    "recency_squared": "Nonlinear recency",
    "recency_frequency": "Recent and frequent activity together",
}


CAUSE_ORDER = [
    "Non-payment of rent",
    "Lease violation",
    "Nuisance / disturbance",
    "Owner/family occupancy",
    "Withdrawal / demolition",
    "Multiple stated causes",
    "Other single cause",
    "Unknown / not specified",
]


def cause_category(value: object) -> str:
    """Collapse source cause strings into mutually exclusive policy categories."""
    if not isinstance(value, str) or not value.strip():
        return "Unknown / not specified"
    terms = []
    for part in value.split(","):
        term = " ".join(part.split()).strip()
        if term and term.lower() not in {"unknown", "nan", "none"} and term not in terms:
            terms.append(term)
    if not terms:
        return "Unknown / not specified"
    if len(terms) > 1:
        return "Multiple stated causes"
    term = terms[0].lower()
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


def prepare_notices(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the LAHD notice-level CSV without discarding unmapped records."""
    df = raw.copy()
    df["Date_Filed"] = pd.to_datetime(df.get("Date Received"), errors="coerce", format="%m-%d-%Y")
    df["full_address"] = df.get("full_address", "").fillna("").astype(str).str.strip()
    df.loc[df["full_address"].eq(""), "full_address"] = "Unknown address"

    district = pd.to_numeric(df.get("Council District"), errors="coerce")
    df["Council_District"] = district.astype("Int64")
    df.loc[df["Council_District"].eq(0), "Council_District"] = pd.NA

    df["Zip"] = df.get("Zip", "").fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    df["Cause"] = df.get("Cause", "").fillna("").astype(str)
    # Vectorized categorization keeps the 300k-row source fast enough for an
    # interactive first load. Commas indicate multiple distinct source causes.
    normalized_cause = df["Cause"].str.strip().str.lower()
    unknown = normalized_cause.isin(["", "unknown", "nan", "none"])
    multiple = normalized_cause.str.contains(",", regex=False) & ~unknown
    category = pd.Series("Other single cause", index=df.index, dtype="object")
    category.loc[unknown] = "Unknown / not specified"
    category.loc[normalized_cause.str.contains("non-payment of rent", na=False)] = "Non-payment of rent"
    category.loc[normalized_cause.str.contains("violation of the rental agreement", na=False)] = "Lease violation"
    category.loc[normalized_cause.str.contains("disorderly behavior|nuisance", regex=True, na=False)] = "Nuisance / disturbance"
    category.loc[normalized_cause.str.contains("owner/family occupancy", regex=False, na=False)] = "Owner/family occupancy"
    category.loc[normalized_cause.str.contains("withdrawing units|demolition", regex=True, na=False)] = "Withdrawal / demolition"
    category.loc[multiple] = "Multiple stated causes"
    df["Cause_Category"] = category
    df["Rent_Owed_num"] = pd.to_numeric(
        df.get("Rent Owed", "").astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["long"] = pd.to_numeric(df.get("long"), errors="coerce")
    return df.dropna(subset=["Date_Filed"]).reset_index(drop=True)


def annual_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Daily cumulative notices by calendar year."""
    if df.empty:
        return pd.DataFrame(columns=["year", "day_of_year", "calendar_day", "notices", "cumulative_notices"])
    daily = (
        df.assign(
            year=df["Date_Filed"].dt.year,
            day_of_year=df["Date_Filed"].dt.dayofyear,
            calendar_day=df["Date_Filed"].dt.strftime("%b %d"),
        )
        .groupby(["year", "day_of_year", "calendar_day"], as_index=False)
        .size()
        .rename(columns={"size": "notices"})
        .sort_values(["year", "day_of_year"])
    )
    daily["cumulative_notices"] = daily.groupby("year")["notices"].cumsum()
    daily["year"] = daily["year"].astype(str)
    return daily


def calendar_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Notice totals for a month-of-year by weekday heatmap."""
    if df.empty:
        return pd.DataFrame(columns=["month", "weekday", "notices"])
    month_order = list(range(1, 13))
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = (
        df.assign(
            month_num=df["Date_Filed"].dt.month,
            month=df["Date_Filed"].dt.strftime("%b"),
            weekday=df["Date_Filed"].dt.day_name(),
        )
        .groupby(["month_num", "month", "weekday"], as_index=False)
        .size()
        .rename(columns={"size": "notices"})
    )
    full = pd.MultiIndex.from_product([month_order, weekday_order], names=["month_num", "weekday"]).to_frame(index=False)
    month_labels = pd.DataFrame({"month_num": month_order, "month": pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%b")})
    return full.merge(month_labels, on="month_num").merge(grouped, on=["month_num", "month", "weekday"], how="left").fillna({"notices": 0})


def property_concentration(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rank addresses and calculate cumulative share of all notices."""
    if df.empty:
        return pd.DataFrame(), {"top_1_pct": 0.0, "top_5_pct": 0.0, "top_10_pct": 0.0}
    ranked = df.groupby("full_address", as_index=False).size().rename(columns={"size": "notices"})
    ranked = ranked.sort_values(["notices", "full_address"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["address_percentile"] = ranked["rank"] / len(ranked) * 100
    ranked["cumulative_share"] = ranked["notices"].cumsum() / ranked["notices"].sum() * 100

    def share_at(percent: float) -> float:
        count = max(1, int(np.ceil(len(ranked) * percent / 100)))
        return float(ranked.iloc[count - 1]["cumulative_share"])

    summary = {"top_1_pct": share_at(1), "top_5_pct": share_at(5), "top_10_pct": share_at(10)}
    return ranked, summary


def compressed_pareto(ranked: pd.DataFrame, max_points: int = 1800) -> pd.DataFrame:
    """Retain the steep head and evenly sample the long tail for fast rendering."""
    if len(ranked) <= max_points:
        return ranked
    head = np.arange(min(500, len(ranked)))
    tail = np.linspace(len(head), len(ranked) - 1, max_points - len(head), dtype=int)
    return ranked.iloc[np.unique(np.concatenate([head, tail]))].copy()


def cause_district(df: pd.DataFrame) -> pd.DataFrame:
    """Mutually exclusive cause composition within each council district."""
    work = df.dropna(subset=["Council_District"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["district", "cause", "notices", "district_share"])
    out = (
        work.groupby(["Council_District", "Cause_Category"], observed=True)
        .size()
        .rename("notices")
        .reset_index()
    )
    totals = out.groupby("Council_District")["notices"].transform("sum")
    out["district_share"] = out["notices"] / totals * 100
    out["district"] = out["Council_District"].astype(int).astype(str)
    out["cause"] = pd.Categorical(out["Cause_Category"], categories=CAUSE_ORDER, ordered=True)
    return out.sort_values(["Council_District", "cause"])


def repeat_intervals(df: pd.DataFrame, cap_days: int = 730) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Days between distinct notice dates at the same address and a compact histogram."""
    dates = df[["full_address", "Date_Filed"]].drop_duplicates().sort_values(["full_address", "Date_Filed"])
    dates["days_to_next"] = dates.groupby("full_address")["Date_Filed"].diff().dt.days
    intervals = dates.dropna(subset=["days_to_next"]).copy()
    intervals["days_to_next"] = intervals["days_to_next"].astype(int)
    if intervals.empty:
        return intervals, pd.DataFrame(columns=["interval", "addresses"])
    bins = list(range(0, cap_days + 31, 30)) + [np.inf]
    labels = [f"{start}–{min(start + 29, cap_days)}" for start in range(0, cap_days, 30)] + [f">{cap_days}"]
    intervals["interval"] = pd.cut(intervals["days_to_next"], bins=bins, labels=labels, right=False, include_lowest=True)
    histogram = intervals.groupby("interval", observed=False).size().rename("address_intervals").reset_index()
    histogram["interval"] = histogram["interval"].astype(str)
    return intervals, histogram


def rent_box_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute box-plot statistics by cause without sending raw records to the browser."""
    work = df[df["Rent_Owed_num"].gt(0)].copy()
    if work.empty:
        return pd.DataFrame()
    cutoff = work["Rent_Owed_num"].quantile(0.95)
    work = work[work["Rent_Owed_num"].le(cutoff)]
    return (
        work.groupby("Cause_Category", observed=True)["Rent_Owed_num"]
        .agg(
            count="size",
            lower=lambda s: s.quantile(0.05),
            q1=lambda s: s.quantile(0.25),
            median="median",
            q3=lambda s: s.quantile(0.75),
            upper=lambda s: s.quantile(0.95),
        )
        .reset_index()
        .sort_values("median", ascending=False)
    )


def hotspot_persistence(df: pd.DataFrame, precision: int = 2) -> pd.DataFrame:
    """Grid cells ranked by how many distinct months contain at least one notice."""
    work = df.dropna(subset=["lat", "long"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["grid_lat"] = work["lat"].round(precision)
    work["grid_long"] = work["long"].round(precision)
    work["month"] = work["Date_Filed"].dt.to_period("M").astype(str)
    months_in_view = max(1, work["month"].nunique())
    out = (
        work.groupby(["grid_lat", "grid_long"], as_index=False)
        .agg(
            notices=("Date_Filed", "size"),
            active_months=("month", "nunique"),
            addresses=("full_address", "nunique"),
            latest_notice=("Date_Filed", "max"),
        )
    )
    out["persistence_pct"] = out["active_months"] / months_in_view * 100
    return out.sort_values(["active_months", "notices"], ascending=False)


def building_risk_portfolio(df: pd.DataFrame, recent_days: int = 180) -> pd.DataFrame:
    """Create a transparent, relative building-attention score.

    The score is descriptive, not predictive. Every component is ranked
    against the other addresses in the currently selected population.
    """
    work = df[df["full_address"].ne("Unknown address") & df["Date_Filed"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    reference_end = work["Date_Filed"].max().normalize()
    recent_start = reference_end - pd.Timedelta(days=recent_days - 1)
    work["recent_notice"] = work["Date_Filed"].ge(recent_start).astype(int)
    work["positive_rent"] = work["Rent_Owed_num"].where(work["Rent_Owed_num"].gt(0), 0).fillna(0)
    work["nonpayment_notice"] = work["Cause_Category"].eq("Non-payment of rent").astype(int)

    portfolio = (
        work.groupby("full_address", as_index=False)
        .agg(
            notices=("Date_Filed", "size"),
            filing_dates=("Date_Filed", "nunique"),
            recent_notices=("recent_notice", "sum"),
            total_rent_owed=("positive_rent", "sum"),
            first_notice=("Date_Filed", "min"),
            latest_notice=("Date_Filed", "max"),
            nonpayment_notices=("nonpayment_notice", "sum"),
            Council_District=("Council_District", "first"),
            Zip=("Zip", "first"),
            lat=("lat", "first"),
            long=("long", "first"),
        )
    )
    portfolio["nonpayment_share"] = portfolio["nonpayment_notices"] / portfolio["notices"] * 100

    component_weights = {
        "notices": 0.30,
        "filing_dates": 0.25,
        "recent_notices": 0.25,
        "total_rent_owed": 0.20,
    }
    for field in component_weights:
        portfolio[f"{field}_percentile"] = portfolio[field].rank(method="average", pct=True) * 100

    portfolio["risk_score"] = sum(
        portfolio[f"{field}_percentile"] * weight
        for field, weight in component_weights.items()
    ).round(1)
    portfolio["priority_tier"] = pd.cut(
        portfolio["risk_score"],
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=["Baseline", "Monitor", "Elevated", "High priority"],
        right=False,
    ).astype(str)
    portfolio["recent_window_start"] = recent_start
    portfolio["reference_end"] = reference_end
    return portfolio.sort_values(
        ["risk_score", "notices", "filing_dates"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _forecast_snapshot(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    include_target: bool = True,
) -> pd.DataFrame:
    """Build one leakage-safe address snapshot using only data known at cutoff."""
    cutoff = pd.Timestamp(cutoff).normalize()
    history_start = cutoff - pd.Timedelta(days=FORECAST_LOOKBACK_DAYS - 1)
    history = df[
        df["full_address"].ne("Unknown address")
        & df["Date_Filed"].between(history_start, cutoff)
    ].copy()
    if history.empty:
        return pd.DataFrame()

    age_days = (cutoff - history["Date_Filed"].dt.normalize()).dt.days
    history["in_30"] = age_days.le(29).astype(int)
    history["in_90"] = age_days.le(89).astype(int)
    history["in_prior_90"] = age_days.between(90, 179).astype(int)
    history["positive_rent"] = history["Rent_Owed_num"].where(history["Rent_Owed_num"].gt(0), 0).fillna(0)
    history["rent_reported"] = history["Rent_Owed_num"].gt(0).astype(int)
    history["nonpayment"] = history["Cause_Category"].eq("Non-payment of rent").astype(int)
    history["filing_date_90"] = history["Date_Filed"].where(history["in_90"].eq(1))
    history["activity_month"] = history["Date_Filed"].dt.to_period("M")

    snapshot = (
        history.groupby("full_address", as_index=False)
        .agg(
            notices_30=("in_30", "sum"),
            notices_90=("in_90", "sum"),
            notices_prior_90=("in_prior_90", "sum"),
            notices_365=("Date_Filed", "size"),
            filing_dates_90=("filing_date_90", "nunique"),
            filing_dates_365=("Date_Filed", "nunique"),
            active_months=("activity_month", "nunique"),
            latest_notice=("Date_Filed", "max"),
            nonpayment_notices_365=("nonpayment", "sum"),
            rent_reported_365=("rent_reported", "sum"),
            rent_owed_365=("positive_rent", "sum"),
            Council_District=("Council_District", "last"),
            Zip=("Zip", "last"),
            lat=("lat", "last"),
            long=("long", "last"),
        )
    )
    snapshot["days_since_last"] = (cutoff - snapshot["latest_notice"].dt.normalize()).dt.days.clip(0, FORECAST_LOOKBACK_DAYS)
    snapshot["nonpayment_share_365"] = snapshot["nonpayment_notices_365"] / snapshot["notices_365"]
    snapshot["rent_reporting_share_365"] = snapshot["rent_reported_365"] / snapshot["notices_365"]
    snapshot["momentum_raw"] = snapshot["notices_90"] - snapshot["notices_prior_90"]
    snapshot["momentum_label"] = np.select(
        [snapshot["momentum_raw"].gt(0), snapshot["momentum_raw"].lt(0)],
        ["Rising", "Falling"],
        default="Stable",
    )
    snapshot["log_notices_30"] = np.log1p(snapshot["notices_30"])
    snapshot["log_notices_90"] = np.log1p(snapshot["notices_90"])
    snapshot["log_notices_365"] = np.log1p(snapshot["notices_365"])
    snapshot["log_filing_dates_90"] = np.log1p(snapshot["filing_dates_90"])
    snapshot["log_filing_dates_365"] = np.log1p(snapshot["filing_dates_365"])
    snapshot["recency"] = 1 - snapshot["days_since_last"] / FORECAST_LOOKBACK_DAYS
    snapshot["momentum"] = np.sign(snapshot["momentum_raw"]) * np.log1p(snapshot["momentum_raw"].abs())
    snapshot["log_rent_owed_365"] = np.log1p(snapshot["rent_owed_365"])
    snapshot["active_month_share"] = snapshot["active_months"] / 13
    snapshot["filing_days_per_notice"] = snapshot["filing_dates_365"] / snapshot["notices_365"]
    snapshot["recent_activity_share"] = snapshot["notices_90"] / snapshot["notices_365"]
    snapshot["recency_squared"] = snapshot["recency"] ** 2
    snapshot["recency_frequency"] = snapshot["recency"] * snapshot["log_notices_90"]
    snapshot["cutoff"] = cutoff

    if include_target:
        future_end = cutoff + pd.Timedelta(days=FORECAST_HORIZON_DAYS)
        future = df[
            df["Date_Filed"].gt(cutoff)
            & df["Date_Filed"].le(future_end)
            & df["full_address"].ne("Unknown address")
        ]
        target = future.groupby("full_address").size()
        snapshot["target_any_notice_90"] = snapshot["full_address"].map(target).fillna(0).gt(0).astype(int)
        snapshot["target_notice_count_90"] = snapshot["full_address"].map(target).fillna(0).astype(int)
    return snapshot


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic link."""
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float = 1.0,
    max_iter: int = 60,
) -> np.ndarray:
    """Fit a small L2-regularized logistic model with Newton updates."""
    design = np.column_stack([np.ones(len(x)), x]).astype(float)
    target = y.astype(float)
    beta = np.zeros(design.shape[1], dtype=float)
    penalty_matrix = np.eye(design.shape[1]) * penalty
    penalty_matrix[0, 0] = 0.0

    for _ in range(max_iter):
        probabilities = _sigmoid(design @ beta)
        weights = np.clip(probabilities * (1 - probabilities), 1e-6, None)
        information = design.T @ (design * weights[:, None]) + penalty_matrix
        gradient = design.T @ (target - probabilities) - penalty_matrix @ beta
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        beta += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def _average_precision(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Average precision, equal to area under the stepwise precision-recall curve."""
    positives = int(np.sum(y_true))
    if positives == 0:
        return 0.0
    order = np.argsort(-probabilities, kind="mergesort")
    ranked = y_true[order]
    # Evaluate at distinct thresholds so tied scores cannot depend on row order.
    ends = np.r_[np.flatnonzero(np.diff(probabilities[order])), len(order) - 1]
    hits = np.cumsum(ranked)[ends]
    precision = hits / (ends + 1)
    return float(np.sum(precision * np.diff(np.r_[0, hits])) / positives)


def _cluster_probability_intervals(x_train, y_train, beta, groups, x_score, penalty):
    """Approximate 95% Wald intervals, clustering repeated snapshots by address.

    The sandwich uses the penalized information matrix. Features, regularization
    and model selection are held fixed; these bounds do not cover those sources
    of uncertainty, reporting errors, or future policy/distribution changes.
    """
    from scipy.stats import t

    design = np.column_stack([np.ones(len(x_train)), x_train])
    score_design = np.column_stack([np.ones(len(x_score)), x_score])
    probabilities = _sigmoid(design @ beta)
    group_codes, unique_groups = pd.factorize(groups)
    n, k = design.shape
    g = len(unique_groups)
    if g < 2 or n <= k:
        return np.full(len(x_score), np.nan), np.full(len(x_score), np.nan), g
    penalty_matrix = np.eye(k) * penalty
    penalty_matrix[0, 0] = 0
    information = design.T @ (design * (probabilities * (1 - probabilities))[:, None]) + penalty_matrix
    bread = np.linalg.pinv(information)
    cluster_scores = np.zeros((g, k))
    np.add.at(cluster_scores, group_codes, design * (y_train - probabilities)[:, None])
    meat = cluster_scores.T @ cluster_scores
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread.T
    variance = np.einsum("ij,jk,ik->i", score_design, covariance, score_design).clip(min=0)
    logit = score_design @ beta
    margin = float(t.ppf(.975, g - 1)) * np.sqrt(variance)
    return _sigmoid(logit - margin), _sigmoid(logit + margin), g


def _model_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return proper scoring and ranking metrics for a probability forecast."""
    return {
        "brier_score": float(np.mean((probabilities - y_true) ** 2)),
        "average_precision": _average_precision(y_true, probabilities),
        "event_rate": float(np.mean(y_true)),
    }


def _calibration_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Create held-out probability bins and 95% Wilson intervals for group rates."""
    frame = pd.DataFrame({"actual": y_true.astype(int), "predicted": probabilities.astype(float)})
    unique = int(frame["predicted"].nunique())
    if unique < 2:
        frame["bin"] = "All scores"
    else:
        bins = min(10, unique)
        frame["bin"] = pd.qcut(frame["predicted"], q=bins, duplicates="drop")
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            addresses=("actual", "size"),
            predicted_probability=("predicted", "mean"),
            observed_rate=("actual", "mean"),
            events=("actual", "sum"),
        )
        .reset_index(drop=True)
    )
    z = 1.959963984540054
    n = grouped["addresses"].astype(float)
    p = grouped["observed_rate"].astype(float)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2)) / denominator
    grouped["interval_low"] = (center - margin).clip(0, 1)
    grouped["interval_high"] = (center + margin).clip(0, 1)
    return grouped


def _assign_empirical_intervals(
    probabilities: np.ndarray,
    calibration: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Attach each live score to the nearest held-out calibration bin."""
    if calibration.empty:
        return np.full(len(probabilities), np.nan), np.full(len(probabilities), np.nan)
    centers = calibration["predicted_probability"].to_numpy(dtype=float)
    nearest = np.abs(probabilities[:, None] - centers[None, :]).argmin(axis=1)
    return (
        calibration["interval_low"].to_numpy(dtype=float)[nearest],
        calibration["interval_high"].to_numpy(dtype=float)[nearest],
    )


def _calibrate_live_probabilities(
    probabilities: np.ndarray,
    calibration: pd.DataFrame,
) -> np.ndarray:
    """Map raw scores to a monotone interpolation of held-out observed rates."""
    if calibration.empty:
        return probabilities
    ordered = calibration.sort_values("predicted_probability")
    centers = ordered["predicted_probability"].to_numpy(dtype=float)
    observed = np.maximum.accumulate(ordered["observed_rate"].to_numpy(dtype=float))
    return np.interp(probabilities, centers, observed, left=observed[0], right=observed[-1])


def _ridge_predictive_building_forecast(
    df: pd.DataFrame,
    as_of_date: object | None = None,
) -> dict[str, object]:
    """Train, backtest, and score a transparent 90-day recurrence forecast.

    Eligible addresses have at least one notice in the trailing 365 days. The
    holdout is the most recent fully observable 90-day snapshot; all features
    are calculated as they would have been known at their cutoff.
    """
    work = df[df["full_address"].ne("Unknown address")].copy()
    if work.empty:
        return {"forecast": pd.DataFrame(), "calibration": pd.DataFrame(), "coefficients": pd.DataFrame()}
    work = work.sort_values("Date_Filed")
    available_end = work["Date_Filed"].max().normalize()
    as_of = min(pd.Timestamp(as_of_date).normalize(), available_end) if as_of_date is not None else available_end
    work = work[work["Date_Filed"].le(as_of)].copy()
    if work.empty:
        return {"forecast": pd.DataFrame(), "calibration": pd.DataFrame(), "coefficients": pd.DataFrame()}

    recent_floor = work["Date_Filed"].min().normalize()
    first_cutoff = recent_floor + pd.Timedelta(days=FORECAST_LOOKBACK_DAYS)
    last_target_cutoff = as_of - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
    if first_cutoff >= last_target_cutoff:
        return {"forecast": pd.DataFrame(), "calibration": pd.DataFrame(), "coefficients": pd.DataFrame()}

    # Anchor backwards from the latest mature outcome. Adjacent target windows
    # do not overlap, including the final window (no short appended interval).
    cutoffs = list(reversed(pd.date_range(last_target_cutoff, first_cutoff, freq="-90D")))
    snapshots = [_forecast_snapshot(work, cutoff, include_target=True) for cutoff in cutoffs]
    snapshots = [snapshot for snapshot in snapshots if not snapshot.empty]
    if len(snapshots) < 4:
        return {"forecast": pd.DataFrame(), "calibration": pd.DataFrame(), "coefficients": pd.DataFrame()}

    holdout = snapshots[-1].copy()
    validation = snapshots[-2].copy()
    training = pd.concat(snapshots[:-2], ignore_index=True)
    feature_mean = training[FORECAST_FEATURES].mean().to_numpy(dtype=float)
    feature_scale = training[FORECAST_FEATURES].std(ddof=0).replace(0, 1).fillna(1).to_numpy(dtype=float)
    x_train = (training[FORECAST_FEATURES].to_numpy(dtype=float) - feature_mean) / feature_scale
    y_train = training["target_any_notice_90"].to_numpy(dtype=int)
    x_holdout = (holdout[FORECAST_FEATURES].to_numpy(dtype=float) - feature_mean) / feature_scale
    y_holdout = holdout["target_any_notice_90"].to_numpy(dtype=int)

    baseline_features = ["log_notices_90", "recency"]
    baseline_indexes = [FORECAST_FEATURES.index(feature) for feature in baseline_features]
    x_validation = (validation[FORECAST_FEATURES].to_numpy(dtype=float) - feature_mean) / feature_scale
    y_validation = validation["target_any_notice_90"].to_numpy(dtype=int)
    baseline_beta = _fit_ridge_logistic(x_train[:, baseline_indexes], y_train, penalty=2.0)
    baseline_validation = _sigmoid(np.column_stack([np.ones(len(validation)), x_validation[:, baseline_indexes]]) @ baseline_beta)
    validation_baseline_metrics = _model_metrics(y_validation, baseline_validation)
    trials = []
    for penalty in (0.5, 10.0, 100.0):
        beta = _fit_ridge_logistic(x_train, y_train, penalty=penalty)
        probability = _sigmoid(np.column_stack([np.ones(len(validation)), x_validation]) @ beta)
        metrics = _model_metrics(y_validation, probability)
        trials.append((metrics["brier_score"], penalty, metrics))
    _, selected_penalty, validation_candidate_metrics = min(trials, key=lambda trial: trial[0])
    promotion_gate = (
        validation_candidate_metrics["brier_score"] < validation_baseline_metrics["brier_score"]
        and validation_candidate_metrics["average_precision"] >= validation_baseline_metrics["average_precision"]
    )
    # Freeze the choice before looking at the final test. Refit only on outcomes
    # fully observed by its forecast origin; repeated addresses are intentional.
    test_training = pd.concat(snapshots[:-1], ignore_index=True)
    test_mean = test_training[FORECAST_FEATURES].mean().to_numpy(dtype=float)
    test_scale = test_training[FORECAST_FEATURES].std(ddof=0).replace(0, 1).fillna(1).to_numpy(dtype=float)
    x_train = (test_training[FORECAST_FEATURES].to_numpy(dtype=float) - test_mean) / test_scale
    y_train = test_training["target_any_notice_90"].to_numpy(dtype=int)
    x_holdout = (holdout[FORECAST_FEATURES].to_numpy(dtype=float) - test_mean) / test_scale
    candidate_beta = _fit_ridge_logistic(x_train, y_train, penalty=selected_penalty)
    candidate_holdout = _sigmoid(np.column_stack([np.ones(len(x_holdout)), x_holdout]) @ candidate_beta)
    baseline_beta = _fit_ridge_logistic(x_train[:, baseline_indexes], y_train, penalty=2.0)
    baseline_holdout = _sigmoid(
        np.column_stack([np.ones(len(x_holdout)), x_holdout[:, baseline_indexes]]) @ baseline_beta
    )
    candidate_metrics = _model_metrics(y_holdout, candidate_holdout)
    baseline_metrics = _model_metrics(y_holdout, baseline_holdout)
    served_holdout = candidate_holdout if promotion_gate else baseline_holdout
    served_metrics = _model_metrics(y_holdout, served_holdout)
    calibration = _calibration_table(y_holdout, served_holdout)

    all_training = pd.concat(snapshots, ignore_index=True)
    final_mean = all_training[FORECAST_FEATURES].mean().to_numpy(dtype=float)
    final_scale = all_training[FORECAST_FEATURES].std(ddof=0).replace(0, 1).fillna(1).to_numpy(dtype=float)
    final_x = (all_training[FORECAST_FEATURES].to_numpy(dtype=float) - final_mean) / final_scale
    final_y = all_training["target_any_notice_90"].to_numpy(dtype=int)
    if promotion_gate:
        final_beta = _fit_ridge_logistic(final_x, final_y, penalty=selected_penalty)
    else:
        baseline_final = _fit_ridge_logistic(final_x[:, baseline_indexes], final_y, penalty=2.0)
        final_beta = np.zeros(len(FORECAST_FEATURES) + 1)
        final_beta[0] = baseline_final[0]
        final_beta[np.array(baseline_indexes) + 1] = baseline_final[1:]

    live = _forecast_snapshot(work, as_of, include_target=False)
    live_x = (live[FORECAST_FEATURES].to_numpy(dtype=float) - final_mean) / final_scale
    live_design = np.column_stack([np.ones(len(live_x)), live_x])
    raw_probabilities = _sigmoid(live_design @ final_beta)
    live["uncalibrated_probability"] = raw_probabilities
    # Calibration is an independent test diagnostic, not a transformation fitted
    # on the test labels. These are the same type of scores evaluated above.
    live["forecast_probability"] = raw_probabilities
    interval_indexes = list(range(len(FORECAST_FEATURES))) if promotion_gate else baseline_indexes
    interval_beta = final_beta if promotion_gate else baseline_final
    interval_low, interval_high, interval_clusters = _cluster_probability_intervals(
        final_x[:, interval_indexes], final_y, interval_beta,
        all_training["full_address"], live_x[:, interval_indexes],
        selected_penalty if promotion_gate else 2.0,
    )
    live["ci_lower_95"] = interval_low
    live["ci_upper_95"] = interval_high
    live["forecast_tier"] = pd.cut(
        live["forecast_probability"],
        bins=[-np.inf, 0.15, 0.30, 0.50, np.inf],
        labels=["Lower", "Watch", "Elevated", "High"],
        right=False,
    ).astype(str)

    contributions = live_x * final_beta[1:]
    contribution_columns = [f"contribution_{feature}" for feature in FORECAST_FEATURES]
    live[contribution_columns] = contributions
    top_driver_indexes = np.argsort(-contributions, axis=1)[:, :3]
    live["top_drivers"] = [
        ", ".join(FORECAST_FEATURE_LABELS[FORECAST_FEATURES[index]] for index in indexes if row[index] > 0)
        or "Model baseline"
        for row, indexes in zip(contributions, top_driver_indexes)
    ]
    live = live.sort_values(["forecast_probability", "notices_90"], ascending=False).reset_index(drop=True)

    coefficients = pd.DataFrame(
        {
            "feature": FORECAST_FEATURES,
            "feature_label": [FORECAST_FEATURE_LABELS[feature] for feature in FORECAST_FEATURES],
            "standardized_coefficient": final_beta[1:],
        }
    ).sort_values("standardized_coefficient", ascending=False)
    metadata = {
        "model_version": FORECAST_MODEL_VERSION,
        "forecast_as_of": as_of,
        "trained_through": pd.Timestamp(cutoffs[-1]) + pd.Timedelta(days=FORECAST_HORIZON_DAYS),
        "holdout_cutoff": pd.Timestamp(cutoffs[-1]),
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "lookback_days": FORECAST_LOOKBACK_DAYS,
        "training_rows": int(len(all_training)),
        "training_snapshots": int(len(snapshots)),
        "holdout_addresses": int(len(holdout)),
        "promotion_gate_passed": bool(promotion_gate),
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "served_metrics": served_metrics,
        "selected_model": "Enhanced recurrence model" if promotion_gate else "Recent-history baseline",
        "selected_penalty": selected_penalty if promotion_gate else 2.0,
        "selection_cutoff": pd.Timestamp(cutoffs[-2]),
        "validation_candidate_metrics": validation_candidate_metrics,
        "validation_baseline_metrics": validation_baseline_metrics,
        "test_training_outcomes_through": pd.Timestamp(cutoffs[-2]) + pd.Timedelta(days=90),
        "forecast_end": as_of + pd.Timedelta(days=90),
        "confidence_level": .95,
        "interval_method": "Approximate address-clustered Wald interval on the logistic scale",
        "interval_clusters": interval_clusters,
    }
    return {
        "forecast": live,
        "calibration": calibration,
        "coefficients": coefficients,
        "metadata": metadata,
    }


def predictive_building_forecast(df, as_of_date=None):
    """Compare XGBoost with ridge models without using final test labels to select."""
    from xgboost_forecast import predictive_forecast
    return predictive_forecast(df, as_of_date)
