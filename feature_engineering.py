"""Reusable analytics for the EvictorBookLA Atlas Streamlit application."""

from __future__ import annotations

import json
from io import BytesIO

import numpy as np
import pandas as pd


def property_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filing activity by parcel (APN), the best available owner proxy."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["Apn"] = work["Apn"].fillna("").astype(str)
    work.loc[work["Apn"].isin(["", "nan", "None"]), "Apn"] = "Unknown"
    return (
        work.groupby("Apn", as_index=False)
        .agg(
            addresses=("full_address", "nunique"),
            filing_nodes=("full_address", "size"),
            affected_tenants=("Eviction_Count", "sum"),
            total_rent_owed=("Total_Rent_Owed", "sum"),
            first_filing=("Date_Filed", "min"),
            latest_filing=("Date_Filed", "max"),
            repeat_addresses=("repeat_count", lambda s: int((s > 1).sum())),
        )
        .sort_values(["affected_tenants", "filing_nodes"], ascending=False)
    )


def address_profiles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("full_address", as_index=False)
        .agg(
            apns=("Apn", lambda s: ", ".join(sorted(set(map(str, s))))),
            filing_nodes=("Date_Filed", "size"),
            affected_tenants=("Eviction_Count", "sum"),
            total_rent_owed=("Total_Rent_Owed", "sum"),
            first_filing=("Date_Filed", "min"),
            latest_filing=("Date_Filed", "max"),
            causes=("Cause", lambda s: ", ".join(sorted({x for x in s if x}))),
        )
        .sort_values(["affected_tenants", "filing_nodes"], ascending=False)
    )


def monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "evictions"])
    work = df.copy()
    work["month"] = work["Date_Filed"].dt.to_period("M").dt.to_timestamp()
    trend = (
        work.groupby("month", as_index=False)["Eviction_Count"]
        .sum()
        .rename(columns={"Eviction_Count": "evictions"})
        .sort_values("month")
    )
    # Preserve calendar gaps instead of silently compressing time.
    months = pd.date_range(trend["month"].min(), trend["month"].max(), freq="MS")
    return trend.set_index("month").reindex(months).rename_axis("month").reset_index()


def anomaly_table(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    trend = monthly_series(df)
    if trend.empty:
        return trend
    trend["baseline"] = trend["evictions"].shift(1).rolling(window, min_periods=window).mean()
    trend["baseline_sd"] = trend["evictions"].shift(1).rolling(window, min_periods=window).std()
    trend["z_score"] = (trend["evictions"] - trend["baseline"]) / trend["baseline_sd"].replace(0, np.nan)
    trend["change_pct"] = (trend["evictions"] / trend["baseline"].replace(0, np.nan) - 1) * 100
    trend["is_alert"] = trend["z_score"].abs() >= 2
    return trend


def grid_hotspots(df: pd.DataFrame, precision: int = 2) -> pd.DataFrame:
    """Transparent grid hotspot score; this is descriptive, not a significance test."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["grid_lat"] = work["lat"].round(precision)
    work["grid_long"] = work["long"].round(precision)
    out = (
        work.groupby(["grid_lat", "grid_long"], as_index=False)
        .agg(
            affected_tenants=("Eviction_Count", "sum"),
            filing_nodes=("full_address", "size"),
            unique_addresses=("full_address", "nunique"),
            total_rent_owed=("Total_Rent_Owed", "sum"),
        )
    )
    sd = out["affected_tenants"].std()
    out["hotspot_z"] = 0.0 if not sd or np.isnan(sd) else (
        out["affected_tenants"] - out["affected_tenants"].mean()
    ) / sd
    return out.sort_values("hotspot_z", ascending=False)


def forecast_monthly(df: pd.DataFrame, months: int = 6) -> tuple[pd.DataFrame, dict]:
    """OLS mean confidence and future-observation prediction intervals (95%)."""
    from scipy.stats import t
    trend = monthly_series(df)
    if len(trend) < 4:
        return pd.DataFrame(), {"mae": np.nan, "method": "insufficient history"}
    if trend["evictions"].isna().any():
        return pd.DataFrame(), {"mae": np.nan, "method": "Missing calendar months in the filtered series; forecast withheld."}
    y = trend["evictions"].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    holdout = min(3, max(1, len(y) // 4))
    train_n = len(y) - holdout
    train_coef = np.polyfit(x[:train_n], y[:train_n], 1)
    backtest = np.maximum(0, np.polyval(train_coef, x[train_n:]))
    mae = float(np.mean(np.abs(backtest - y[train_n:])))
    coef = np.polyfit(x, y, 1)
    fitted = np.polyval(coef, x)
    # Two fitted coefficients consume two residual degrees of freedom.
    residual_variance = float(np.sum((y - fitted) ** 2) / (len(y) - 2))
    resid_sd = np.sqrt(residual_variance)
    future_x = np.arange(len(y), len(y) + months, dtype=float)
    raw_pred = np.polyval(coef, future_x)
    pred = np.maximum(0, raw_pred)
    leverage = 1 / len(y) + (future_x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    critical = float(t.ppf(.975, df=len(y) - 2))
    mean_margin = critical * np.sqrt(residual_variance * leverage)
    prediction_margin = critical * np.sqrt(residual_variance * (1 + leverage))
    dates = pd.date_range(trend["month"].max() + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    forecast = pd.DataFrame(
        {
            "month": dates,
            "forecast": pred,
            "ci_lower_95": np.maximum(0, raw_pred - mean_margin),
            "ci_upper_95": np.maximum(0, raw_pred + mean_margin),
            "lower": np.maximum(0, raw_pred - prediction_margin),
            "upper": np.maximum(0, raw_pred + prediction_margin),
        }
    )
    return forecast, {"mae": mae, "method": "linear trend", "residual_sd": resid_sd,
                      "confidence_level": .95, "residual_df": len(y) - 2}


def correlation_table(neighborhood_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = neighborhood_metrics.select_dtypes(include=[np.number])
    if "evictions_total" not in numeric or len(numeric.columns) < 2:
        return pd.DataFrame()
    rows = []
    for metric in numeric.columns.drop("evictions_total"):
        pairs = numeric[[metric, "evictions_total"]].replace([np.inf, -np.inf], np.nan).dropna()
        valid = len(pairs) >= 3 and pairs[metric].nunique() > 1 and pairs["evictions_total"].nunique() > 1
        rows.append({"metric": metric,
                     "correlation_with_evictions": pairs[metric].corr(pairs["evictions_total"]) if valid else np.nan,
                     "paired_neighborhoods": len(pairs),
                     "status": "Descriptive only" if valid else "Insufficient pairs or constant values"})
    return pd.DataFrame(rows).sort_values("correlation_with_evictions", key=abs, ascending=False)


def routine_filer_summary(df, minimum_active_months=6, frequency="Y"):
    """Retrospective address cohorts defined by distinct active calendar months."""
    work = df.dropna(subset=["Date_Filed", "full_address"]).copy()
    work = work[work["full_address"].astype(str).str.strip().ne("")]
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["active_month"] = work["Date_Filed"].dt.to_period("M")
    addresses = work.groupby("full_address", as_index=False).agg(
        active_months=("active_month", "nunique"), notices=("Eviction_Count", "sum"),
        first_received=("Date_Filed", "min"), last_received=("Date_Filed", "max"))
    addresses["cohort"] = np.where(addresses["active_months"] >= minimum_active_months,
                                   "Routine", "Infrequent")
    work = work.merge(addresses[["full_address", "cohort"]], on="full_address", validate="many_to_one")
    work["period"] = work["Date_Filed"].dt.to_period(frequency)
    periods = pd.period_range(work["period"].min(), work["period"].max(), freq=frequency)
    index = pd.MultiIndex.from_product([periods, ["Routine", "Infrequent"]], names=["period", "cohort"])
    summary = work.groupby(["period", "cohort"])["Eviction_Count"].sum().reindex(index, fill_value=0).rename("notices").reset_index()
    totals = summary.groupby("period")["notices"].transform("sum")
    summary["share_percent"] = summary["notices"].div(totals.replace(0, np.nan)).mul(100)
    summary["period"] = summary["period"].dt.to_timestamp()
    return summary, addresses.sort_values(["active_months", "notices"], ascending=False)


def selection_geojson(df: pd.DataFrame) -> bytes:
    features = []
    for row in df.itertuples():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row.long, row.lat]},
                "properties": {
                    "address": row.full_address,
                    "date": row.Date_Filed.strftime("%Y-%m-%d"),
                    "eviction_count": int(row.Eviction_Count),
                    "rent_owed": float(row.Total_Rent_Owed),
                    "apn": str(row.Apn),
                },
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")


def report_html(df: pd.DataFrame, area: str, date_label: str) -> bytes:
    total = int(df["Eviction_Count"].sum()) if not df.empty else 0
    rent = float(df["Total_Rent_Owed"].sum()) if not df.empty else 0
    top = address_profiles(df).head(20)
    table = top.to_html(index=False, border=0, classes="data") if not top.empty else "<p>No records.</p>"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>EvictorBookLA Atlas report</title>
    <style>body{{font:15px system-ui;max-width:1000px;margin:40px auto;color:#172033}}h1{{color:#084594}}
    .cards{{display:flex;gap:16px}}.card{{padding:14px;background:#eef5ff;border-radius:8px}}table{{border-collapse:collapse;width:100%}}
    td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}@media print{{button{{display:none}}}}</style></head>
    <body><button onclick='print()'>Print / Save PDF</button><h1>EvictorBookLA Atlas filtered report</h1>
    <p><b>Area:</b> {area}<br><b>Period:</b> {date_label}</p><div class='cards'>
    <div class='card'><b>{total:,}</b><br>Affected tenants</div><div class='card'><b>{len(df):,}</b><br>Address-date nodes</div>
    <div class='card'><b>${rent:,.0f}</b><br>Reported rent owed</div></div><h2>Leading addresses</h2>{table}
    <h2>Method note</h2><p>Counts represent eviction notices in the source data, not court judgments or verified physical removals. Rent owed is reported, not independently verified.</p></body></html>"""
    return html.encode("utf-8")
