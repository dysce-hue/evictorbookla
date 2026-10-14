"""Focused presentation for the BETA address-recurrence forecast."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_forecast(result, eligible_addresses, records, inspector):
    forecast = result.get("forecast", pd.DataFrame())
    meta = result.get("metadata", {})
    if forecast.empty or not meta:
        st.info("More history is needed for separate training, selection, and 90-day testing. Choose a later end date.")
        return
    view = forecast[forecast.full_address.isin(eligible_addresses)].copy()
    st.caption(
        f"Forecast window: {meta['forecast_as_of'] + pd.Timedelta(days=1):%b %d, %Y}–"
        f"{meta['forecast_end']:%b %d, %Y} · Source records through {meta['forecast_as_of']:%b %d, %Y}"
    )
    st.info("Experimental probability of at least one new LAHD notice at an address observed in the past year. Not a prediction of completed evictions or wrongdoing.")
    st.caption("The sidebar end date sets the forecast origin. Other sidebar filters select displayed addresses; training and validation use the full source history through that date.")
    outlook, performance, methods = st.tabs(["Address outlook", "Model performance", "How it works"])
    with outlook:
        if view.empty:
            st.info("No eligible addresses match the sidebar filters.")
        else:
            search = st.text_input("Find an address", placeholder="Street number or street name", key="forecast_search")
            with st.expander("Refine the address list"):
                left, right = st.columns(2)
                minimum = left.slider("Minimum 90-day probability", 0, 100, 0, format="%d%%", key="forecast_minimum_v2")
                count = right.selectbox("Addresses in chart", [10, 15, 25], index=0)
            view = view[view.full_address.str.contains(search.strip(), case=False, regex=False) & view.forecast_probability.ge(minimum / 100)]
            if view.empty:
                st.info("No matches. Clear the address search or lower the minimum probability.")
            else:
                cards = st.columns(3)
                cards[0].metric("Matching addresses", f"{len(view):,}")
                cards[1].metric("Median probability", f"{view.forecast_probability.median():.1%}")
                cards[2].metric("Probability ≥50%", f"{view.forecast_probability.ge(.5).sum():,}")
                st.subheader("Highest estimated recurrence")
                top = view.head(count).sort_values("forecast_probability")
                fig = go.Figure(go.Bar(
                    x=top.forecast_probability, y=top.full_address, orientation="h",
                    marker_color="#3b6fb6", text=top.forecast_probability.map(lambda p: f"{p:.1%}"),
                    textposition="outside", cliponaxis=False,
                    error_x=dict(type="data", symmetric=False,
                                 array=top.ci_upper_95-top.forecast_probability,
                                 arrayminus=top.forecast_probability-top.ci_lower_95, color="#172033"),
                    customdata=top[["notices_90", "days_since_last", "ci_lower_95", "ci_upper_95"]],
                    hovertemplate="%{y}<br>90-day probability: %{x:.1%}<br>Approx. 95% CI: %{customdata[2]:.1%}–%{customdata[3]:.1%}<br>Recent notices: %{customdata[0]}<br>Days since latest: %{customdata[1]}<extra></extra>",
                ))
                fig.update_layout(height=max(370, len(top)*32), margin=dict(l=10,r=65,t=15,b=35),
                                  xaxis=dict(range=[0,1.08], tickformat=".0%", title="Probability of a new notice"),
                                  yaxis=dict(title=None, automargin=True))
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Whiskers: {meta['interval_method']}. They are conditional on the selected model and exclude model-selection uncertainty, reporting errors, cross-address/time dependence, and policy changes. Future 0/1 outcomes do not fall inside a probability interval; nominal 95% coverage is unverified.")
                if view[["ci_lower_95", "ci_upper_95"]].isna().any().any():
                    st.warning("Some confidence intervals are unavailable. Ridge estimation needs enough independent address clusters and training rows; XGBoost needs at least 20 successful cluster-bootstrap refits. Missing bounds are not zero uncertainty.")
                table = view[["full_address", "forecast_probability", "ci_lower_95", "ci_upper_95", "forecast_tier", "notices_90", "active_months", "days_since_last", "momentum_label"]].rename(columns={
                    "full_address":"Address", "forecast_probability":"90-day probability (%)", "forecast_tier":"Tier",
                    "ci_lower_95":"95% CI lower (%)", "ci_upper_95":"95% CI upper (%)",
                    "notices_90":"Notices in 90 days", "active_months":"Active months in past year",
                    "days_since_last":"Days since latest", "momentum_label":"Momentum"})
                for field in ["90-day probability (%)", "95% CI lower (%)", "95% CI upper (%)"]:
                    table[field] *= 100
                with st.expander(f"All {len(table):,} matching addresses and download"):
                    st.dataframe(table.head(1000), hide_index=True, width="stretch", column_config={
                        "90-day probability (%)":st.column_config.ProgressColumn(format="%.1f%%",min_value=0,max_value=100),
                        "95% CI lower (%)":st.column_config.NumberColumn(format="%.1f%%"),
                        "95% CI upper (%)":st.column_config.NumberColumn(format="%.1f%%")})
                    export = table.assign(forecast_as_of=str(meta["forecast_as_of"].date()), forecast_end=str(meta["forecast_end"].date()), model=meta["selected_model"], interval_method=meta["interval_method"], interval_scope="Selected model fixed; excludes data errors, selection uncertainty and future policy changes")
                    st.download_button("Download matching forecasts", export.to_csv(index=False), "90_day_notice_forecasts.csv", "text/csv")
                    st.caption("Table shows up to 1,000 addresses; the download includes every matching address. Tiers are display thresholds, not validated intervention cutoffs.")
                with st.expander("Inspect an address and its history", expanded=True):
                    inspector(records, view)
    with performance:
        served = meta["served_metrics"]
        baseline = meta["baseline_metrics"]
        st.subheader(meta["selected_model"])
        if meta.get("xgboost_status"):
            st.caption(meta["xgboost_status"])
        if meta["promotion_gate_passed"]:
            st.success("Enhanced model selected using earlier rolling validation periods. The latest evaluation below was not used by this selection procedure.")
        else:
            st.info("The enhanced candidate did not improve both selection metrics. The simpler baseline is serving forecasts.")
        cards = st.columns(3)
        cards[0].metric("Test probability error", f"{served['brier_score']:.5f}", help="Brier score: average squared error between probability and outcome. Lower is better; not an accuracy percentage.")
        cards[1].metric("Test ranking score", f"{served['average_precision']:.4f}", help="Average precision summarizes the precision–recall curve. Higher is better; not a percentage correctly predicted.")
        cards[2].metric("Test addresses", f"{meta['holdout_addresses']:,}")
        if "precision_top_10" in served:
            st.metric("Precision among highest-risk 10%", f"{served['precision_top_10']:.1%}", help="Fraction of the highest-scored 10% of eligible test addresses with a new notice within 90 days. Ties at the boundary receive fractional inclusion. This is not precision for all addresses.")
        st.caption(f"Test outcomes: {meta['holdout_cutoff'] + pd.Timedelta(days=1):%b %d, %Y}–{meta['forecast_as_of']:%b %d, %Y}. Observed recurrence: {served['event_rate']:.1%}. This is one retrospective period, not a guarantee of future performance.")
        comparison = result.get("model_comparison", pd.DataFrame([
            {"Model":"Enhanced recurrence", **meta["candidate_metrics"]},
            {"Model":"Recent-history baseline", **baseline},
        ])).rename(columns={"brier_score":"Brier error ↓", "average_precision":"Average precision ↑", "event_rate":"Observed recurrence", "log_loss":"Log loss ↓", "precision_top_10":"Top-10% precision ↑"})
        st.dataframe(comparison.style.format({"Brier error ↓":"{:.4f}","Average precision ↑":"{:.4f}","Observed recurrence":"{:.1%}", "Log loss ↓":"{:.4f}", "Top-10% precision ↑":"{:.1%}"}), hide_index=True, width="stretch")
        st.download_button("Download independent test comparison", comparison.to_csv(index=False), "notice_model_comparison.csv", "text/csv")
        if "validation_trials" in result:
            with st.expander("Earlier model-selection scores and parameters"):
                st.dataframe(result["validation_trials"], hide_index=True, width="stretch")
                st.caption("Scores are equal-weight means across up to three earlier 90-day periods. XGBoost must lower Brier without reducing average precision or top-decile precision versus eligible ridge. Selection is frozen before the latest evaluation.")
                if "rolling_validation" in result:
                    st.download_button("Download scores for each validation period", result["rolling_validation"].to_csv(index=False), "rolling_validation.csv", "text/csv")
        calibration = result["calibration"]
        st.subheader("Do the probabilities match observed outcomes?")
        fig = go.Figure(go.Scatter(x=calibration.predicted_probability, y=calibration.observed_rate,
            mode="markers+lines", name="Selected model", marker_color="#3b6fb6",
            error_y=dict(type="data", symmetric=False, array=calibration.interval_high-calibration.observed_rate,
                         arrayminus=calibration.observed_rate-calibration.interval_low),
            customdata=calibration[["addresses"]],
            hovertemplate="Predicted: %{x:.1%}<br>Observed: %{y:.1%}<br>Addresses: %{customdata[0]}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect agreement",line=dict(dash="dot",color="#94a3b8")))
        fig.update_layout(height=420,xaxis=dict(title="Mean predicted probability",tickformat=".0%",range=[0,1]),yaxis=dict(title="Observed recurrence",tickformat=".0%",range=[0,1]),legend=dict(orientation="h"))
        st.plotly_chart(fig,width="stretch")
        st.caption("Bars show 95% Wilson confidence intervals for observed rates in test groups. These are separate from the address probability intervals. Geographic dependence and reporting delays can make them optimistic. Test outcomes are not used to remap live probabilities.")
    with methods:
        st.markdown("""### Target and coverage
Eligible addresses had at least one notice in the preceding 365 days. The target is any new notice in the next 90 days. Unseen addresses cannot be scored. Address text is not a verified parcel or ownership identifier; units may be combined or split by source formatting.

### Training and safeguards
Snapshots are 90 days apart with non-overlapping outcome windows. With at least five mature snapshots, up to three earlier validation origins each use older training snapshots and a separate immediately preceding calibration snapshot. Equal-weight mean scores select the candidate. The final origin evaluates the frozen choice. No latest-period labels tune or select models in this procedure. That period was examined during previous development, so a fresh prospective evaluation is still required. Shorter histories use the legacy ridge-only fallback.

XGBoost and enhanced ridge use the same 15 historical features. Six booster configurations test depth 2–4, L2 penalty 10/30, 160 or 320 rounds, learning rate .05/.03, minimum child weight 10/30, and optional 365-day half-life training weights. Each tests monotone sigmoid calibration minimizing either log loss or Brier error plus .001 times slope squared. XGBoost must lower mean validation Brier without reducing mean average precision or highest-risk-decile precision versus eligible ridge. Two CPU threads keep resource use bounded. No demographic characteristics or actual rent-paid data are features.

For live XGBoost forecasts, the latest mature snapshot is reserved for calibration and earlier mature snapshots train the base model. Ridge models refit on all mature snapshots. New validated source files or a changed end date trigger rebuilding. Test scores describe the historical procedure, not a direct test of the latest refit. XGBoost explanations use native TreeSHAP contributions to calibrated log-odds; gain importance is not a causal effect.

### Confidence intervals
Ridge models use approximate address-clustered sandwich-Wald intervals. XGBoost uses 30 whole-address bootstrap refits: each sampled address carries all its training and calibration snapshots, both the booster and calibration are refitted, and the bootstrap logit standard deviation supplies a normal 95% margin around the fitted logit. At least 20 refits are required. Model choice, features and hyperparameters remain fixed. These are approximate probability-confidence intervals—not intervals for completed evictions or individual 0/1 outcomes. They exclude selection uncertainty, reporting errors, cross-address/time dependence and future policy changes. Thirty refits are a computational compromise, not evidence of established 95% coverage.

### Limits
These are observational notice records, not verified eviction outcomes. Historical snapshots are reconstructed from the current extract, without original publication timestamps; later backfills may bias retrospective tests. Source omissions, delayed reports, policy changes, address mismatches, and lack of building-unit denominators can bias forecasts. Missing records are not proof that no notice occurred. Use this for research and supportive outreach—not tenant screening or punitive decisions. Additional independent periods and fairness checks are needed before operational deployment.""")
        st.caption(f"Version: {meta['model_version']} · {meta['training_snapshots']} snapshots · {meta['training_rows']:,} address-snapshots · {meta['selected_model']}")
        if meta.get("selected_parameters"):
            st.json(meta["selected_parameters"])
        if not result.get("feature_importance", pd.DataFrame()).empty:
            st.dataframe(result["feature_importance"].rename(columns={"feature_label":"Historical indicator", "gain":"XGBoost gain importance"}), hide_index=True, width="stretch")
            st.caption("Gain summarizes training split improvements, not causal effects or percentage contributions to eviction risk.")
        elif not result["coefficients"].empty:
            st.dataframe(result["coefficients"].rename(columns={"feature_label":"Historical indicator","standardized_coefficient":"Log-odds coefficient"})[["Historical indicator","Log-odds coefficient"]], hide_index=True, width="stretch")
