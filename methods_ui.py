"""Code-grounded methods reference shared by ALPHA, BETA and the methods page."""
import pandas as pd
import streamlit as st


def topic(identifier, title, kind, where, text, equations, sources):
    return dict(id=identifier, title=title, kind=kind, where=where,
                text=text, equations=equations, sources=sources)


METHODS = [
    topic("counting", "Source grain, aggregation and missing data", "Counting rules", "ALPHA + BETA",
          "One LAHD row is a reported notice—not a unique tenant, apartment or completed eviction. "
          "Date_Filed means Date Received, not Notice Date. ALPHA excludes invalid dates or missing coordinates, then groups by address and received date; "
          "Eviction_Count is rows in that node, and Total_Rent_Owed sums numerically parsed reported amounts (not just positive values). "
          "BETA keeps valid-date rows even without coordinates and substitutes ‘Unknown address’ for blanks, but excludes that placeholder from forecast and attention scores. "
          "ALPHA repeat_count counts address-date nodes across loaded mapped history, before later UI filters. "
          "Exact duplicate rows are removed during import; identical real notices cannot always be distinguished without stable IDs. "
          "Currency characters are removed before numeric conversion. District 0 becomes missing. APN is a parcel reference, not verified ownership. "
          "An uploaded owner crosswalk is grouped after an APN join; duplicate crosswalk keys can multiply counts and must be checked.",
          [r"N(S)=\sum_i\mathbf{1}(i\in S),\quad N_{a,t}=\sum_i\mathbf{1}(a_i=a,t_i=t)",
           r"O_{a,t}=\sum_{i:a_i=a,t_i=t}O_i,\qquad A(S)=|\{a_i:i\in S\}|"],
          "app.py: load_evictions, load_rent_records; policy_metrics.py: prepare_notices; data_refresh.py"),
    topic("time_composition", "Monthly counts, cumulative activity and cause shares", "Descriptive statistics", "ALPHA + BETA",
          "Monthly totals sum notice counts, not address-date node counts. BETA's cumulative chart sums daily counts by year and day-of-year; leap years shift day-of-year alignment after February. "
          "Calendar heatmap cells pool month-of-year × weekday across selected years; they are totals, not adjusted for the number of Mondays or months observed. "
          "Cause shares divide category counts by the relevant district or displayed population. BETA's current vectorized categorization treats any comma-containing cause as multiple; "
          "ALPHA deduplicates split cause terms after node aggregation. Cause shares therefore need not be directly equivalent between pages. "
          "Zero heatmap cells mean no selected observed records; they do not prove source completeness.",
          [r"N_m=\sum_{i:month_i=m}1,\quad C_y(d)=\sum_{u\le d}N_{y,u}",
           r"H_{m,w}=\sum_i\mathbf{1}(month_i=m,weekday_i=w),\quad Share_{d,c}=100\frac{N_{d,c}}{\sum_cN_{d,c}}"],
          "feature_engineering.py: monthly_series; policy_metrics.py: annual_cumulative, calendar_pattern, cause_district; app.py: causes"),
    topic("rent", "Reported rent owed: means, quantiles, trimming and density", "Descriptive distributions", "ALPHA + BETA",
          "Notice-level rent views keep positive amounts and, for trimmed distributions, values ≤ the population's 95th percentile. Ties mean this may remove fewer than exactly 5%. "
          "Pandas/NumPy default quantiles use linear interpolation between zero-based sorted observations. BETA's cause boxes recompute 5th, 25th, 50th, 75th and 95th percentiles after global trimming. "
          "ALPHA's address-date histogram instead trims summed node amounts and requests 40 histogram bins; this is a different grain. "
          "The ALPHA density curve approximates Gaussian smoothing of a 512-bin histogram, not an exact raw-record KDE. Its bandwidth uses the smaller positive SD or IQR/1.349, "
          "with a fallback scale max(cutoff/100,1). The Gaussian kernel is truncated at min(ceil(4σ_bins),200) bins and normalized; boundary mass can be lost. "
          "Individual marks plot one positive trimmed notice amount horizontally; vertical positions are deterministic jitter with no substantive meaning. They are not verified individual units.",
          [r"\bar O=\frac1n\sum_iO_i,\quad h_q=(n-1)q,\quad Q_q=(1-\gamma)x_{(j)}+\gamma x_{(j+1)},\ j=\lfloor h_q\rfloor,\ \gamma=h_q-j",
           r"h=\max(0.9s_*n^{-1/5},\ cutoff/512),\quad s_*=\min_{positive}\left(s,\frac{Q_{.75}-Q_{.25}}{1.349}\right)",
           r"K_j\propto e^{-j^2/(2\sigma_{bins}^2)},\quad\sum_jK_j=1,\quad\hat f_b=\frac{(counts*K)_b}{n\Delta x}"],
          "policy_metrics.py: rent_box_summary; app.py: Rent Owed Distribution"),
    topic("concentration", "Address concentration and Pareto curve", "Descriptive concentration", "BETA",
          "Rank addresses by notice count descending, with address text breaking ties. The top 1%, 5%, and 10% use ceil(A×share), with at least one address. "
          "The displayed long curve keeps its first 500 points and evenly samples the tail up to 1,800 points; totals and summary shares use the full ranked data. "
          "This is not a Gini coefficient, owner concentration, or a size-adjusted filing rate.",
          [r"L(k)=100\frac{\sum_{j=1}^{k}N_{(j)}}{\sum_{j=1}^{A}N_{(j)}},\quad k_q=\max(1,\lceil qA\rceil)"],
          "policy_metrics.py: property_concentration, compressed_pareto"),
    topic("routine", "Routine-address cohorts and repeat intervals", "Retrospective classification", "ALPHA + BETA",
          "An address is routine when it has notices in at least K distinct selected calendar months (default K=6). Cohorts are fixed within the selected retrospective period and change with filters. "
          "The stacked upper boundary is total notice volume; percentage mode is each cohort's share of selected notices, not an eviction rate. Empty periods are reindexed as zero observed selected notices. "
          "Repeat intervals are differences between consecutive distinct received dates at an address. BETA groups intervals into 30-day bins, with a final tail bin. "
          "Current default edges make the last finite bin [720,750) and the tail ≥750, although its label says >730; read exact interval values when using the tail. "
          "Neither retrospective cohort membership nor shorter intervals identify a landlord or prove displacement.",
          [r"Routine_a=\mathbf{1}(|\{m:N_{a,m}>0\}|\ge K),\quad Share_{c,t}=100\frac{N_{c,t}}{\sum_cN_{c,t}}",
           r"\Delta_{a,j}=t_{a,j}-t_{a,j-1}"],
          "feature_engineering.py: routine_filer_summary; policy_metrics.py: repeat_intervals"),
    topic("spatial", "Spatial matching, grid hotspots and persistence", "Descriptive spatial summaries", "ALPHA + BETA",
          "ALPHA assigns points to neighborhood polygons using ‘within’; border points and unmapped records can be excluded. Overlapping polygons can duplicate matches. "
          "Grid cells round latitude and longitude to 2 or 3 decimals, so their physical size varies by latitude; they are not equal-area 1 km/100 m squares. "
          "ALPHA compares each observed grid cell's notice total with the mean and sample SD across observed cells. BETA persistence divides active months in a cell by distinct months represented in selected mapped data, "
          "not necessarily all calendar months in the sidebar. Zero SD gives an ALPHA grid score of zero. These are not Moran's I, Getis–Ord Gi*, or tests of spatial significance.",
          [r"g_i=(round(lat_i,p),round(lon_i,p)),\quad z_g=\frac{N_g-\bar N}{s_N}",
           r"Persistence_g=100\frac{|\{m:N_{g,m}>0\}|}{|\{m:m\text{ occurs in selected mapped records}\}|}"],
          "feature_engineering.py: grid_hotspots; policy_metrics.py: hotspot_persistence; app.py: spatial joins"),
    topic("correlation", "Pearson correlation and optional normalized rates", "Descriptive association", "ALPHA",
          "Each correlation observation is a neighborhood with matched filtered notice activity. Complete finite pairs are used; at least three pairs and variation in both columns are required. "
          "r describes linear association, not causal effects, individual characteristics or a legal standard. No p-values, confounder adjustment, spatial correction or multiple-testing correction is implemented. "
          "Counts depend on area population and rental stock; node/address totals overlap mechanically with notice totals. An optional denominator file enables notices per 1,000 renter households; "
          "its geography and vintage must match, keys must be unique, and denominators must be positive. Missing or zero denominators do not define a rate.",
          [r"r=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i(x_i-\bar x)^2\sum_i(y_i-\bar y)^2}}",
           r"Rate_j=1000\frac{N_j}{RenterHouseholds_j}"],
          "feature_engineering.py: correlation_table; app.py: Compare areas"),
    topic("rent_comparison", "Zillow asking-rent overlay and monthly comparison", "Source estimates + arithmetic means", "ALPHA",
          "Zillow ZORI is a supplied smoothed market asking-rent index, not a model fitted by this app and not actual contract rent. "
          "For each month, the comparison uses an unweighted mean across available ZORI values for filtered eligible ZIPs. The ZIP count can change when data is missing. "
          "Rent owed is the mean positive reported notice amount in the same month over the matched address/ZIP population. Missing means stay missing; observation counts may be zero. "
          "Dollar units match, but asking rent is monthly and arrears may cover several months or repeated notices. Their ratio is not verified months of unpaid contract rent. "
          "ZIP data join to Census ZCTAs by five-digit code, not exact USPS delivery boundaries.",
          [r"\bar Z_t=\frac{\sum_{z\in available_t}ZORI_{z,t}}{|available_t|},\quad\bar O_t=\frac{\sum_{i\in matched_t,O_i>0}O_i}{n_{positive,t}}"],
          "app.py: load_zori_zip, Rent Comparison; data/zillow_zori_la_zip_monthly.csv"),
    topic("anomalies", "Trend-anomaly baseline", "Descriptive screening rule", "ALPHA",
          "The baseline excludes the current month and requires W complete preceding calendar months. Its SD uses W−1. An alert appears at |z|≥2; "
          "this is a chosen screening threshold, not a calibrated significance test. Missing calendar counts stay missing; zero baseline or SD makes the respective division undefined. "
          "An incomplete month, source revision or changed map filter can generate an apparent anomaly. Policy lines annotate dates only; they do not estimate policy effects.",
          [r"B_t=\frac1W\sum_{j=1}^WN_{t-j},\quad s_t^2=\frac{\sum_{j=1}^W(N_{t-j}-B_t)^2}{W-1}",
           r"z_t=\frac{N_t-B_t}{s_t},\quad\Delta_t(\%)=100\left(\frac{N_t}{B_t}-1\right),\quad Alert_t=\mathbf1(|z_t|\ge2)"],
          "feature_engineering.py: anomaly_table; app.py: Alerts & policy"),
    topic("ols", "Monthly linear forecast, MAE and 95% bands", "Forecast model", "ALPHA",
          "OLS regresses filtered mapped monthly notice counts on month index, with at least four months and no missing calendar counts. "
          "MAE uses an earlier fit and the final min(3,max(1,floor(n/4))) months as holdout. The displayed model is refitted to all available months. "
          "Mean confidence bounds estimate the linear trend's expected count; wider prediction bounds include future count variation. Student-t uses n−2 degrees of freedom. "
          "Point forecasts and bounds are clipped at zero. Intervals are pointwise, not simultaneous. Adequate trend, independent normal constant-variance errors are assumed; "
          "seasonality, serial correlation, partial months and reporting delays can invalidate nominal 95% coverage. Out-of-time coverage has not been established.",
          [r"(\hat\alpha,\hat\beta)=\arg\min_{\alpha,\beta}\sum_t(N_t-\alpha-\beta t)^2,\quad\hat N_x=\hat\alpha+\hat\beta x",
           r"MAE=\frac1h\sum_{t\in holdout}|N_t-\hat N_t^{earlier}|,\quad s^2=\frac{SSE}{n-2},\quad\ell_x=\frac1n+\frac{(x-\bar t)^2}{\sum_t(t-\bar t)^2}",
           r"CI_{mean}=\hat N_x\pm t_{.975,n-2}\sqrt{s^2\ell_x},\quad PI_{count}=\hat N_x\pm t_{.975,n-2}\sqrt{s^2(1+\ell_x)}"],
          "feature_engineering.py: forecast_monthly; app.py: Forecast"),
    topic("attention", "Relative building attention score", "Weighted descriptive score", "BETA",
          "Each component is average-tie percentile rank among currently selected non-unknown addresses, multiplied by 100. "
          "N=notice rows, D=distinct received dates, R=notices in the last 180 days ending at the latest selected record, O=sum of positive reported arrears. "
          "Scores are rounded to one decimal. Tiers: Baseline <40; Monitor 40–<60; Elevated 60–<80; High priority ≥80. "
          "Weights and tiers are project choices, not learned probabilities or validated thresholds. Filters change ranks; a single-address population receives rank 100. "
          "Property size is not controlled. Nonpayment share is 100 times exclusively categorized nonpayment notices divided by all address notices.",
          [r"P(x_a)=100\frac{average\_rank(x_a)}{A},\quad Score_a=.30P(N_a)+.25P(D_a)+.25P(R_a)+.20P(O_a)"],
          "policy_metrics.py: building_risk_portfolio"),
    topic("features", "All 15 recurrence-model features", "Feature engineering", "BETA",
          "Eligibility requires a notice at a known address in the preceding 365 days. Count windows include cutoff day: last 30 ages 0–29, last 90 ages 0–89, prior 90 ages 90–179, annual ages 0–364. "
          "Let N_w=count in w days, D_w=distinct dates, M=distinct calendar months, d=days since latest notice, NP=exclusively nonpayment notices, RP=notices with positive reported arrears, "
          "O=sum of positive arrears. Active months divide by 13 because 365 days can touch 13 calendar months. Missing dollars contribute zero to positive-dollar sums but not to reporting share. "
          "Columns are standardized with training mean and population SD (ddof=0); zero/missing SD is replaced by 1. "
          "Protected demographics, ZORI, unit counts and verified ownership are not features. Event dates precede cutoff, but current extracts may contain later backfills; publication-time availability is unverified.",
          [r"f_{1:5}=\log(1+N_{30}),\log(1+N_{90}),\log(1+N_{365}),\log(1+D_{90}),\log(1+D_{365})",
           r"f_6=r=1-d/365,\quad f_7=sign(N_{90}-N_{prior90})\log(1+|N_{90}-N_{prior90}|)",
           r"f_8=NP/N_{365},\ f_9=RP/N_{365},\ f_{10}=\log(1+O),\ f_{11}=M/13",
           r"f_{12}=D_{365}/N_{365},\ f_{13}=N_{90}/N_{365},\ f_{14}=r^2,\ f_{15}=r\log(1+N_{90}),\quad z_j=(f_j-\mu_j)/s_j"],
          "policy_metrics.py: FORECAST_FEATURES, _forecast_snapshot, predictive_building_forecast"),
    topic("logistic", "90-day ridge logistic regression and baseline", "Forecast model", "BETA",
          "Target: any new LAHD notice at an eligible address over cutoff+1 through cutoff+90. This is a binary recurrence target, not expected notice count or tenant displacement. "
          "Enhanced regression uses all 15 standardized features; baseline uses only log(1+90-day notices) and recency, with λ=2. "
          "The intercept is unpenalized. The numerical fit uses Newton updates, at most 60 iterations, stopping at maximum coefficient step <10⁻⁶. "
          "Logits are clipped to ±35; fitting weights have a 10⁻⁶ floor. A pseudoinverse is used if the information matrix is singular. "
          "Coefficient contributions z_jβ_j sum with the intercept to log-odds; they are not causal effects. Probability tiers are Lower <15%, Watch 15–<30%, Elevated 30–<50%, High ≥50%.",
          [r"Y_{a,t}=\mathbf1\left(\sum_{u=t+1}^{t+90}N_{a,u}>0\right),\quad p=\sigma(\eta)=\frac1{1+e^{-\eta}},\quad\eta=\beta_0+z^T\beta",
           r"\hat\beta=\arg\min_\beta\left[-\sum_i\{y_i\log p_i+(1-y_i)\log(1-p_i)\}+\frac\lambda2\sum_{j>0}\beta_j^2\right]",
           r"\beta_{new}=\beta+(X^TWX+\lambda D)^{-1}\{X^T(y-p)-\lambda D\beta\},\quad Contribution_j=z_j\beta_j"],
          "policy_metrics.py: _sigmoid, _fit_ridge_logistic, predictive_building_forecast"),
    topic("validation", "Chronological selection, Brier score, AP and calibration", "Forecast evaluation", "BETA",
          "The XGBoost comparison needs five nonempty mature snapshots; origins are 90 days apart with non-overlapping targets. Up to three earlier rolling validation origins each reserve the preceding mature snapshot for calibration and train on older snapshots. Equal-weight mean validation scores select models; the final period evaluates the frozen choice. Four-snapshot histories use legacy ridge. "
          "Select ridge λ from {0.5,10,100}; it must improve mean validation Brier without reducing AP versus baseline. XGBoost must lower mean Brier without reducing mean AP or precision among the highest-risk 10% versus eligible ridge. Top-decile boundary ties use fractional inclusion. "
          "Test models train on older snapshots; XGBoost calibration uses the selection period, whose outcomes finish by the test origin. Live XGBoost reserves the latest mature snapshot for calibration; live ridge uses all mature snapshots. Test metrics evaluate the procedure, not the latest refit directly. "
          "Brier is average squared probability error (lower better), not percent accuracy. AP integrates precision over changes in recall at distinct descending score thresholds; tied scores share a threshold. "
          "Calibration uses up to 10 equal-frequency score bins (duplicate boundaries dropped), comparing mean prediction with observed event share. Live probabilities are not remapped using test outcomes. "
          "The model rebuilds on source modification, version or selected cutoff changes. The latest evaluation was inspected during previous development: it is evaluation-only in this procedure, not a fresh prospective test. Independent future periods and fairness checks remain needed.",
          [r"Brier=\frac1n\sum_i(p_i-y_i)^2,\quad AP=\sum_k(Recall_k-Recall_{k-1})Precision_k",
           r"Precision=TP/(TP+FP),\quad Recall=TP/\sum_i y_i,\quad\bar p_b=\frac1{n_b}\sum_{i\in b}p_i,\quad\hat q_b=\frac1{n_b}\sum_{i\in b}y_i"],
          "policy_metrics.py: _average_precision, _model_metrics, _calibration_table; xgboost_forecast.py: predictive_forecast, _metrics"),
    topic("xgboost", "Calibrated XGBoost and address-bootstrap uncertainty", "Forecast model", "BETA",
          "Six XGBoost configurations use depth 2–4, L2 10/30, 160 rounds at .05 or 320 at .03, minimum child weight 10/30 and optional 365-day half-life training weights. L1=.1, histogram max_bin=64, two CPU threads, seed=2026. No latest-period early stopping or calibration is used. "
          "Each configuration tests monotone sigmoid calibration on a separate earlier mature period: nonnegative slope A and intercept B minimize weighted binary log loss or squared probability error, plus .001A². Single-class calibration uses a smoothed constant event rate. "
          "Thirty bootstrap refits jointly resample whole address clusters across base-training and calibration; repeated snapshots for one sampled address share its multiplicity. The booster and calibration are both refitted. The standard deviation of bootstrap calibrated logits supplies an approximate normal 95% margin around the point-estimate logit; at least 20 refits are needed. Feature definitions, selected family and hyperparameters are fixed. This is not a percentile interval or a calibrated coverage guarantee. Cross-address/time dependence, selection uncertainty, reporting error and policy changes are excluded. "
          "TreeSHAP contributions sum to the booster margin; multiplying feature contributions by calibration slope explains the calibrated logit. Global gain importance measures split improvement, not causation. Rebuilding uses source modification, forecast origin and model version; the app caches the result.",
          [r"F(x)=F_0+\sum_{m=1}^{M}\eta f_m(x),\quad M\in\{160,320\},\quad p_{raw}=\sigma(F(x)),\quad p_{cal}=\sigma(AF(x)+B),\quad A\geq0",
           r"L_{Brier}(A,B)=\frac{\sum_i w_i[\sigma(AF_i+B)-y_i]^2}{\sum_i w_i}+.001A^2,\quad w_{time}=2^{-age/365}",
           r"LogLoss=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]",
           r"s_{boot,a}=SD(\eta_a^{*(1)},\ldots,\eta_a^{*(B)}),\quad CI_a=\sigma(\hat\eta_a\pm1.959964s_{boot,a})",
           r"F(x)=\phi_0+\sum_j\phi_j,\quad\eta_{cal}=A\phi_0+B+\sum_jA\phi_j"],
          "xgboost_forecast.py: _fit_booster, _fit_platt, _bootstrap_intervals, predictive_forecast; forecast_ui.py"),
    topic("intervals", "95% confidence intervals: group rates and address probabilities", "Approximate uncertainty", "BETA",
          "Wilson bounds estimate observed recurrence in test calibration groups, with z=1.959964. Ridge address-probability bounds use a sandwich covariance clustered by address across repeated snapshots; XGBoost's separate bootstrap procedure is documented in its model section. "
          "X includes the intercept; D leaves it unpenalized; G=training addresses; n=rows; k=coefficients including intercept. H's inverse is computed using a pseudoinverse. "
          "Bounds are unavailable with G<2 or n≤k. All are pointwise. Address intervals treat transformations, selected features and λ as fixed; "
          "regularization bias, model selection, cross-address dependence, source error and policy changes can affect coverage. "
          "A narrow interval is not evidence of a guaranteed future event. Independent 95% coverage is unverified. Individual binary outcomes do not lie inside an interval for a probability.",
          [r"CI_{Wilson}=\frac{\hat q+z^2/(2n)\pm z\sqrt{\hat q(1-\hat q)/n+z^2/(4n^2)}}{1+z^2/n}",
           r"H=X^TWX+\lambda D,\quad U_g=\sum_{i\in g}x_i(y_i-p_i),\quad V=cH^+\left(\sum_gU_gU_g^T\right)H^+",
           r"c=\frac{G}{G-1}\frac{n-1}{n-k},\quad CI_a=\sigma\left(\eta_a\pm t_{.975,G-1}\sqrt{x_a^TVx_a}\right)"],
          "policy_metrics.py: _calibration_table, _cluster_probability_intervals; forecast_ui.py"),
    topic("map_sources", "Map encodings, demographic fields, CPI and quality percentages", "Visual encodings + supplied estimates", "ALPHA + BETA",
          "Mapbox clustering groups screen-near points (cluster radius 40 pixels); point_count counts address-date features, not notice rows. "
          "Heatmap weights linearly map a node's notice count 0→0 and 20→1; intensity and radius vary by zoom. This is visual density, not a fitted risk surface. "
          "Identical rounded coordinates receive slight deterministic display jitter; polygon interactions use displayed points, so border assignments may differ from unjittered analyses. "
          "3D height comes from Mapbox render_height/height and base fields, not forecast risk. ZORI colors interpolate between nearest-index sorted 10th/50th/90th percentiles; "
          "other overlays use min/midrange/max. Exact hover values are retained. This visual percentile differs from pandas linear quantiles. "
          "Neighborhood Census percentages and medians are supplied, not estimated by this app. Other_share is the residual outside four named groups; exact Census variable mapping/vintage is unverified. "
          "The hidden legacy median_income_cpi_adj field divides supplied income by latest national rent CPI: it is not a valid inflation-adjusted dollar-income estimate. "
          "FRED CUSR0000SEHA is a seasonally adjusted rent index (1982–1984=100), not dollar rent. Quality shares divide flagged rows by selected source rows. "
          "Mapped/source coverage is a coordinate-completeness fraction, not proof of geocode correctness.",
          [r"Q^{visual}_q=x_{(round((n-1)q))},\quad LegacyIncomeIndex=Income/CPI_{latest}",
           r"QualityShare=100\frac{N_{condition}}{N_{selected}},\quad MappedShare=100\frac{N_{coordinates}}{N_{source}}"],
          "app.py: Mapbox layers, neighborhood overlays, Data admin; fred_data.py; app_beta.py: Data quality"),
]


def methods_markdown():
    lines = ["# EVICTOR BOOK — Mathematical methods", "Code-grounded reference for ALPHA and BETA. Notices are not completed evictions."]
    for item in METHODS:
        lines.extend([f"\n## {item['title']}", f"{item['kind']} · {item['where']}", item['text']])
        lines.extend(f"\n$$\n{eq}\n$$" for eq in item["equations"])
        lines.append(f"\nImplementation: {item['sources']}")
    return "\n\n".join(lines)


def render_mathematical_methods():
    st.header("Mathematical methods")
    st.caption("Shared reference for every analytical calculation used in ALPHA and BETA. Descriptive scores, forecasts, source estimates and visual encodings are identified separately.")
    st.info("Three forecast model families are implemented: monthly OLS, address-level ridge logistic regression, and calibrated XGBoost. XGBoost is a challenger selected only by earlier evaluation—not assumed superior. No causal policy-effect, survival, neural-network, Moran's I or owner-level eviction model is implemented.")
    st.download_button("Download all methods and formulas", methods_markdown(), "evictor_book_mathematical_methods.md", "text/markdown", key="methods_download")
    selected = st.selectbox("Find a method", ["All methods"] + [m["title"] for m in METHODS], key="method_topic")
    if selected == "All methods":
        st.dataframe(pd.DataFrame([{"Method":m["title"], "Type":m["kind"], "Where":m["where"]} for m in METHODS]), hide_index=True, width="stretch")
    shown = [m for m in METHODS if selected == "All methods" or m["title"] == selected]
    for index, item in enumerate(shown):
        with st.expander(item["title"], expanded=selected != "All methods" or index == 0):
            st.caption(f"{item['kind']} · {item['where']}")
            st.markdown(item["text"])
            for equation in item["equations"]:
                st.latex(equation)
            st.caption(f"Implementation: {item['sources']}")
