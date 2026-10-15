"""Chronological, calibrated XGBoost challenger for 90-day notice recurrence.

The target is a reported LAHD notice, never a completed eviction. Model choice
is frozen before the final test. Publication-time backfill is not observable.
"""
import numpy as np
import pandas as pd
import ctypes
import sys
from pathlib import Path
from functools import lru_cache
from scipy.optimize import minimize
import policy_metrics as pm

BOOST_ROUNDS = 160
BOOTSTRAP_REPLICATES = 30
BASE_INDEXES = [pm.FORECAST_FEATURES.index(f) for f in ["log_notices_90", "recency"]]
XGB_CONFIGS = [dict(depth=2, penalty=10.), dict(depth=3, penalty=10.), dict(depth=3, penalty=30.),
               dict(depth=3, penalty=30., rounds=320, eta=.03),
               dict(depth=4, penalty=30., child=30),
               dict(depth=3, penalty=30., half_life=365),
               dict(depth=4, penalty=30., child=30, enriched=True)]


@lru_cache(maxsize=1)
def _xgboost():
    # Optional project-local macOS runtime; Linux wheels use their own runtime.
    runtime = Path(sys.prefix) / "lib" / "libomp.dylib"
    if sys.platform == "darwin" and runtime.exists():
        ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    import xgboost
    return xgboost


def _matrix(frame, features=None):
    return frame[features or pm.FORECAST_FEATURES].to_numpy(dtype=np.float32)


def _logit(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_platt(scores, targets, weights=None, objective="logloss"):
    """Positive-slope sigmoid calibration; fit only on earlier mature outcomes."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    rate = (np.dot(w, y) + .5) / (w.sum() + 1)
    if np.unique(y[w > 0]).size < 2:
        return np.array([0., _logit(rate)])
    def loss(theta):
        z = theta[0] * scores + theta[1]
        p = pm._sigmoid(z)
        if objective == "brier":
            value = np.sum(w * (p-y)**2) / w.sum() + .001 * theta[0] ** 2
            error = 2 * w * (p-y) * p * (1-p) / w.sum()
        else:
            value = np.sum(w * (np.logaddexp(0, z) - y * z)) / w.sum() + .001 * theta[0] ** 2
            error = w * (p-y) / w.sum()
        gradient = np.array([np.dot(error, scores) + .002 * theta[0], error.sum()])
        return value, gradient
    fit = minimize(loss, [1., 0.], jac=True, method="L-BFGS-B", bounds=[(0., None), (None, None)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError("XGBoost probability calibration did not converge")
    return fit.x


def _fit_booster(training, config, weights=None):
    xgb = _xgboost()
    y = training.target_any_notice_90.to_numpy(dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float).copy()
    if config.get("half_life"):
        age = (training.cutoff.max() - training.cutoff).dt.days.to_numpy()
        w *= np.exp2(-age / config["half_life"])
    keep = w > 0
    if not keep.any():
        raise ValueError("Empty bootstrap training sample")
    features = pm.FORECAST_FEATURES + (pm.EXTRA_FORECAST_FEATURES if config.get("enriched") else [])
    matrix = xgb.DMatrix(_matrix(training,features)[keep], label=y[keep], weight=w[keep],
                         feature_names=features, nthread=2)
    rate = float(np.clip(np.average(y[keep], weights=w[keep]), 1e-4, 1-1e-4))
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  max_bin=64, max_depth=config["depth"], eta=config.get("eta", .05), min_child_weight=config.get("child", 10),
                  reg_lambda=config["penalty"], reg_alpha=.1, subsample=1., colsample_bytree=1.,
                  nthread=2, seed=2026, base_score=rate, verbosity=0)
    return xgb.train(params, matrix, num_boost_round=config.get("rounds", BOOST_ROUNDS))


def _booster_predict(booster, frame, **kwargs):
    xgb = _xgboost()
    return booster.predict(xgb.DMatrix(_matrix(frame,booster.feature_names), feature_names=booster.feature_names, nthread=2), **kwargs)


def _fit_model(training, calibration, config, booster=None):
    if config["family"] == "xgboost":
        booster = _fit_booster(training, config) if booster is None else booster
        slope, intercept = _fit_platt(_booster_predict(booster, calibration, output_margin=True),
                                      calibration.target_any_notice_90, objective=config.get("calibration", "logloss"))
        return dict(config=config, booster=booster, slope=slope, intercept=intercept)
    indexes = BASE_INDEXES if config["family"] == "baseline" else list(range(len(pm.FORECAST_FEATURES)))
    x = _matrix(training).astype(float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.
    beta = pm._fit_ridge_logistic(((x - mean) / scale)[:, indexes],
                                 training.target_any_notice_90.to_numpy(dtype=int), config["penalty"])
    return dict(config=config, beta=beta, indexes=indexes, mean=mean, scale=scale)


def _predict(model, frame):
    if model["config"]["family"] == "xgboost":
        return pm._sigmoid(model["slope"] * _booster_predict(model["booster"], frame, output_margin=True)
                           + model["intercept"])
    x = ((_matrix(frame) - model["mean"]) / model["scale"])[:, model["indexes"]]
    return pm._sigmoid(np.column_stack([np.ones(len(x)), x]) @ model["beta"])


def _metrics(target, probability):
    metrics = pm._model_metrics(np.asarray(target, dtype=int), probability)
    p = np.clip(probability, 1e-8, 1-1e-8)
    metrics["log_loss"] = float(-np.mean(target * np.log(p) + (1-target) * np.log(1-p)))
    # Fractional inclusion at the boundary makes top-decile precision tie-safe.
    budget = max(1., len(p) * .10)
    threshold = np.sort(p)[::-1][int(np.ceil(budget))-1]
    above, tied = p > threshold, p == threshold
    fraction = (budget - above.sum()) / tied.sum()
    metrics["precision_top_10"] = float((np.asarray(target)[above].sum() + fraction*np.asarray(target)[tied].sum()) / budget)
    return metrics


def _bootstrap_intervals(training, calibration, live, model, replicates=BOOTSTRAP_REPLICATES):
    """Approximate logit-Wald CI using whole-address bootstrap refits.

    Resample address clusters jointly across base-training and calibration.
    Feature definitions, model choice and hyperparameters remain fixed.
    """
    addresses = pd.concat([training.full_address, calibration.full_address], ignore_index=True)
    codes, unique = pd.factorize(addresses)
    count = len(unique)
    if count < 2:
        return np.full(len(live), np.nan), np.full(len(live), np.nan), count, 0
    rng = np.random.default_rng(2026)
    draws = []
    for _ in range(replicates):
        multiplicity = np.bincount(rng.integers(count, size=count), minlength=count)
        train_weight = multiplicity[codes[:len(training)]]
        calibration_weight = multiplicity[codes[len(training):]]
        if not train_weight.any() or not calibration_weight.any():
            continue
        booster = _fit_booster(training, model["config"], train_weight)
        slope, intercept = _fit_platt(_booster_predict(booster, calibration, output_margin=True),
                                      calibration.target_any_notice_90, calibration_weight,
                                      objective=model["config"].get("calibration", "logloss"))
        draws.append(slope * _booster_predict(booster, live, output_margin=True) + intercept)
    if len(draws) < 20:
        return np.full(len(live), np.nan), np.full(len(live), np.nan), count, len(draws)
    margin = 1.959963984540054 * np.std(np.asarray(draws), axis=0, ddof=1)
    center = _logit(_predict(model, live))
    return pm._sigmoid(center-margin), pm._sigmoid(center+margin), count, len(draws)


def predictive_forecast(df, as_of_date=None):
    work = df[df.full_address.ne("Unknown address")].sort_values("Date_Filed").copy()
    if work.empty:
        return pm._ridge_predictive_building_forecast(df, as_of_date)
    end = work.Date_Filed.max().normalize()
    as_of = min(pd.Timestamp(as_of_date).normalize(), end) if as_of_date is not None else end
    work = work[work.Date_Filed.le(as_of)]
    if work.empty:
        return pm._ridge_predictive_building_forecast(df, as_of_date)
    first = work.Date_Filed.min().normalize() + pd.Timedelta(days=365)
    last = as_of - pd.Timedelta(days=90)
    cutoffs = list(reversed(pd.date_range(last, first, freq="-90D"))) if last > first else []
    snapshots = [pm._forecast_snapshot(work, cutoff, True) for cutoff in cutoffs]
    snapshots = [s for s in snapshots if not s.empty]
    if len(snapshots) < 5:
        result = pm._ridge_predictive_building_forecast(df, as_of_date)
        if result.get("metadata"):
            result["metadata"]["xgboost_status"] = "Not evaluated: five mature snapshots required for separate calibration, selection and testing."
        return result
    try:
        _xgboost()
    except (ImportError, OSError, ValueError):
        result = pm._ridge_predictive_building_forecast(df, as_of_date)
        result["metadata"]["xgboost_status"] = "XGBoost unavailable; using ridge fallback. Install project requirements (and libomp on macOS), then restart."
        return result

    selection, holdout = snapshots[-2:]
    configs = [dict(family="baseline", penalty=2.)]
    configs += [dict(family="ridge", penalty=p) for p in (.5, 10., 100.)]
    configs += [dict(family="xgboost", calibration=cal, **c) for c in XGB_CONFIGS for cal in ("logloss", "brier")]
    trials = []
    fold_indexes = list(range(max(2, len(snapshots)-4), len(snapshots)-1))
    fold_rows = []
    capacity_rows = []
    booster_cache = {}
    for config in configs:
        scores = []
        for i in fold_indexes:
            fold_training = pd.concat(snapshots[:i-1], ignore_index=True)
            cache_key = (i, tuple(sorted((k,v) for k,v in config.items() if k != "calibration")))
            model = _fit_model(fold_training, snapshots[i-1], config, booster_cache.get(cache_key))
            if config["family"] == "xgboost":
                booster_cache[cache_key] = model["booster"]
            probability = _predict(model, snapshots[i])
            score = _metrics(snapshots[i].target_any_notice_90.to_numpy(), probability)
            from outreach_metrics import capacity_table
            capacity_rows.extend(capacity_table(snapshots[i].target_any_notice_90.to_numpy(), probability,
                snapshots[i].full_address).assign(config=str(config), origin=str(snapshots[i].cutoff.iloc[0].date())).to_dict("records"))
            scores.append(score)
            fold_rows.append(dict(config=str(config), origin=str(snapshots[i].cutoff.iloc[0].date()), **score))
        metrics = {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}
        trials.append(dict(config=config, metrics=metrics))
    booster_cache.clear()
    baseline_selection = trials[0]["metrics"]
    eligible = [t for t in trials[1:] if t["config"]["family"] == "ridge" and t["metrics"]["brier_score"] < baseline_selection["brier_score"]
                and t["metrics"]["average_precision"] >= baseline_selection["average_precision"]]
    selected = min(eligible, key=lambda t: t["metrics"]["brier_score"]) if eligible else trials[0]
    xgb_eligible = [t for t in trials if t["config"]["family"] == "xgboost"
                    and t["metrics"]["brier_score"] < selected["metrics"]["brier_score"]
                    and t["metrics"]["average_precision"] >= selected["metrics"]["average_precision"]
                    and t["metrics"]["precision_top_10"] >= selected["metrics"]["precision_top_10"]]
    old_eligible = [t for t in xgb_eligible if not t["config"].get("enriched")]
    if old_eligible:
        anchor = min(old_eligible,key=lambda t:t["metrics"]["brier_score"])["metrics"]
        xgb_eligible = [t for t in xgb_eligible if not t["config"].get("enriched") or
            (t["metrics"]["brier_score"] < anchor["brier_score"] and t["metrics"]["average_precision"] >= anchor["average_precision"] and t["metrics"]["precision_top_10"] >= anchor["precision_top_10"])]
    if xgb_eligible:
        selected = min(xgb_eligible, key=lambda t: t["metrics"]["brier_score"])
    from outreach_metrics import choose_capacity
    selected_capacity = choose_capacity(pd.DataFrame(capacity_rows), selected["config"])
    best_ridge = min([t for t in trials if t["config"]["family"] == "ridge"], key=lambda t: t["metrics"]["brier_score"])
    best_xgb = min([t for t in trials if t["config"]["family"] == "xgboost"], key=lambda t: t["metrics"]["brier_score"])
    # Evaluate the exact promoted configuration, not an ineligible same-family trial.
    if selected["config"]["family"] == "xgboost":
        best_xgb = selected
    elif selected["config"]["family"] == "ridge":
        best_ridge = selected
    names = {"baseline": "Recent-history baseline", "ridge": "Enhanced ridge logistic", "xgboost": "Calibrated XGBoost"}
    # All outcomes used here finish by the final test's forecast origin.
    test_training = pd.concat(snapshots[:-2], ignore_index=True)
    test_rows = []
    test_models = {}
    test_capacities = []
    for trial in [trials[0], best_ridge, best_xgb]:
        family = trial["config"]["family"]
        fitted = _fit_model(test_training, selection, trial["config"])
        probability = _predict(fitted, holdout)
        metrics = _metrics(holdout.target_any_notice_90.to_numpy(), probability)
        test_rows.append(dict(Model=names[family], Selected=family == selected["config"]["family"], **metrics))
        test_models[family] = (metrics, probability)
        test_capacities.append(capacity_table(holdout.target_any_notice_90.to_numpy(), probability, holdout.full_address).assign(Model=names[family]))
    incumbent = _fit_model(test_training, selection, dict(family="xgboost", depth=4, penalty=30., child=30, calibration="brier"))
    incumbent_metrics = _metrics(holdout.target_any_notice_90.to_numpy(), _predict(incumbent, holdout))
    test_rows.append(dict(Model="Previous XGBoost (15 features)", Selected=False, **incumbent_metrics))
    test_capacities.append(capacity_table(holdout.target_any_notice_90.to_numpy(), _predict(incumbent,holdout), holdout.full_address).assign(Model="Previous XGBoost (15 features)"))
    family = selected["config"]["family"]
    served_metrics, served_holdout = test_models[family]
    all_training = pd.concat(snapshots, ignore_index=True)
    live_training = pd.concat(snapshots[:-1], ignore_index=True) if family == "xgboost" else all_training
    live_model = _fit_model(live_training, snapshots[-1], selected["config"])
    live = pm._forecast_snapshot(work, as_of, False)
    live["forecast_probability"] = _predict(live_model, live)
    live["uncalibrated_probability"] = (_booster_predict(live_model["booster"], live)
                                          if family == "xgboost" else live.forecast_probability)
    bootstrap_count = 0
    if family == "xgboost":
        feature_names = live_model["booster"].feature_names
        low, high, clusters, bootstrap_count = _bootstrap_intervals(live_training, snapshots[-1], live, live_model)
        contribution = live_model["slope"] * _booster_predict(live_model["booster"], live, pred_contribs=True)[:, :-1]
        gains = live_model["booster"].get_score(importance_type="gain")
        importance = pd.DataFrame({"feature_label": [pm.FORECAST_FEATURE_LABELS[f] for f in feature_names],
                                   "gain": [gains.get(f, 0.) for f in feature_names]}).sort_values("gain", ascending=False)
        coefficients = pd.DataFrame(columns=["feature", "feature_label", "standardized_coefficient"])
        interval_method = "Approximate 95% address-cluster bootstrap logit-Wald interval (30 refits; model selection fixed)"
    else:
        feature_names = pm.FORECAST_FEATURES
        indexes = live_model["indexes"]
        standardized = (_matrix(live_training) - live_model["mean"]) / live_model["scale"]
        score = (_matrix(live) - live_model["mean"]) / live_model["scale"]
        low, high, clusters = pm._cluster_probability_intervals(standardized[:, indexes],
            live_training.target_any_notice_90.to_numpy(), live_model["beta"], live_training.full_address,
            score[:, indexes], selected["config"]["penalty"])
        beta = np.zeros(len(pm.FORECAST_FEATURES))
        beta[indexes] = live_model["beta"][1:]
        contribution = score * beta
        coefficients = pd.DataFrame({"feature": pm.FORECAST_FEATURES,
            "feature_label": [pm.FORECAST_FEATURE_LABELS[f] for f in pm.FORECAST_FEATURES],
            "standardized_coefficient": beta}).sort_values("standardized_coefficient", ascending=False)
        importance = pd.DataFrame()
        interval_method = "Approximate address-clustered Wald interval on the logistic scale"
    live["ci_lower_95"], live["ci_upper_95"] = low, high
    live["forecast_tier"] = pd.cut(live.forecast_probability, [-np.inf, .15, .30, .50, np.inf],
                                  labels=["Lower", "Watch", "Elevated", "High"], right=False).astype(str)
    live[[f"contribution_{f}" for f in feature_names]] = contribution
    top = np.argsort(-contribution, axis=1)[:, :3]
    live["top_drivers"] = [", ".join(pm.FORECAST_FEATURE_LABELS[feature_names[j]] for j in indexes if row[j] > 0)
                           or "Model baseline" for row, indexes in zip(contribution, top)]
    metadata = dict(model_version=pm.FORECAST_MODEL_VERSION, forecast_as_of=as_of,
        trained_through=as_of, forecast_horizon_days=90, lookback_days=365,
        training_rows=len(all_training), base_model_training_rows=len(live_training), training_snapshots=len(snapshots),
        holdout_addresses=len(holdout), promotion_gate_passed=family != "baseline", selected_model=names[family],
        selected_family=family, selected_penalty=selected["config"]["penalty"], selected_parameters=selected["config"], feature_names=feature_names, selected_capacity=selected_capacity,
        selection_cutoff=cutoffs[-2], holdout_cutoff=cutoffs[-1], calibration_cutoff=cutoffs[-3],
        test_training_outcomes_through=cutoffs[-2]+pd.Timedelta(days=90), forecast_end=as_of+pd.Timedelta(days=90),
        candidate_metrics=test_models["ridge"][0], baseline_metrics=test_models["baseline"][0],
        xgboost_metrics=test_models["xgboost"][0], incumbent_metrics=incumbent_metrics, served_metrics=served_metrics,
        validation_candidate_metrics=best_ridge["metrics"], validation_baseline_metrics=baseline_selection,
        validation_selected_metrics=selected["metrics"], validation_xgboost_metrics=best_xgb["metrics"],
        confidence_level=.95, interval_method=interval_method, interval_clusters=clusters,
        bootstrap_refits=bootstrap_count, validation_periods=len(fold_indexes),
        xgboost_status="Selected using rolling earlier periods; latest period is evaluation-only. This latest period was also inspected during prior development, so it is not a fresh prospective test.")
    return dict(forecast=live.sort_values(["forecast_probability", "notices_90"], ascending=False).reset_index(drop=True),
                outreach_validation=pd.DataFrame(capacity_rows), outreach_evaluation=pd.concat(test_capacities,ignore_index=True),
                metadata=metadata, calibration=pm._calibration_table(holdout.target_any_notice_90.to_numpy(), served_holdout),
                coefficients=coefficients, feature_importance=importance, model_comparison=pd.DataFrame(test_rows),
                rolling_validation=pd.DataFrame(fold_rows),
                validation_trials=pd.DataFrame([dict(Model=names[t["config"]["family"]], **t["config"], **t["metrics"]) for t in trials]))
