import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import policy_metrics as pm
from feature_engineering import forecast_monthly


def fixture():
    rng = np.random.default_rng(12)
    rows = []
    for address in range(40):
        days = rng.choice(1100, size=15 + address * 2, replace=False)
        for day in days:
            rows.append(dict(full_address=f"{address} TEST ST", Date_Filed=pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(day)),
                             Rent_Owed_num=1000., Cause_Category="Non-payment of rent", Council_District=1, Zip="90001", lat=34., long=-118.))
    return pd.DataFrame(rows).sort_values("Date_Filed")


class ForecastTests(unittest.TestCase):
    def test_average_precision_ties(self):
        self.assertAlmostEqual(pm._average_precision(np.array([1, 0, 1, 0]), np.ones(4)), .5)
        self.assertAlmostEqual(pm._average_precision(np.array([0, 1, 0, 1]), np.ones(4)), .5)

    def test_snapshot_does_not_use_future_features(self):
        data = fixture()
        cutoff = pd.Timestamp("2025-01-01")
        a = pm._forecast_snapshot(data, cutoff, False)
        b = pm._forecast_snapshot(data[data.Date_Filed.le(cutoff)], cutoff, False)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(np.isfinite(a[pm.FORECAST_FEATURES]).all().all())

    def test_chronology_and_score_bounds(self):
        result = pm.predictive_building_forecast(fixture())
        meta = result["metadata"]
        self.assertLessEqual(meta["test_training_outcomes_through"], meta["holdout_cutoff"])
        self.assertEqual(meta["selection_cutoff"] + pd.Timedelta(days=90), meta["holdout_cutoff"])
        self.assertTrue(result["forecast"].forecast_probability.between(0, 1).all())
        forecast = result["forecast"]
        self.assertTrue(np.isfinite(forecast[["ci_lower_95", "ci_upper_95"]]).all().all())
        self.assertTrue(forecast.ci_lower_95.between(0, 1).all())
        self.assertTrue(forecast.ci_upper_95.between(0, 1).all())
        self.assertTrue(forecast.ci_lower_95.le(forecast.forecast_probability).all())
        self.assertTrue(forecast.ci_upper_95.ge(forecast.forecast_probability).all())
        self.assertEqual(meta["confidence_level"], .95)
        self.assertEqual(meta["forecast_end"] - meta["forecast_as_of"], pd.Timedelta(days=90))

    def test_fallback_really_uses_baseline(self):
        # Equal candidate/baseline scores cannot pass the strict improvement gate.
        with patch.object(pm, "_model_metrics", return_value=dict(brier_score=.2, average_precision=.3, event_rate=.4)):
            result = pm.predictive_building_forecast(fixture())
        self.assertFalse(result["metadata"]["promotion_gate_passed"])
        coefficients = result["coefficients"].set_index("feature").standardized_coefficient
        other = coefficients.drop(["log_notices_90", "recency"])
        self.assertTrue(other.eq(0).all())
        forecast = result["forecast"]
        self.assertTrue(forecast.ci_lower_95.le(forecast.forecast_probability).all())
        self.assertTrue(forecast.ci_upper_95.ge(forecast.forecast_probability).all())

    def test_cluster_intervals_match_manual_covariance(self):
        from scipy.stats import t
        x = np.array([[-1.], [0.], [1.], [2.], [-.5], [.5]])
        y = np.array([0., 1., 1., 0., 0., 1.])
        beta = np.array([-.2, .4])
        groups = np.array(["A", "A", "B", "B", "C", "C"])
        score_x = np.array([[-.25], [.75]])
        low, high, count = pm._cluster_probability_intervals(x, y, beta, groups, score_x, 2.)
        design = np.column_stack([np.ones(6), x])
        p = pm._sigmoid(design @ beta)
        h = design.T @ np.diag(p * (1-p)) @ design + np.diag([0., 2.])
        scores = np.array([np.sum(design[groups == g] * (y-p)[groups == g, None], axis=0) for g in ["A", "B", "C"]])
        v = (3/2) * (5/4) * np.linalg.inv(h) @ scores.T @ scores @ np.linalg.inv(h)
        score_design = np.column_stack([np.ones(2), score_x])
        margin = t.ppf(.975, 2) * np.sqrt(np.diag(score_design @ v @ score_design.T))
        np.testing.assert_allclose(low, pm._sigmoid(score_design @ beta - margin))
        np.testing.assert_allclose(high, pm._sigmoid(score_design @ beta + margin))
        self.assertEqual(count, 3)
        order = np.array([5, 2, 0, 3, 1, 4])
        reordered = pm._cluster_probability_intervals(x[order], y[order], beta, groups[order], score_x, 2.)
        np.testing.assert_allclose(reordered[0], low)
        np.testing.assert_allclose(reordered[1], high)

    def test_cluster_intervals_insufficient_clusters(self):
        low, high, count = pm._cluster_probability_intervals(
            np.zeros((4, 1)), np.array([0, 1, 0, 1]), np.zeros(2),
            np.array(["A"] * 4), np.zeros((1, 1)), 2.)
        self.assertEqual(count, 1)
        self.assertTrue(np.isnan(low).all() and np.isnan(high).all())

    def test_monthly_confidence_and_prediction_intervals(self):
        from scipy.stats import t
        counts = np.array([12, 17, 14, 23, 21, 28])
        dates = pd.date_range("2025-01-01", periods=len(counts), freq="MS")
        data = pd.DataFrame({"Date_Filed": np.repeat(dates, counts), "Eviction_Count": 1})
        forecast, diagnostics = forecast_monthly(data, months=3)
        x = np.arange(6.)
        design = np.column_stack([np.ones(6), x])
        beta = np.linalg.solve(design.T @ design, design.T @ counts)
        s2 = np.sum((counts - design @ beta) ** 2) / 4
        future = np.column_stack([np.ones(3), np.arange(6., 9.)])
        expected = future @ beta
        leverage = np.diag(future @ np.linalg.inv(design.T @ design) @ future.T)
        mean_margin = t.ppf(.975, 4) * np.sqrt(s2 * leverage)
        prediction_margin = t.ppf(.975, 4) * np.sqrt(s2 * (1 + leverage))
        np.testing.assert_allclose(forecast.ci_lower_95, np.maximum(0, expected-mean_margin))
        np.testing.assert_allclose(forecast.ci_upper_95, expected+mean_margin)
        np.testing.assert_allclose(forecast.lower, np.maximum(0, expected-prediction_margin))
        np.testing.assert_allclose(forecast.upper, expected+prediction_margin)
        self.assertTrue(forecast.lower.le(forecast.ci_lower_95).all())
        self.assertTrue(forecast.upper.ge(forecast.ci_upper_95).all())
        self.assertEqual(diagnostics["residual_df"], 4)

    def test_insufficient_history(self):
        data = fixture()
        result = pm.predictive_building_forecast(data, "2023-06-01")
        self.assertTrue(result["forecast"].empty)
        self.assertTrue(pm.predictive_building_forecast(data, "2020-01-01")["forecast"].empty)


if __name__ == "__main__":
    unittest.main()
