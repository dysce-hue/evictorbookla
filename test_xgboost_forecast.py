import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import policy_metrics as pm
import xgboost_forecast as xf
from test_forecast import fixture


class XGBoostTests(unittest.TestCase):
    def test_platt_monotone_and_finite(self):
        scores = np.array([-3., -2., -1., 0., 1., 2., 3.])
        a, b = xf._fit_platt(scores, [0, 0, 1, 0, 1, 1, 1])
        self.assertGreaterEqual(a, 0)
        self.assertTrue(np.isfinite([a, b]).all())
        self.assertTrue(np.all(np.diff(pm._sigmoid(a*scores+b)) >= 0))
        a, b = xf._fit_platt(scores, np.ones(7))
        self.assertEqual(a, 0)
        self.assertLess(pm._sigmoid(b), 1)

    def test_forced_xgboost_path_and_intervals(self):
        def metric(error, ap):
            return dict(brier_score=error, average_precision=ap, event_rate=.5)
        metrics = [metric(.20, .7)] + [metric(.15, .8)] * 3 + [metric(.10, .9)] * 3
        metrics += [metric(.20, .7), metric(.15, .8), metric(.10, .9)]
        with patch.object(pm, "_model_metrics", side_effect=metrics):
            result = pm.predictive_building_forecast(fixture())
        self.assertEqual(result["metadata"]["selected_family"], "xgboost")
        self.assertEqual(result["metadata"]["bootstrap_refits"], 30)
        forecast = result["forecast"]
        self.assertTrue(np.isfinite(forecast[["forecast_probability", "ci_lower_95", "ci_upper_95"]]).all().all())
        self.assertTrue(forecast.ci_lower_95.le(forecast.forecast_probability).all())
        self.assertTrue(forecast.ci_upper_95.ge(forecast.forecast_probability).all())
        self.assertEqual(len(result["model_comparison"]), 3)
        self.assertFalse(result["feature_importance"].empty)
        self.assertTrue(result["coefficients"].empty)

    def test_final_outcomes_do_not_select_model(self):
        data = fixture()
        origin = pd.Timestamp("2025-12-01")
        first = pm.predictive_building_forecast(data, origin)
        extra = data.iloc[[0]].copy()
        extra["Date_Filed"] = origin - pd.Timedelta(days=1)
        changed = pd.concat([data, extra], ignore_index=True)
        second = pm.predictive_building_forecast(changed, origin)
        self.assertEqual(first["metadata"]["selected_parameters"], second["metadata"]["selected_parameters"])
        pd.testing.assert_frame_equal(first["validation_trials"], second["validation_trials"])
        self.assertLessEqual(first["metadata"]["test_training_outcomes_through"], first["metadata"]["holdout_cutoff"])


if __name__ == "__main__":
    unittest.main()
