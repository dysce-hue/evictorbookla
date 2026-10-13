"""Regression checks for official-source formats and transparent statistics."""
import io
import unittest

import numpy as np
import pandas as pd

from data_refresh import EXPECTED_SOURCE, normalize_source, parse_source_content, validate, RefreshError
from feature_engineering import anomaly_table, correlation_table, forecast_monthly, routine_filer_summary
from fred_data import fetch_rent_cpi


class RefreshAnalyticsTests(unittest.TestCase):
    def source(self):
        frame = pd.DataFrame([{column: "" for column in EXPECTED_SOURCE} for _ in range(2)])
        frame["Date Received"] = ["01-03-2024", "2024-02-03"]
        frame["Notice Date"] = frame["Date Received"]
        frame["Address"] = ["1 MAIN ST", "2 MAIN ST"]
        frame["City"] = "Los Angeles"
        frame["Zip"] = "90001"
        return frame

    def test_csv_and_workbook_have_same_rows(self):
        source = self.source()
        csv, kind = parse_source_content(source.to_csv(index=False).encode())
        self.assertEqual(kind, "csv")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            source.to_excel(writer, sheet_name="Notices", index=False)
            pd.DataFrame({"Notes": ["cover page"]}).to_excel(writer, sheet_name="Read me", index=False)
        xlsx, kind = parse_source_content(buffer.getvalue())
        self.assertEqual(kind, "xlsx")
        self.assertEqual(len(xlsx), len(csv))
        self.assertEqual(normalize_source(xlsx)["Date Received"].tolist(), ["01-03-2024", "02-03-2024"])

    def test_previous_calendar_months_and_gap(self):
        frame = pd.DataFrame({"Date_Filed": pd.date_range("2024-01-01", periods=4, freq="MS"),
                              "Eviction_Count": [10, 20, 30, 60]})
        result = anomaly_table(frame, window=3)
        self.assertTrue(result["baseline"].iloc[:3].isna().all())
        self.assertEqual(result.iloc[3]["baseline"], 20)
        self.assertEqual(result.iloc[3]["z_score"], 4)
        gap = frame.drop(index=1)
        self.assertTrue(anomaly_table(gap)["baseline"].isna().all())
        self.assertTrue(forecast_monthly(gap)[0].empty)

    def test_correlation_uses_complete_pairs(self):
        result = correlation_table(pd.DataFrame({"evictions_total": [1, 2, 3, 4],
                                                 "income": [2, 4, 6, np.nan], "constant": [1]*4})).set_index("metric")
        self.assertAlmostEqual(result.loc["income", "correlation_with_evictions"], 1)
        self.assertEqual(result.loc["income", "paired_neighborhoods"], 3)
        self.assertTrue(pd.isna(result.loc["constant", "correlation_with_evictions"]))

    def test_invalid_key_goes_directly_to_public_csv(self):
        class Response:
            text = "observation_date,CUSR0000SEHA\n2024-01-01,100\n"
            def raise_for_status(self):
                pass
        class Session:
            def get(self, url, **kwargs):
                if "api.stlouisfed" in url:
                    raise AssertionError("Malformed key sent to API")
                return Response()
        frame, source, notes = fetch_rent_cpi("short", Session())
        self.assertEqual(len(frame), 1)
        self.assertEqual(source, "FRED public CSV")
        self.assertNotIn("short", str(notes))

    def test_historical_month_loss_rejected(self):
        frame = normalize_source(self.source())
        frame["lat"], frame["long"] = 34., -118.
        old = pd.concat([frame, frame], ignore_index=True)
        candidate = old.copy()
        candidate["Date Received"] = "02-03-2024"
        with self.assertRaises(RefreshError):
            validate(candidate, old)

    def test_routine_cohorts_preserve_counts_and_use_distinct_months(self):
        frame = pd.DataFrame({"full_address": ["A", "A", "A", "B"],
                              "Date_Filed": pd.to_datetime(["2024-01-01", "2024-01-02", "2025-02-01", "2025-02-01"]),
                              "Eviction_Count": [3, 4, 5, 8]})
        series, addresses = routine_filer_summary(frame, 2)
        self.assertEqual(series["notices"].sum(), 20)
        self.assertEqual(addresses.set_index("full_address").loc["A", "cohort"], "Routine")
        self.assertEqual(addresses.set_index("full_address").loc["B", "cohort"], "Infrequent")
        self.assertTrue(np.allclose(series.groupby("period")["share_percent"].sum(), 100))


if __name__ == "__main__":
    unittest.main()
