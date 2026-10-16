import unittest
import pandas as pd
from property_panel import normalize_apn,build_panel

class PropertyPanelTests(unittest.TestCase):
    def fixture(self):
        notices = pd.DataFrame({"Apn":["1234-567-890","1234567890","1234567890","2222222222","invalid"],
            "Date Received":["2024-01-01","2024-02-10","2024-07-01","2024-08-31","2024-01-01"]})
        inventory = pd.DataFrame({"APN":["1234567890","3333333333","3333333333"],"RSO_Units":[12,4,8]})
        return notices,inventory

    def test_strict_ids_no_fabricated_confidence(self):
        result = normalize_apn(pd.Series(["1234-567-890","1234567890.0","AB1234567890","0000000000",None]))
        self.assertEqual(result.iloc[0],"1234567890")
        self.assertEqual(result.iloc[1],"1234567890")
        self.assertTrue(result.iloc[2:].isna().all())

    def test_panel_preserves_zeros_and_unknown_stages(self):
        notices,inventory = self.fixture()
        panel,audit = build_panel(notices,inventory,"2026-10-16T12:00:00Z")
        self.assertFalse(panel.duplicated(["apn","origin"]).any())
        self.assertEqual(audit["universe_apns_without_observed_notice"],1)
        zero = panel[panel.apn.eq("3333333333")]
        self.assertTrue(zero.observed_notices_month.eq(0).all())
        self.assertTrue(zero.rso_units_current.isna().all())
        self.assertTrue(zero.unit_count_ambiguous.all())
        self.assertTrue(panel.training_target_notice_next_90d.isna().all())
        self.assertFalse(panel.training_eligible.any())
        self.assertTrue(panel[["ud_filing_next_90d","judgment_next_90d","writ_next_90d","lockout_next_90d"]].isna().all().all())
        self.assertEqual(panel.observed_notices_month.sum(),3)

    def test_features_do_not_read_future_and_labels_mature(self):
        notices,inventory = self.fixture()
        panel,_ = build_panel(notices,inventory,"2026-10-16T12:00:00Z")
        first = panel[panel.apn.eq("1234567890") & panel.origin.eq(pd.Timestamp("2024-01-31"))].iloc[0]
        self.assertEqual(first.prior_notices_90,1)
        self.assertEqual(first.observed_notice_next_90d,1)
        latest = panel[panel.origin.eq(pd.Timestamp("2024-08-31"))]
        self.assertTrue(latest.observed_notice_next_90d.isna().all())
        changed = pd.concat([notices,notices.iloc[[1]]],ignore_index=True)
        newer,_ = build_panel(changed,inventory,"2026-10-16T12:00:00Z")
        a = panel[panel.origin.eq(pd.Timestamp("2024-01-31"))]
        b = newer[newer.origin.eq(pd.Timestamp("2024-01-31"))]
        pd.testing.assert_frame_equal(a[["apn","prior_notices_30","prior_notices_90"]].reset_index(drop=True),b[["apn","prior_notices_30","prior_notices_90"]].reset_index(drop=True))

if __name__ == "__main__":
    unittest.main()
