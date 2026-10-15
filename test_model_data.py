import tempfile
import unittest
from pathlib import Path
import pandas as pd
from data_refresh import EXPECTED_SOURCE, download_source
from model_data import archive_download, capture_inventory, reporting_delay_report, point_in_time_attributes


class ModelDataTests(unittest.TestCase):
    def test_download_hook_archives_exact_payload(self):
        content = pd.DataFrame([self.source("2026-01-01", str(i)+" MAIN STREET LOS ANGELES") for i in range(40)]).to_csv(index=False).encode()
        class Response:
            headers = {}
            def raise_for_status(self):
                pass
        response = Response()
        response.content = content
        class Session:
            def get(self, *args, **kwargs):
                return response
        with tempfile.TemporaryDirectory() as temp:
            frame, meta = download_source(Session(), temp)
            self.assertEqual(len(frame), 40)
            self.assertEqual(meta["bytes"], len(content))
            self.assertEqual(len(capture_inventory(temp)), 1)

    def source(self, received, address="1 MAIN ST"):
        row = {c:"" for c in EXPECTED_SOURCE}
        row.update({"Date Received":received, "Notice Date":received, "Address":address})
        return row

    def test_archive_hash_dedup_and_baseline_censoring(self):
        with tempfile.TemporaryDirectory() as temp:
            initial = pd.DataFrame([self.source("2026-01-01")]).to_csv(index=False).encode()
            archive_download(initial, temp, "test", "2026-01-10T12:00:00Z")
            archive_download(initial, temp, "test", "2026-01-11T12:00:00Z")
            self.assertEqual(len(list(Path(temp).glob("*.payload.gz"))), 1)
            self.assertEqual(len(capture_inventory(temp)), 2)
            self.assertTrue(reporting_delay_report(temp).empty)
            revised = pd.DataFrame([self.source("2026-01-01"), self.source("2026-01-12", "2 MAIN ST")]).to_csv(index=False).encode()
            archive_download(revised, temp, "test", "2026-01-15T12:00:00Z")
            report = reporting_delay_report(temp)
            self.assertEqual(report.observed_delay_days.tolist(), [3])
            self.assertEqual(report.received_date.tolist(), ["2026-01-12"])

    def test_archive_timezone_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                archive_download(b"test", temp, "test", "2026-01-01")
            result = archive_download(pd.DataFrame([self.source("2026-01-01")]).to_csv(index=False).encode(), temp, "test", "2026-01-10T12:00:00Z")
            import gzip
            (Path(temp) / f"{result['sha256']}.payload.gz").write_bytes(gzip.compress(b"changed"))
            with self.assertRaisesRegex(ValueError, "checksum"):
                reporting_delay_report(temp)

    def test_stale_source_not_used_as_reporting_delay_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            archive_download(pd.DataFrame([self.source("2026-01-12")]).to_csv(index=False).encode(),temp,"test","2026-01-15T12:00:00Z")
            archive_download(pd.DataFrame([self.source("2025-01-01","2 MAIN ST")]).to_csv(index=False).encode(),temp,"test","2026-01-16T12:00:00Z")
            report = reporting_delay_report(temp)
            self.assertTrue(report.empty)
            self.assertEqual(report.attrs["excluded_stale_snapshots"],1)

    def test_attributes_obey_both_dates_and_preserve_unmatched(self):
        ids = pd.DataFrame({"building_id":["A", "B", "C", "A"]})
        attributes = pd.DataFrame({"building_id":["A", "A", "B"],
            "effective_at":["2024-01-01", "2025-01-01", "2027-01-01"],
            "available_at":["2024-01-01", "2026-01-01", "2024-01-01"],
            "source_url":["https://example.org/source"]*3, "rental_units":[12, 20, 5]})
        result = point_in_time_attributes(ids, attributes, "2025-06-01")
        self.assertEqual(len(result), 4)
        self.assertEqual(result.rental_units.iloc[0], 12)
        self.assertTrue(result.rental_units.iloc[1:3].isna().all())
        self.assertEqual(result.rental_units.iloc[3], 12)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            point_in_time_attributes(ids, pd.concat([attributes, attributes.iloc[[0]]]), "2025-06-01")
        attributes.loc[0, "rental_units"] = 0
        with self.assertRaisesRegex(ValueError, "positive integers"):
            point_in_time_attributes(ids, attributes, "2025-06-01")


if __name__ == "__main__":
    unittest.main()
