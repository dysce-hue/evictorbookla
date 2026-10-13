import unittest
from unittest.mock import patch
import plotly.graph_objects as go
import pandas as pd
from ui_helpers import embed_html, readable_timeline, readable_neighborhoods, address_timeline, routine_chart


class DisplayTests(unittest.TestCase):
    def test_supported_iframe(self):
        with patch("ui_helpers.st.iframe") as iframe:
            embed_html("<p>test</p>", height=720)
            self.assertEqual(iframe.call_args.kwargs["height"], 720)

    def test_readable_timeline(self):
        figure = readable_timeline(go.Figure())
        self.assertEqual(figure.layout.legend.orientation, "h")
        self.assertEqual(figure.layout.yaxis.title.text, "Notice records")
        self.assertTrue(figure.layout.xaxis.automargin)

    def test_neighborhood_rate_label(self):
        figure = readable_neighborhoods(go.Figure(), rate=True)
        self.assertIn("1,000", figure.layout.xaxis.title.text)
        self.assertEqual(figure.layout.yaxis.categoryorder, "total ascending")

    def test_share_lines_not_stacked_and_years_not_repeated(self):
        data = pd.DataFrame({"period": pd.to_datetime(["2023-01-01", "2023-01-01", "2024-01-01", "2024-01-01"]),
                             "cohort": ["Routine", "Infrequent"] * 2,
                             "notices": [42, 58, 70, 30], "share_percent": [42, 58, 70, 30]})
        fig = routine_chart(data, {"Routine": 2, "Infrequent": 8}, share=True)
        self.assertTrue(all(not trace.stackgroup for trace in fig.data))
        self.assertEqual(tuple(fig.layout.xaxis.ticktext), ("2023", "2024"))
        self.assertEqual(tuple(fig.data[1].y), (58, 30))

    def test_count_mode_remains_additive(self):
        data = pd.DataFrame({"period": pd.to_datetime(["2023-01-01"] * 2),
                             "cohort": ["Routine", "Infrequent"],
                             "notices": [42, 58], "share_percent": [42, 58]})
        fig = routine_chart(data, {}, share=False)
        self.assertTrue(all(trace.stackgroup == "notices" for trace in fig.data))

    def test_monthly_timeline_preserves_notice_total(self):
        history = pd.DataFrame({"Date_Filed": pd.to_datetime(["2024-02-01", "2024-02-15"]),
                                "Cause_Category": ["Non-payment of rent"] * 2,
                                "Eviction_Count": [3, 5], "Total_Rent_Owed": [100, 200]})
        fig = address_timeline(history)
        self.assertEqual(sum(fig.data[0].y), 8)
        self.assertEqual(len(fig.data[0].x), 1)
        self.assertGreater(fig.data[0].width, 86400000)


if __name__ == "__main__":
    unittest.main()
