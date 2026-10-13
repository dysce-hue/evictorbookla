import unittest
from unittest.mock import patch
import plotly.graph_objects as go
from ui_helpers import embed_html, readable_timeline, readable_neighborhoods


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


if __name__ == "__main__":
    unittest.main()
