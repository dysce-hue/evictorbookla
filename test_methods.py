"""Coverage and focused rendering checks for the shared methods reference."""
import unittest
from pathlib import Path

from methods_ui import METHODS, methods_markdown


class MethodsTests(unittest.TestCase):
    def test_inventory_and_export(self):
        expected = {"counting", "time_composition", "rent", "concentration", "routine", "spatial", "correlation",
                    "rent_comparison", "anomalies", "ols", "attention", "features", "logistic", "validation", "xgboost", "intervals", "map_sources"}
        self.assertEqual({m["id"] for m in METHODS}, expected)
        self.assertEqual(len(METHODS), len(expected))
        text = methods_markdown()
        for method in METHODS:
            self.assertIn(method["title"], text)
            self.assertTrue(method["equations"] and method["sources"])
            for source in method["sources"].split(";"):
                filename = source.strip().split(":")[0]
                self.assertTrue(Path(filename).exists(), filename)

    def test_render_all_and_selected_method(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file("math_page.py").run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.expander), len(METHODS))
        selected = next(m for m in METHODS if m["id"] == "ols")
        app.selectbox[0].set_value(selected["title"]).run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.expander), 1)
        self.assertEqual(app.expander[0].label, selected["title"])
        app.selectbox[0].set_value("All methods").run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.expander), len(METHODS))


if __name__ == "__main__":
    unittest.main()
