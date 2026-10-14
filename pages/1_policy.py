"""Expose BETA when Community Cloud still uses the original app.py entry point."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "app_beta.py"), run_name="__main__")
