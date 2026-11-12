"""Slow smoke test: runs dashboard/app.py headlessly with Streamlit's AppTest.

Loads both committed models and downloads the MiniLM weights (~90 MB) on first
run, so it is skipped unless RUN_SLOW=1.  Run with: RUN_SLOW=1 pytest tests/test_dashboard_smoke.py
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run")


def test_dashboard_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.button) >= 1  # the Predict button is present
