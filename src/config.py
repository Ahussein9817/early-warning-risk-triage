"""Loads config.yaml so every module reads tunable values from one place."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the project config as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)
