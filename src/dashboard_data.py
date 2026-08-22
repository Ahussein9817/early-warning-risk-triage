"""Prepares the small data bundle the dashboard needs: valid category options
for each structured field (so dropdown selections always map to a real
trained one-hot column) and the top-N company list. Kept separate from the
full 5,600-row datasets so the deployed dashboard only bundles what it needs.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import load_config
from src.features import compute_top_companies
from src.label import load_raw_data

logger = logging.getLogger(__name__)


class DashboardDataError(Exception):
    """Raised when the labeled dataset is missing."""


def build_categories(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Build the category option lists for each dropdown field, plus the
    top-N company list used for bucketing.
    """
    categorical_fields = config["features"]["categorical_fields"]
    company_top_n = config["features"]["company_top_n"]

    categories = {}
    for field in categorical_fields:
        values = sorted({r[field] for r in records if r.get(field)})
        categories[field] = values

    categories["company_options"] = compute_top_companies(records, company_top_n) + ["Other"]
    return categories


def run_dashboard_data(config_path: Path | None = None) -> Path:
    config = load_config(config_path) if config_path is not None else load_config()

    labeled_path = Path(config["data"]["processed_dir"]) / config["data"]["labeled_filename"]
    try:
        records = load_raw_data(labeled_path)
    except Exception as e:
        raise DashboardDataError(f"Could not load labeled data at {labeled_path}: {e}") from e

    categories = build_categories(records, config)

    output_path = Path(config["data"]["processed_dir"]) / "dashboard_categories.json"
    output_path.write_text(json.dumps(categories, indent=2))
    logger.info("Saved dashboard categories to %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output_path = run_dashboard_data()
    except DashboardDataError as e:
        print(f"Dashboard data prep failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Dashboard data prep complete: {output_path}")


if __name__ == "__main__":
    main()
