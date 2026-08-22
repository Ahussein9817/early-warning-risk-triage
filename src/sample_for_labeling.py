"""Generates the hand-labeling template for the Section 6 evaluation sample.

Draws a random sample from the model test split (never the train split — the
whole point is to evaluate on rows the models never trained on) for the
builder to manually review and label. The exported CSV deliberately excludes
company_response, timely, and the proxy `escalated` label — those are what
the proxy label rule is built from, so seeing them while hand-labeling would
just reproduce the proxy's own answer rather than provide an independent
validation signal (same leakage principle as label.py/features.py, applied
to human judgment instead of a model).
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.models import load_structured_features, make_train_test_split

logger = logging.getLogger(__name__)

EXPORTED_FIELDS = [
    "complaint_id", "product", "sub_product", "issue", "sub_issue",
    "state", "company", "date_received", "complaint_what_happened",
]


class SamplingError(Exception):
    """Raised when required upstream files are missing."""


def load_labeled_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load the proxy-labeled dataset, keyed by complaint_id."""
    if not path.exists():
        raise SamplingError(
            f"Labeled data file not found at {path}. Run `python -m src.label` first."
        )
    with open(path) as f:
        records = json.load(f)
    return {r["complaint_id"]: r for r in records}


def sample_test_ids(test_ids: list[str], n: int, seed: int) -> list[str]:
    """Deterministically sample `n` complaint_ids from the test split."""
    rng = random.Random(seed)
    return rng.sample(test_ids, n)


def build_template(sample_ids: list[str], records_by_id: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the hand-labeling template DataFrame for `sample_ids`."""
    rows = []
    for cid in sample_ids:
        record = records_by_id[cid]
        row = {field: record.get(field) for field in EXPORTED_FIELDS}
        row["complaint_what_happened"] = row["complaint_what_happened"] or ""
        row["escalated"] = ""
        row["labeler_notes"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def run_sampling(config_path: Path | None = None) -> Path:
    """Orchestrate: load labeled data + test split, sample, write template, return path."""
    config = load_config(config_path) if config_path is not None else load_config()

    labeled_path = Path(config["data"]["processed_dir"]) / config["data"]["labeled_filename"]
    records_by_id = load_labeled_records(labeled_path)

    structured_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["structured_features_filename"]
    )
    structured_df = load_structured_features(structured_path)
    _, test_ids = make_train_test_split(structured_df, config)

    n = config["data"]["hand_label_sample_size"]
    sample_ids = sample_test_ids(test_ids, n, config["seed"])
    logger.info(
        "Sampled %d complaint_ids from the %d-row test split for hand-labeling.",
        len(sample_ids), len(test_ids),
    )

    template_df = build_template(sample_ids, records_by_id)
    n_narrative = (template_df["complaint_what_happened"] != "").sum()
    logger.info(
        "Template composition: %d/%d have a narrative (%.1f%%).",
        n_narrative, len(template_df), 100 * n_narrative / len(template_df),
    )

    output_dir = Path(config["data"]["labeled_path"]).parent
    output_path = output_dir / config["data"]["hand_label_template_filename"]
    output_dir.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(output_path, index=False)
    logger.info("Saved hand-labeling template to %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output_path = run_sampling()
    except SamplingError as e:
        print(f"Sampling failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Sampling complete: template written to {output_path}")


if __name__ == "__main__":
    main()
