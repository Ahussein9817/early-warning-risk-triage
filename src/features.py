"""Section 3 — Structured features.

Builds the structured-only feature matrix used to train the baseline model
(Stage 1). Deliberately excludes company_response/timely (the label's own
inputs — see label.py) and product (constant across this dataset), and
buckets the high-cardinality `company` field into top-N + "Other" — see
config.yaml `features` for the full rationale.

Acceptance criteria this module supports (see BUILD_INSTRUCTIONS.md):
- Baseline model input is structured-only, independent of the label rule's
  own inputs and of the text signal being tested in the fused model.
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


class FeaturesError(Exception):
    """Raised when the labeled dataset is missing, empty, or malformed."""


def load_labeled_data(path: Path) -> list[dict[str, Any]]:
    """Load the labeled dataset, raising FeaturesError on missing/empty/malformed input."""
    if not path.exists():
        raise FeaturesError(
            f"Labeled data file not found at {path}. Run `python -m src.label` first."
        )

    with open(path) as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError as e:
            raise FeaturesError(f"Labeled data file at {path} is not valid JSON: {e}") from e

    if not records:
        raise FeaturesError(f"Labeled data file at {path} is empty.")
    if not isinstance(records, list):
        raise FeaturesError(
            f"Labeled data file at {path} does not contain a list of records "
            f"(got {type(records).__name__})."
        )
    return records


def compute_top_companies(records: list[dict[str, Any]], n: int) -> list[str]:
    """Return the `n` most frequent `company` values in `records`, most frequent first."""
    counts = Counter(r["company"] for r in records)
    return [company for company, _ in counts.most_common(n)]


def bucket_company(company: str, top_companies: list[str]) -> str:
    """Return `company` unchanged if it's in `top_companies`, else "Other"."""
    return company if company in top_companies else "Other"


def build_structured_features(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Build the one-hot-encoded structured feature matrix, indexed by complaint_id.

    Returns the feature DataFrame (including `escalated` and `complaint_id`
    columns alongside the encoded features) and the list of top companies used
    for bucketing, so callers can log/document the encoding that was applied.
    """
    features_cfg = config["features"]
    categorical_fields = features_cfg["categorical_fields"]
    company_top_n = features_cfg["company_top_n"]

    top_companies = compute_top_companies(records, company_top_n)
    top_companies_set = set(top_companies)
    covered = sum(1 for r in records if r["company"] in top_companies_set)
    coverage_pct = 100 * covered / len(records)
    logger.info(
        "Bucketing company into top %d + 'Other' (covers %d/%d records, %.2f%%). "
        "Top companies: %s",
        company_top_n, covered, len(records), coverage_pct, top_companies,
    )

    df = pd.DataFrame(records)
    df["company_bucketed"] = df["company"].apply(lambda c: bucket_company(c, top_companies))

    encode_cols = list(categorical_fields) + ["company_bucketed"]
    dummies = pd.get_dummies(df[encode_cols], columns=encode_cols)

    feature_matrix = pd.concat(
        [df[["complaint_id", "escalated"]], dummies], axis=1
    )
    return feature_matrix, top_companies


def save_features(feature_matrix: pd.DataFrame, output_path: Path) -> None:
    """Write the feature matrix to `output_path` as CSV and log its shape."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(output_path, index=False)
    n_feature_cols = feature_matrix.shape[1] - 2  # exclude complaint_id, escalated
    logger.info(
        "Saved structured feature matrix to %s (%d rows, %d feature columns).",
        output_path, feature_matrix.shape[0], n_feature_cols,
    )


def run_features(config_path: Path | None = None) -> Path:
    """Orchestrate Section 3: load labeled data, build features, save, return path."""
    config = load_config(config_path) if config_path is not None else load_config()

    labeled_path = Path(config["data"]["processed_dir"]) / config["data"]["labeled_filename"]
    records = load_labeled_data(labeled_path)

    feature_matrix, _ = build_structured_features(records, config)

    output_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["structured_features_filename"]
    )
    save_features(feature_matrix, output_path)
    return output_path


def main() -> None:
    """CLI entry point: run Section 3, print a clear message and exit(1) on
    FeaturesError instead of a raw traceback.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output_path = run_features()
    except FeaturesError as e:
        print(f"Feature build failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Feature build complete: structured features written to {output_path}")


if __name__ == "__main__":
    main()
