"""Section 2, Label.

Derives the proxy escalation label for every complaint in the raw ingested
dataset. Isolated in its own module so the rule is auditable, see
config.yaml `label.relief_company_responses` and BUILD_INSTRUCTIONS.md
Architecture section for the rule and its rationale (consumer_disputed was
discontinued by CFPB in 2017; timely substitutes for it).

Rule: escalated = 1 if company_response is a relief category, OR
timely == "No", else 0. Deliberately excludes has_narrative /
complaint_what_happened so the label stays independent of the text signal
the fused model is being tested on.

Acceptance criteria this module is responsible for (see BUILD_INSTRUCTIONS.md):
- The proxy-label rule is printed/logged when applied, not silently baked in.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import load_config

logger = logging.getLogger(__name__)


class LabelError(Exception):
    """Raised when the raw dataset is missing, empty, or malformed."""


def compute_escalated_label(record: dict[str, Any], relief_responses: list[str]) -> int:
    """Return 1 if `record` meets the proxy escalation rule, else 0.

    Defensive: if both company_response and timely are missing (None), the
    rule's inputs can't actually determine escalation. Logs a warning and
    defaults to 0 rather than silently guessing via Python's None-is-falsy
    behavior (None in [...] and None == "No" both evaluate to False without
    this check, which would look identical to a real "not escalated" case).
    """
    company_response = record.get("company_response")
    timely = record.get("timely")

    if company_response is None and timely is None:
        logger.warning(
            "complaint_id=%s has null company_response and timely; "
            "defaulting escalated=0 (cannot determine from proxy rule inputs).",
            record.get("complaint_id"),
        )
        return 0

    if company_response in relief_responses:
        return 1
    if timely == "No":
        return 1
    return 0


def apply_labels(
    records: list[dict[str, Any]], relief_responses: list[str]
) -> list[dict[str, Any]]:
    """Add an `escalated` field to every record using the proxy label rule.

    Logs the rule in plain language once, so it's auditable rather than
    silently baked into the code.
    """
    logger.info(
        "Applying proxy label rule: escalated=1 if company_response in %s "
        "OR timely=='No', else 0.",
        relief_responses,
    )
    labeled = []
    for record in records:
        labeled_record = dict(record)
        labeled_record["escalated"] = compute_escalated_label(record, relief_responses)
        labeled.append(labeled_record)
    return labeled


def log_label_distribution(labeled_records: list[dict[str, Any]]) -> None:
    """Log the count and percentage of escalated vs. non-escalated records."""
    total = len(labeled_records)
    escalated = sum(r["escalated"] for r in labeled_records)
    not_escalated = total - escalated
    logger.info(
        "Label distribution: %d/%d escalated (%.1f%%), %d/%d not escalated (%.1f%%).",
        escalated, total, 100 * escalated / total,
        not_escalated, total, 100 * not_escalated / total,
    )


def load_raw_data(path: Path) -> list[dict[str, Any]]:
    """Load the raw ingested dataset, raising LabelError on missing/empty/malformed input."""
    if not path.exists():
        raise LabelError(
            f"Raw data file not found at {path}. Run `python -m src.ingest` first."
        )

    with open(path) as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError as e:
            raise LabelError(f"Raw data file at {path} is not valid JSON: {e}") from e

    if not records:
        raise LabelError(f"Raw data file at {path} is empty.")
    if not isinstance(records, list):
        raise LabelError(
            f"Raw data file at {path} does not contain a list of records "
            f"(got {type(records).__name__})."
        )
    return records


def save_labeled_data(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write labeled records to `output_path` as JSON and log the record count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Saved %d labeled records to %s", len(records), output_path)


def run_label(config_path: Path | None = None) -> Path:
    """Orchestrate the label section: load raw data, apply labels, save, return path."""
    config = load_config(config_path) if config_path is not None else load_config()

    raw_path = Path(config["data"]["raw_dir"]) / config["data"]["raw_filename"]
    records = load_raw_data(raw_path)

    relief_responses = config["label"]["relief_company_responses"]
    labeled_records = apply_labels(records, relief_responses)
    log_label_distribution(labeled_records)

    output_path = Path(config["data"]["processed_dir"]) / config["data"]["labeled_filename"]
    save_labeled_data(labeled_records, output_path)
    return output_path


def main() -> None:
    """CLI entry point: run label, print a clear message and exit(1) on
    LabelError instead of a raw traceback.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output_path = run_label()
    except LabelError as e:
        print(f"Label failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Label complete: labeled data written to {output_path}")


if __name__ == "__main__":
    main()
