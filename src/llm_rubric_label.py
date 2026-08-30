"""LLM-applied version of the hand-labeling rubric (README Methodology,
"Hand-labeling rubric" section), used to scale evaluation beyond the 106-row
hand-labeled sample without pretending it IS hand-labeled data.

This produces `llm_rubric_label`, a distinct column/file from the real
`escalated` column in hand_labeled.csv. It is an approximation, validated
against the real hand labels (Step 1 below), not a second source of ground
truth. Never merge llm_rubric_label results into the hand-labeled comparison
report, keep them reported side by side per the module's caller.

Step 1 (this module's primary entry point, `run_validation`): apply the
rubric via `claude -p` to the same 106 rows already hand-labeled, blind to
the real labels (the prompt never sees `escalated`, `company_response`, or
`timely` — same leakage exclusion as the original hand-labeling template).
Report agreement rate and a confusion matrix against the real hand labels.

Step 2 (scale to the remaining ~1,014 test-split rows) only happens if the
user decides Step 1's agreement rate is strong enough, this module doesn't
make that call itself.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

# These 3 hand-labeled rows were used as worked examples in the prompt below
# (drawn from real disagreement cases to calibrate the rubric). Re-validating
# on the same 106 rows means these 3 are no longer blind, the model sees
# something close to their own answer embedded in the prompt. Excluded from
# the headline agreement statistic; reported separately for transparency.
WORKED_EXAMPLE_COMPLAINT_IDS = {"8671767", "7619977", "10305892"}

RUBRIC_PROMPT_TEMPLATE = """You are judging a CFPB debt-collection complaint using this rubric only. \
Respond with exactly one character: 1 or 0. No other text, no explanation.

Rubric: does this complaint describe a specific, substantiated, serious problem? \
Examples of serious: debt not owed with specifics, identity theft, threatened or \
actual lawsuit without proper notice, repeated unwanted contact after being told \
to stop, real financial or credit harm. Examples of NOT serious: vague complaints, \
minor procedural issues, complaints without concrete substantiation.

If a narrative is present, judge primarily from its content. If no narrative is \
present, judge from the issue/sub_issue category alone, using the exact same \
standard as when a narrative is present, do not default to "not serious" just \
because there is no narrative to read.

When a case is ambiguous or borderline, resolve toward escalated=1 (serious), not \
toward 0. Do not default to "not serious" under uncertainty.

Worked examples (all correctly labeled escalated=1 despite being borderline, not \
obviously extreme cases):

Example A (narrative present): issue="Electronic communications", sub_issue="Frequent \
or repeated messages". Narrative: a company is trying to collect a debt from someone \
who says they have been seriously ill for about a year, lost their income, and does \
not know what to do. Labeled 1: describes ongoing, specific collection pressure on \
someone in genuine financial/health hardship, a real, substantiated harm even though \
it is not one single dramatic violation.

Example B (narrative present): issue="Attempts to collect debt not owed", sub_issue=\
"Debt is not yours". Narrative: consumer alleges a debt collector publicly reported \
account information without authority, written consent, or a court order, citing the \
Fair Debt Collection Practices Act. Labeled 1: a specific, substantiated claim of an \
unauthorized action and a named legal violation, not a vague complaint.

Example C (no narrative): issue="Took or threatened to take negative or legal action", \
sub_issue="Threatened or suggested your credit would be damaged". No narrative text. \
Labeled 1 from the category alone: this sub_issue itself describes a coercive threat \
against the consumer; the same standard applied to a narrative describing a threat \
would call this serious, so the category alone should too.

Complaint fields:
product: {product}
sub_product: {sub_product}
issue: {issue}
sub_issue: {sub_issue}
state: {state}
company: {company}
date_received: {date_received}
narrative: {narrative}

Respond with exactly one character: 1 (serious, escalated) or 0 (not serious, not escalated)."""


class LlmRubricLabelError(Exception):
    """Raised when required upstream files are missing or the LLM call fails unexpectedly."""


def build_prompt(record: dict[str, Any]) -> str:
    """Build the rubric prompt for one record. Never includes company_response,
    timely, or escalated, the same leakage exclusion as the human hand-labeling
    template.
    """
    return RUBRIC_PROMPT_TEMPLATE.format(
        product=record.get("product") or "(missing)",
        sub_product=record.get("sub_product") or "(missing)",
        issue=record.get("issue") or "(missing)",
        sub_issue=record.get("sub_issue") or "(missing)",
        state=record.get("state") or "(missing)",
        company=record.get("company") or "(missing)",
        date_received=record.get("date_received") or "(missing)",
        narrative=record.get("complaint_what_happened") or "(no narrative provided)",
    )


def call_llm(prompt: str, model: str, timeout_seconds: int = 60) -> str:
    """Call `claude -p` with `prompt`, return the raw stdout text."""
    result = subprocess.run(
        ["claude", "-p", prompt, "--model", model, "--output-format", "text"],
        capture_output=True, text=True, timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise LlmRubricLabelError(f"claude -p exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def parse_label(raw_response: str) -> int | None:
    """Parse a 1/0 label from the raw response. Returns None (not a guess) if
    the response isn't cleanly parseable, rather than silently defaulting.
    """
    cleaned = raw_response.strip()
    if cleaned == "1":
        return 1
    if cleaned == "0":
        return 0
    # tolerate a trailing period or surrounding whitespace/quotes, nothing looser
    stripped = cleaned.strip(".\"' \n\t")
    if stripped == "1":
        return 1
    if stripped == "0":
        return 0
    return None


def label_records(
    records: list[dict[str, Any]], model: str, log_every: int = 10
) -> pd.DataFrame:
    """Apply the rubric to each record via the LLM, returning a DataFrame of
    complaint_id, llm_rubric_label (nullable int), raw_response.
    """
    rows = []
    for i, record in enumerate(records, start=1):
        prompt = build_prompt(record)
        try:
            raw = call_llm(prompt, model)
            label = parse_label(raw)
            if label is None:
                logger.warning(
                    "complaint_id=%s: unparseable LLM response %r, recording as null.",
                    record["complaint_id"], raw,
                )
        except (subprocess.TimeoutExpired, LlmRubricLabelError) as e:
            logger.warning("complaint_id=%s: LLM call failed (%s), recording as null.",
                            record["complaint_id"], e)
            raw, label = None, None

        rows.append({"complaint_id": record["complaint_id"], "llm_rubric_label": label, "raw_response": raw})
        if i % log_every == 0 or i == len(records):
            logger.info("Labeled %d/%d records.", i, len(records))

    return pd.DataFrame(rows)


def compute_agreement(
    llm_df: pd.DataFrame, hand_df: pd.DataFrame, exclude_ids: set[str] | None = None
) -> dict[str, Any]:
    """Compare llm_rubric_label against the real hand-labeled escalated column.
    `exclude_ids` (e.g. rows used as in-prompt worked examples) are dropped
    before computing the headline stats, since they're no longer blind.
    """
    merged = llm_df.merge(hand_df[["complaint_id", "escalated"]], on="complaint_id", how="inner")
    if exclude_ids:
        merged = merged[~merged["complaint_id"].isin(exclude_ids)]
    valid = merged.dropna(subset=["llm_rubric_label"])
    n_unparseable = len(merged) - len(valid)

    agree = (valid["llm_rubric_label"] == valid["escalated"]).sum()
    n_valid = len(valid)
    agreement_rate = agree / n_valid if n_valid > 0 else None

    confusion = {
        "both_escalated": int(((valid["llm_rubric_label"] == 1) & (valid["escalated"] == 1)).sum()),
        "both_not_escalated": int(((valid["llm_rubric_label"] == 0) & (valid["escalated"] == 0)).sum()),
        "llm_escalated_hand_not": int(((valid["llm_rubric_label"] == 1) & (valid["escalated"] == 0)).sum()),
        "llm_not_hand_escalated": int(((valid["llm_rubric_label"] == 0) & (valid["escalated"] == 1)).sum()),
    }

    return {
        "n_total": len(merged),
        "n_valid": n_valid,
        "n_unparseable": n_unparseable,
        "agreement_rate": agreement_rate,
        "confusion": confusion,
    }


def load_hand_labeled_records(path: Path) -> list[dict[str, Any]]:
    """Load records as plain dicts, with pandas's NaN normalized to None so
    `record.get(field) or default` works correctly (NaN is truthy in Python,
    an empty CSV cell would otherwise silently fail to fall back to default).
    """
    if not path.exists():
        raise LlmRubricLabelError(f"Hand-labeled file not found at {path}.")
    df = pd.read_csv(path, dtype={"complaint_id": str})
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def run_validation(config_path: Path | None = None, model: str = "sonnet") -> Path:
    """Step 1: label the same 106 hand-labeled rows, blind, and report agreement
    against the real labels. Does NOT proceed to the full test split.
    """
    config = load_config(config_path) if config_path is not None else load_config()

    hand_path = Path(config["data"]["labeled_path"])
    hand_records = load_hand_labeled_records(hand_path)
    hand_df = pd.read_csv(hand_path, dtype={"complaint_id": str})

    logger.info("Step 1: applying LLM rubric to %d hand-labeled rows (blind to real labels).", len(hand_records))
    llm_df = label_records(hand_records, model)

    output_dir = Path(config["data"]["labeled_path"]).parent
    output_path = output_dir / "llm_rubric_labels_validation.csv"
    llm_df.to_csv(output_path, index=False)
    logger.info("Saved LLM rubric labels to %s", output_path)

    agreement = compute_agreement(llm_df, hand_df, exclude_ids=WORKED_EXAMPLE_COMPLAINT_IDS)
    logger.info(
        "Agreement (excluding %d in-prompt worked-example rows, no longer blind): "
        "%d/%d valid comparisons (%d unparseable), agreement rate=%.1f%%.",
        len(WORKED_EXAMPLE_COMPLAINT_IDS),
        agreement["n_valid"], agreement["n_total"], agreement["n_unparseable"],
        100 * agreement["agreement_rate"] if agreement["agreement_rate"] is not None else float("nan"),
    )
    logger.info("Confusion: %s", agreement["confusion"])

    example_rows = llm_df[llm_df["complaint_id"].isin(WORKED_EXAMPLE_COMPLAINT_IDS)].merge(
        hand_df[["complaint_id", "escalated"]], on="complaint_id"
    )
    logger.info("Worked-example rows (not blind, shown separately): %s",
                example_rows[["complaint_id", "llm_rubric_label", "escalated"]].to_dict(orient="records"))

    report = {
        **agreement,
        "excluded_worked_example_ids": sorted(WORKED_EXAMPLE_COMPLAINT_IDS),
        "worked_example_rows_not_blind": example_rows[
            ["complaint_id", "llm_rubric_label", "escalated"]
        ].to_dict(orient="records"),
    }
    report_path = output_dir / "llm_rubric_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved validation report to %s", report_path)

    return report_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report_path = run_validation()
    except LlmRubricLabelError as e:
        print(f"LLM rubric validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Step 1 validation complete: report written to {report_path}")


if __name__ == "__main__":
    main()
