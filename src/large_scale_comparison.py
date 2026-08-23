"""Large-scale complement to Section 6's hand-labeled evaluation.

The hand-labeled comparison (src/evaluate.py) is small (n=106) because
hand-labeling doesn't scale, its bootstrap CIs are wide enough that the
headline has_narrative finding can't be distinguished from noise. This
module runs the same has_narrative-split comparison on the model's test
split instead: 1,120 rows, roughly 20x the hand-labeled sample, scored
against the proxy label rather than genuine human judgment.

Neither measurement is unbiased on its own:
- Hand-labeled: closer to genuine severity judgment, but tiny and (as
  Section 6 found) can itself pick up a labeling-consistency bias.
- Proxy-labeled test split: huge, so far more statistically powerful, but
  inherits the proxy label's own bias (companies concede cheaply on minor
  issues, deny substantive ones outright, see README Methodology).

One thing works in the proxy label's favor here that didn't hold for the
hand-labeled sample: the proxy label was deliberately built (Section 2) to
be independent of has_narrative, so this comparison isn't confounded by
narrative presence correlating with the label the way the hand-labeled
sample turned out to be.

Where the two measurements agree, that's real triangulated evidence. Where
they disagree, that's informative about which bias is dominating.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config
from src.evaluate import bootstrap_auc_ci, format_metric, score_from_proba
from src.models import (
    ModelsError,
    build_baseline_matrices,
    build_fused_matrices,
    load_structured_features,
    load_text_embeddings,
    make_train_test_split,
)

logger = logging.getLogger(__name__)


class LargeScaleComparisonError(Exception):
    """Raised when required upstream files are missing."""


def load_narrative_flags(path: Path) -> dict[str, bool]:
    """Load has_narrative per complaint_id from the proxy-labeled dataset."""
    if not path.exists():
        raise LargeScaleComparisonError(
            f"Labeled data file not found at {path}. Run `python -m src.label` first."
        )
    with open(path) as f:
        records = json.load(f)
    return {r["complaint_id"]: bool(r["has_narrative"]) for r in records}


def compute_test_split_comparison(
    baseline_model: Any,
    fused_model: Any,
    Xb_test: pd.DataFrame,
    Xf_test: pd.DataFrame,
    y_test: pd.Series,
    narrative_flags: dict[str, bool],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Score both models on the full test split and the has_narrative/no_narrative
    subsets, with a paired bootstrap CI on the AUC delta for each.
    """
    has_narrative = pd.Series(
        [narrative_flags[cid] for cid in y_test.index], index=y_test.index
    )
    subgroup_masks = {
        "overall": pd.Series(True, index=y_test.index),
        "has_narrative": has_narrative,
        "no_narrative": ~has_narrative,
    }

    results: dict[str, dict[str, Any]] = {}
    for subgroup_name, mask in subgroup_masks.items():
        y_sub = y_test[mask]
        b_proba = baseline_model.predict_proba(Xb_test[mask])[:, 1]
        f_proba = fused_model.predict_proba(Xf_test[mask])[:, 1]
        y_arr = y_sub.to_numpy()

        results[subgroup_name] = {
            "baseline": score_from_proba(b_proba, y_arr, 0.5),
            "fused": score_from_proba(f_proba, y_arr, 0.5),
            "bootstrap": bootstrap_auc_ci(y_arr, b_proba, f_proba, n_bootstrap, seed),
        }
    return results


def generate_report(results: dict[str, dict[str, Any]]) -> str:
    lines = ["# Large-Scale Test-Split Comparison (proxy label, n=1,120)\n"]
    lines.append(
        "Complement to `comparison_report.md` (the hand-labeled comparison). "
        "Same has_narrative split, ~20x the sample size, scored against the "
        "proxy label instead of hand-judged ground truth. The proxy label was "
        "deliberately built independent of has_narrative (Section 2), so this "
        "comparison is not confounded by narrative presence the way the "
        "hand-labeled sample turned out to be. See module docstring in "
        "`src/large_scale_comparison.py` for the full tradeoff between the two "
        "measurements.\n"
    )

    lines.append("## AUC comparison\n")
    lines.append("| Subgroup | n | Baseline AUC | Fused AUC | Delta | Delta 95% CI | Fused wins in resamples |")
    lines.append("|---|---|---|---|---|---|---|")
    for subgroup in ["overall", "has_narrative", "no_narrative"]:
        b, f = results[subgroup]["baseline"], results[subgroup]["fused"]
        boot = results[subgroup]["bootstrap"]
        delta = f"{f['auc'] - b['auc']:+.4f}" if b["auc"] is not None and f["auc"] is not None else "n/a"
        ci = f"[{boot['delta_ci'][0]:+.4f}, {boot['delta_ci'][1]:+.4f}]" if boot["delta_ci"] else "undefined"
        pct = f"{boot['pct_delta_positive']:.1%}" if boot["pct_delta_positive"] is not None else "n/a"
        lines.append(
            f"| {subgroup} | {b['n']} | {format_metric(b, 'auc')} | {format_metric(f, 'auc')} | "
            f"{delta} | {ci} | {pct} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_large_scale_comparison(config_path: Path | None = None) -> Path:
    config = load_config(config_path) if config_path is not None else load_config()

    structured_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["structured_features_filename"]
    )
    embeddings_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["text_embeddings_filename"]
    )
    labeled_path = Path(config["data"]["processed_dir"]) / config["data"]["labeled_filename"]

    try:
        structured_df = load_structured_features(structured_path)
        embeddings_df = load_text_embeddings(embeddings_path)
    except ModelsError as e:
        raise LargeScaleComparisonError(str(e)) from e

    narrative_flags = load_narrative_flags(labeled_path)

    train_ids, test_ids = make_train_test_split(structured_df, config)

    models_dir = Path(config["models"]["dir"])
    baseline_model = joblib.load(models_dir / config["models"]["baseline_filename"])
    fused_model = joblib.load(models_dir / config["models"]["fused_filename"])

    _, Xb_test, _, yb_test = build_baseline_matrices(structured_df, train_ids, test_ids)
    _, Xf_test, _, yf_test = build_fused_matrices(structured_df, embeddings_df, train_ids, test_ids)

    n_bootstrap = config["evaluate"]["n_bootstrap"]
    results = compute_test_split_comparison(
        baseline_model, fused_model, Xb_test, Xf_test, yb_test,
        narrative_flags, n_bootstrap, config["seed"],
    )

    for subgroup in ["overall", "has_narrative", "no_narrative"]:
        b, f = results[subgroup]["baseline"], results[subgroup]["fused"]
        boot = results[subgroup]["bootstrap"]
        ci_str = f"[{boot['delta_ci'][0]:+.4f}, {boot['delta_ci'][1]:+.4f}]" if boot["delta_ci"] else "undefined"
        logger.info(
            "%s (n=%d): baseline AUC=%s, fused AUC=%s, delta 95%% CI=%s, fused wins %s of resamples.",
            subgroup, b["n"], format_metric(b, "auc"), format_metric(f, "auc"), ci_str,
            f"{boot['pct_delta_positive']:.1%}" if boot["pct_delta_positive"] is not None else "n/a",
        )

    report_text = generate_report(results)
    output_path = Path(config["data"]["processed_dir"]) / "test_split_comparison_report.md"
    output_path.write_text(report_text)
    logger.info("Saved large-scale comparison report to %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output_path = run_large_scale_comparison()
    except LargeScaleComparisonError as e:
        print(f"Large-scale comparison failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Large-scale comparison complete: report written to {output_path}")


if __name__ == "__main__":
    main()
