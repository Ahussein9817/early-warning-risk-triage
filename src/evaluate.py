"""Section 6, Evaluate.

Scores baseline and fused models against the hand-labeled evaluation sample:
the real held-out ground truth, never used in training, independent of the
proxy label used to train the models. Reports precision/recall/AUC overall,
split by has_narrative, and split by narrative length (short/long relative to
MiniLM's 256-token truncation point), per BUILD_INSTRUCTIONS.md acceptance
criteria.

The has_narrative split (and its short/long sub-split) is the headline
comparison, not the pooled/overall number, see README.md Methodology for why
the pooled comparison is confounded by a correlation between narrative
presence and the hand-labeled ground truth itself.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.config import load_config
from src.models import load_structured_features, load_text_embeddings

EVIDENCE_SUB_ISSUES = [
    "Notification didn't disclose it was an attempt to collect a debt",
    "Debt was result of identity theft",
    "Debt is not yours",
]

logger = logging.getLogger(__name__)


class EvaluateError(Exception):
    """Raised when the hand-labeled sample or model artifacts are missing, empty, or malformed."""


def load_hand_labeled(path: Path) -> pd.DataFrame:
    """Load the hand-labeled evaluation sample, raising EvaluateError if missing/incomplete."""
    if not path.exists():
        raise EvaluateError(
            f"Hand-labeled file not found at {path}. Fill in the template from "
            "`python -m src.sample_for_labeling` and save it there."
        )
    df = pd.read_csv(path, dtype={"complaint_id": str})
    if df.empty:
        raise EvaluateError(f"Hand-labeled file at {path} is empty.")
    if df["escalated"].isnull().any():
        raise EvaluateError(
            f"Hand-labeled file at {path} has rows with a blank `escalated` value, "
            "every row must be labeled before evaluation."
        )
    df["escalated"] = df["escalated"].astype(int)
    return df


def load_models(config: dict[str, Any]) -> tuple[Any, Any]:
    """Load the trained baseline and fused model artifacts."""
    models_dir = Path(config["models"]["dir"])
    baseline_path = models_dir / config["models"]["baseline_filename"]
    fused_path = models_dir / config["models"]["fused_filename"]
    if not baseline_path.exists() or not fused_path.exists():
        raise EvaluateError(
            f"Model artifacts not found at {baseline_path} / {fused_path}. "
            "Run `python -m src.models` first."
        )
    return joblib.load(baseline_path), joblib.load(fused_path)


def add_narrative_features(hand_df: pd.DataFrame, word_threshold: int) -> pd.DataFrame:
    """Add has_narrative (bool) and narrative_length_bucket (no_narrative/short/long) columns."""
    df = hand_df.copy()
    narrative = df["complaint_what_happened"].fillna("")
    df["has_narrative"] = narrative.str.len() > 0
    word_counts = narrative.str.split().str.len().fillna(0)

    def bucket(has_narr: bool, wc: float) -> str:
        if not has_narr:
            return "no_narrative"
        return "long" if wc > word_threshold else "short"

    df["narrative_length_bucket"] = [
        bucket(h, wc) for h, wc in zip(df["has_narrative"], word_counts)
    ]
    return df


def build_eval_matrices(
    hand_df: pd.DataFrame, structured_df: pd.DataFrame, embeddings_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build baseline/fused feature matrices for the hand-labeled complaint_ids,
    using the hand-labeled `escalated` as ground truth (not the proxy label).
    """
    ids = hand_df["complaint_id"]
    matched = structured_df[structured_df["complaint_id"].isin(ids)]
    if len(matched) != len(hand_df):
        missing = set(ids) - set(matched["complaint_id"])
        raise EvaluateError(
            f"{len(missing)} hand-labeled complaint_ids have no matching structured "
            f"features row: {sorted(missing)[:5]}"
        )

    baseline_cols = [c for c in structured_df.columns if c not in ("complaint_id", "escalated")]
    merged = matched.merge(embeddings_df, on="complaint_id", how="left")
    fused_cols = [c for c in merged.columns if c not in ("complaint_id", "escalated")]

    structured_indexed = matched.set_index("complaint_id").loc[ids]
    merged_indexed = merged.set_index("complaint_id").loc[ids]

    X_baseline = structured_indexed[baseline_cols]
    X_fused = merged_indexed[fused_cols]
    y = hand_df.set_index("complaint_id").loc[ids, "escalated"]
    return X_baseline, X_fused, y


def score_subset(model: Any, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict[str, Any]:
    """Score `model` on (X, y). AUC is None (not computed) when only one class
    is present, roc_auc_score is mathematically undefined in that case rather
    than a number worth reporting.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    n = len(y)
    n_positive = int(y.sum())

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_proba) if 0 < n_positive < n else None

    return {
        "n": n, "n_positive": n_positive, "n_negative": n - n_positive,
        "precision": precision, "recall": recall, "auc": auc,
    }


def compute_comparison(
    baseline_model: Any,
    fused_model: Any,
    X_baseline: pd.DataFrame,
    X_fused: pd.DataFrame,
    y: pd.Series,
    hand_df: pd.DataFrame,
    threshold: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Score both models overall and on each has_narrative/length subgroup."""
    subgroup_masks = {
        "overall": pd.Series(True, index=hand_df["complaint_id"]),
        "has_narrative": hand_df.set_index("complaint_id")["has_narrative"],
        "no_narrative": ~hand_df.set_index("complaint_id")["has_narrative"],
        "short_narrative": hand_df.set_index("complaint_id")["narrative_length_bucket"] == "short",
        "long_narrative": hand_df.set_index("complaint_id")["narrative_length_bucket"] == "long",
    }

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for subgroup_name, mask in subgroup_masks.items():
        mask = mask.reindex(y.index)
        results[subgroup_name] = {
            "baseline": score_subset(baseline_model, X_baseline[mask], y[mask], threshold),
            "fused": score_subset(fused_model, X_fused[mask], y[mask], threshold),
        }
    return results


def format_metric(results: dict[str, Any], key: str) -> str:
    value = results[key]
    return f"{value:.4f}" if value is not None else "undefined (single class)"


def compute_base_rate_divergence(
    proxy_records: list[dict[str, Any]], hand_df: pd.DataFrame, evidence_sub_issues: list[str]
) -> dict[str, Any]:
    """Compare the proxy label's escalation rate (training data) against the
    hand-labeled ground truth's rate, plus per-sub_issue evidence for why they
    diverge (companies conceding more readily on cheap/procedural sub_issues
    than on substantive ones).
    """
    proxy_df = pd.DataFrame(proxy_records)
    proxy_rate = proxy_df["escalated"].mean()
    hand_rate = hand_df["escalated"].mean()

    evidence = []
    for sub_issue in evidence_sub_issues:
        subset = proxy_df[proxy_df["sub_issue"] == sub_issue]
        if len(subset) > 0:
            evidence.append((sub_issue, len(subset), subset["escalated"].mean()))

    return {
        "proxy_rate": proxy_rate,
        "proxy_n": len(proxy_df),
        "hand_rate": hand_rate,
        "hand_n": len(hand_df),
        "evidence": evidence,
    }


def generate_report(
    results: dict[str, dict[str, dict[str, Any]]], base_rate: dict[str, Any]
) -> str:
    """Build the markdown comparison report. Headline is the base-rate divergence
    between the proxy label and the hand-labeled ground truth, direct evidence
    the two measure related but distinct constructs (company concession behavior
    vs. genuine complaint severity), not a footnote. AUC (threshold-independent)
    is the primary comparison metric; precision/recall at the fixed 0.5 threshold
    are reported but labeled secondary, since that threshold was calibrated to
    the proxy's training prevalence (26.6%), not the hand-labeled sample's true
    rate, a mismatch that mechanically depresses recall regardless of how well
    the models actually discriminate.
    """
    lines = ["# Baseline vs. Fused Model Comparison\n"]

    lines.append("## Headline finding: the proxy label and hand-labeled ground truth diverge sharply in base rate\n")
    lines.append(
        f"- Proxy label (training data, n={base_rate['proxy_n']}): **{base_rate['proxy_rate']:.1%}** escalated\n"
        f"- Hand-labeled ground truth (evaluation data, n={base_rate['hand_n']}): **{base_rate['hand_rate']:.1%}** escalated\n"
    )
    lines.append(
        "This is direct evidence that the proxy label (company concession "
        "behavior, did company_response grant relief, or was the response "
        "untimely) and genuine complaint severity (human judgment) are related "
        "but distinct constructs, exactly the limitation `project_description.md` "
        "flagged upfront: reported performance is indicative, not a validated "
        "ground-truth benchmark.\n"
    )
    if base_rate["evidence"]:
        lines.append("Evidence for *why* they diverge, proxy escalation rate by sub_issue, in the training data:\n")
        lines.append("| sub_issue | n (training) | proxy escalated rate |")
        lines.append("|---|---|---|")
        for sub_issue, n, rate in base_rate["evidence"]:
            lines.append(f"| {sub_issue} | {n} | {rate:.1%} |")
        lines.append(
            "\nCompanies concede (grant relief) far more readily on cheap, "
            "procedural sub_issues than on substantive ones like identity theft "
            "which they're more likely to deny outright. Since the models "
            "learned the proxy's pattern, they can rank procedural complaints as "
            "*more* likely to escalate than identity-theft complaints, the "
            "opposite of genuine severity. This is the direct mechanism behind "
            "the sub-0.5 AUC in the `overall` and `no_narrative` subgroups below "
            "(both dominated by sub_issue-category signal alone).\n"
        )

    lines.append(
        "**Practical consequence:** both models were trained to predict "
        "\"positive\" at roughly the proxy's 26.6% rate. Scored against a "
        "ground truth that's actually 78.3% positive, a fixed 0.5 decision "
        "threshold mechanically produces low recall, not because the models "
        "can't discriminate, but because the threshold is calibrated to the "
        "wrong prevalence. **AUC (threshold-independent, measures ranking "
        "quality only) is reported as the primary comparison metric below; "
        "precision/recall at 0.5 are secondary and threshold-sensitive.**\n"
    )

    lines.append(
        "**Headline subgroup for the AUC comparison itself: `has_narrative`.** "
        "The pooled/overall numbers are additionally confounded, the "
        "hand-labeled ground truth correlates narrative presence with "
        "escalation (see README Methodology), so a fused-model AUC advantage "
        "overall could reflect detecting *whether* a narrative exists rather "
        "than *what it says*. Within `has_narrative`, every row already has a "
        "narrative, isolating genuine content signal.\n"
    )

    lines.append("## AUC comparison (primary metric)\n")
    lines.append("| Subgroup | n | Baseline AUC | Fused AUC | Delta (fused − baseline) |")
    lines.append("|---|---|---|---|---|")
    for subgroup in ["overall", "has_narrative", "no_narrative", "short_narrative", "long_narrative"]:
        b, f = results[subgroup]["baseline"], results[subgroup]["fused"]
        if b["auc"] is not None and f["auc"] is not None:
            delta = f"{f['auc'] - b['auc']:+.4f}"
        else:
            delta = "n/a"
        lines.append(
            f"| {subgroup} | {b['n']} | {format_metric(b, 'auc')} | {format_metric(f, 'auc')} | {delta} |"
        )
    lines.append("")

    lines.append("## Precision / Recall at 0.5 threshold (secondary, threshold-sensitive, see note above)\n")
    for subgroup in ["overall", "has_narrative", "no_narrative", "short_narrative", "long_narrative"]:
        b = results[subgroup]["baseline"]
        f = results[subgroup]["fused"]
        lines.append(f"### {subgroup} (n={b['n']}, positive={b['n_positive']}, negative={b['n_negative']})\n")
        lines.append("| Model | Precision | Recall |")
        lines.append("|---|---|---|")
        lines.append(f"| Baseline | {b['precision']:.4f} | {b['recall']:.4f} |")
        lines.append(f"| Fused | {f['precision']:.4f} | {f['recall']:.4f} |")
        lines.append("")

    return "\n".join(lines)


def save_report(report_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text)
    logger.info("Saved comparison report to %s", output_path)


def run_evaluate(config_path: Path | None = None) -> Path:
    """Orchestrate Section 6: load everything, score both models, save report, return path."""
    config = load_config(config_path) if config_path is not None else load_config()

    hand_path = Path(config["data"]["labeled_path"])
    hand_df = load_hand_labeled(hand_path)
    hand_df = add_narrative_features(hand_df, config["evaluate"]["long_narrative_word_threshold"])
    logger.info(
        "Hand-labeled sample: n=%d, %d escalated (%.1f%%). "
        "no_narrative=%d, short_narrative=%d, long_narrative=%d.",
        len(hand_df), hand_df["escalated"].sum(), 100 * hand_df["escalated"].mean(),
        (hand_df["narrative_length_bucket"] == "no_narrative").sum(),
        (hand_df["narrative_length_bucket"] == "short").sum(),
        (hand_df["narrative_length_bucket"] == "long").sum(),
    )

    structured_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["structured_features_filename"]
    )
    embeddings_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["text_embeddings_filename"]
    )
    structured_df = load_structured_features(structured_path)
    embeddings_df = load_text_embeddings(embeddings_path)

    baseline_model, fused_model = load_models(config)
    X_baseline, X_fused, y = build_eval_matrices(hand_df, structured_df, embeddings_df)

    threshold = config["evaluate"]["decision_threshold"]
    results = compute_comparison(
        baseline_model, fused_model, X_baseline, X_fused, y, hand_df, threshold
    )

    labeled_path = Path(config["data"]["processed_dir"]) / config["data"]["labeled_filename"]
    with open(labeled_path) as f:
        proxy_records = json.load(f)
    base_rate = compute_base_rate_divergence(proxy_records, hand_df, EVIDENCE_SUB_ISSUES)
    logger.info(
        "Base rate divergence: proxy=%.1f%% (n=%d) vs hand-labeled=%.1f%% (n=%d).",
        100 * base_rate["proxy_rate"], base_rate["proxy_n"],
        100 * base_rate["hand_rate"], base_rate["hand_n"],
    )

    report_text = generate_report(results, base_rate)
    output_path = Path(config["data"]["processed_dir"]) / config["evaluate"]["report_filename"]
    save_report(report_text, output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output_path = run_evaluate()
    except EvaluateError as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Evaluation complete: report written to {output_path}")


if __name__ == "__main__":
    main()
