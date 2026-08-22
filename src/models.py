"""Section 5 — Baseline and Fused models.

Trains two logistic regression classifiers — baseline (structured features
only, Section 3) and fused (structured + text embeddings, Sections 3+4) —
on the identical train/test split, so any performance difference between
them is attributable to the text signal and nothing else, not a different
split or different rows.

The AUC logged here is an internal sanity check on the proxy-labeled
train/test split only — it is NOT the project's official result. The
official baseline-vs-fused comparison (Section 6) scores both models
against the separate, hand-labeled evaluation sample.

Acceptance criteria this module supports (see BUILD_INSTRUCTIONS.md):
- Given the same input data and seed, two full pipeline runs produce
  identical train/test splits and identical (or floating-point-tolerance-
  equal) metrics.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import load_config

logger = logging.getLogger(__name__)


class ModelsError(Exception):
    """Raised when structured features or text embeddings are missing, empty, or malformed."""


def load_structured_features(path: Path) -> pd.DataFrame:
    """Load the structured feature matrix, raising ModelsError if missing/empty."""
    if not path.exists():
        raise ModelsError(
            f"Structured features file not found at {path}. Run `python -m src.features` first."
        )
    df = pd.read_csv(path, dtype={"complaint_id": str})
    if df.empty:
        raise ModelsError(f"Structured features file at {path} is empty.")
    return df


def load_text_embeddings(path: Path) -> pd.DataFrame:
    """Load the text embedding matrix, raising ModelsError if missing/empty."""
    if not path.exists():
        raise ModelsError(
            f"Text embeddings file not found at {path}. Run `python -m src.embeddings` first."
        )
    df = pd.read_csv(path, dtype={"complaint_id": str})
    if df.empty:
        raise ModelsError(f"Text embeddings file at {path} is empty.")
    return df


def make_train_test_split(
    structured_df: pd.DataFrame, config: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Split complaint_ids into train/test once, stratified on the label,
    so both baseline and fused models train/test on identical rows.
    """
    split_cfg = config["split"]
    seed = config["seed"]
    ids = structured_df["complaint_id"]
    labels = structured_df[split_cfg["stratify_on"]]

    train_ids, test_ids = train_test_split(
        ids,
        test_size=split_cfg["test_size"],
        random_state=seed,
        stratify=labels,
    )
    return list(train_ids), list(test_ids)


def build_baseline_matrices(
    structured_df: pd.DataFrame, train_ids: list[str], test_ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build X/y train/test matrices from structured features only."""
    feature_cols = [c for c in structured_df.columns if c not in ("complaint_id", "escalated")]
    indexed = structured_df.set_index("complaint_id")

    X_train = indexed.loc[train_ids, feature_cols]
    X_test = indexed.loc[test_ids, feature_cols]
    y_train = indexed.loc[train_ids, "escalated"]
    y_test = indexed.loc[test_ids, "escalated"]
    return X_train, X_test, y_train, y_test


def build_fused_matrices(
    structured_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build X/y train/test matrices from structured features + text embeddings,
    on the same rows/split as build_baseline_matrices.
    """
    merged = structured_df.merge(
        embeddings_df, on="complaint_id", how="left", validate="one_to_one"
    )
    if merged["emb_0"].isnull().any():
        raise ModelsError(
            "Some complaint_ids in structured features have no matching text "
            "embedding row — expected every row to have one (possibly zero-vector)."
        )

    feature_cols = [c for c in merged.columns if c not in ("complaint_id", "escalated")]
    indexed = merged.set_index("complaint_id")

    X_train = indexed.loc[train_ids, feature_cols]
    X_test = indexed.loc[test_ids, feature_cols]
    y_train = indexed.loc[train_ids, "escalated"]
    y_test = indexed.loc[test_ids, "escalated"]
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict[str, Any]
) -> LogisticRegression:
    """Fit a logistic regression classifier per config.yaml `models.logistic_regression`."""
    lr_cfg = config["models"]["logistic_regression"]
    model = LogisticRegression(
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        C=lr_cfg["C"],
        random_state=config["seed"],
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: LogisticRegression, output_path: Path) -> None:
    """Persist `model` to `output_path` with joblib and log it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Saved model to %s", output_path)


def run_models(config_path: Path | None = None) -> tuple[Path, Path]:
    """Orchestrate Section 5: load features, split once, train both models, save, return paths."""
    config = load_config(config_path) if config_path is not None else load_config()

    structured_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["structured_features_filename"]
    )
    embeddings_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["text_embeddings_filename"]
    )
    structured_df = load_structured_features(structured_path)
    embeddings_df = load_text_embeddings(embeddings_path)

    train_ids, test_ids = make_train_test_split(structured_df, config)
    logger.info(
        "Train/test split: %d train / %d test (test_size=%.2f, seed=%d, stratified on escalated).",
        len(train_ids), len(test_ids), config["split"]["test_size"], config["seed"],
    )

    Xb_train, Xb_test, yb_train, yb_test = build_baseline_matrices(
        structured_df, train_ids, test_ids
    )
    baseline_model = train_model(Xb_train, yb_train, config)
    baseline_auc = roc_auc_score(yb_test, baseline_model.predict_proba(Xb_test)[:, 1])
    logger.info(
        "Baseline model: %d train rows, %d features. Internal test-split AUC=%.4f "
        "(sanity check only — the official comparison is Section 6, vs. the hand-labeled sample).",
        len(Xb_train), Xb_train.shape[1], baseline_auc,
    )

    Xf_train, Xf_test, yf_train, yf_test = build_fused_matrices(
        structured_df, embeddings_df, train_ids, test_ids
    )
    fused_model = train_model(Xf_train, yf_train, config)
    fused_auc = roc_auc_score(yf_test, fused_model.predict_proba(Xf_test)[:, 1])
    logger.info(
        "Fused model: %d train rows, %d features. Internal test-split AUC=%.4f "
        "(sanity check only — the official comparison is Section 6, vs. the hand-labeled sample).",
        len(Xf_train), Xf_train.shape[1], fused_auc,
    )

    models_dir = Path(config["models"]["dir"])
    baseline_path = models_dir / config["models"]["baseline_filename"]
    fused_path = models_dir / config["models"]["fused_filename"]
    save_model(baseline_model, baseline_path)
    save_model(fused_model, fused_path)

    return baseline_path, fused_path


def main() -> None:
    """CLI entry point: run Section 5, print a clear message and exit(1) on
    ModelsError instead of a raw traceback.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        baseline_path, fused_path = run_models()
    except ModelsError as e:
        print(f"Model training failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Model training complete: baseline={baseline_path}, fused={fused_path}")


if __name__ == "__main__":
    main()
