"""Independently regenerate the hand-labeled evaluation from COMMITTED artifacts only.

Inputs (all in the repo): data/labeled/hand_labeled.csv (106 rows, raw fields +
narrative + human label), models/{baseline,fused}_model.joblib, and the MiniLM
model from Hugging Face (downloaded). No CFPB data, no predictions file.

The feature vectors are rebuilt from each model's own `feature_names_in_`, using
the same one-hot naming as src/features.py (and the dashboard). Embeddings are
re-generated now, so tiny numeric differences vs the original run are possible if
library versions differ; the comparison below shows exactly how close the results are.

    python scripts/reproduce_eval.py            # writes results/reproduced_hand_label_eval.md

What this does NOT reproduce: the 5,600-row proxy-label numbers (the large-scale
complement) because the raw/processed data and train/test split rows are not committed.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.embeddings import generate_embeddings  # noqa: E402
from src.evaluate import add_narrative_features, compute_comparison, load_hand_labeled  # noqa: E402

# Values printed in README "Evaluation" (README-reported).
README = {
    "overall": (106, 0.4531, 0.4280, -0.0251, (-0.0539, -0.0003)),
    "has_narrative": (51, 0.6701, 0.6875, 0.0174, (-0.1200, 0.1600)),
    "no_narrative": (55, 0.4257, 0.4229, -0.0029, (-0.0117, 0.0000)),
    "short_narrative": (33, 0.7056, 0.7222, 0.0167, (-0.1935, 0.2009)),
}


def build_baseline_matrix(raw: pd.DataFrame, feature_names, categorical_fields) -> pd.DataFrame:
    """One-hot rows matching `feature_names` exactly (unknown categories -> all zero,
    companies outside the trained top-N -> 'Other')."""
    names = list(feature_names)
    X = pd.DataFrame(0.0, index=raw.index, columns=names)
    for field in categorical_fields:
        for idx, val in raw[field].items():
            col = f"{field}_{val}"
            if pd.notna(val) and col in X.columns:
                X.at[idx, col] = 1.0
    for idx, comp in raw["company"].items():
        col = f"company_bucketed_{comp}"
        if col not in X.columns:
            col = "company_bucketed_Other"
        if col in X.columns:
            X.at[idx, col] = 1.0
    return X


def build_fused_matrix(baseline_X: pd.DataFrame, embeddings: np.ndarray, fused_names) -> pd.DataFrame:
    emb = pd.DataFrame(embeddings, index=baseline_X.index,
                       columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    return pd.concat([baseline_X, emb], axis=1).reindex(columns=list(fused_names))


def main() -> None:
    cfg = load_config()
    hand = load_hand_labeled(ROOT / cfg["data"]["labeled_path"])
    base = joblib.load(ROOT / cfg["models"]["dir"] / cfg["models"]["baseline_filename"])
    fused = joblib.load(ROOT / cfg["models"]["dir"] / cfg["models"]["fused_filename"])
    hand = add_narrative_features(hand, cfg["evaluate"]["long_narrative_word_threshold"])
    hand = hand.reset_index(drop=True)

    Xb = build_baseline_matrix(hand, base.feature_names_in_, cfg["features"]["categorical_fields"])
    emb = generate_embeddings(hand.to_dict("records"), cfg["embedding"]["model_name"],
                              cfg["embedding"]["dimension"])
    Xf = build_fused_matrix(Xb, emb, fused.feature_names_in_)
    assert not Xf.isnull().any().any(), "fused matrix has NaNs (feature name mismatch)"

    ids = hand["complaint_id"]
    Xb.index = Xf.index = ids
    y = hand.set_index("complaint_id")["escalated"]
    res = compute_comparison(base, fused, Xb, Xf, y, hand, cfg["evaluate"]["decision_threshold"],
                             cfg["evaluate"]["n_bootstrap"], cfg["seed"])

    lines = ["# Reproduced hand-labeled evaluation (independent re-run from committed artifacts)", "",
             "Compare against README 'Evaluation'. Bootstrap: 2,000 paired resamples, seed from config.", "",
             "| Subgroup | n | metric | README | Reproduced | abs diff |", "|---|---|---|---|---|---|"]
    summary = {}
    for g, (n, b_auc, f_auc, delta, ci) in README.items():
        r = res[g]
        got = {"n": r["baseline"]["n"], "baseline_auc": r["baseline"]["auc"], "fused_auc": r["fused"]["auc"],
               "delta": r["fused"]["auc"] - r["baseline"]["auc"], "ci": r["bootstrap"]["delta_ci"]}
        summary[g] = got
        for name, exp, obs in (("n", n, got["n"]), ("baseline AUC", b_auc, got["baseline_auc"]),
                               ("fused AUC", f_auc, got["fused_auc"]), ("delta", delta, got["delta"]),
                               ("delta CI low", ci[0], got["ci"][0]), ("delta CI high", ci[1], got["ci"][1])):
            lines.append(f"| {g} | {n} | {name} | {exp} | {obs:.4f} | {abs(exp - obs):.4f} |")
    lines += ["", "long_narrative: single-class in the hand-labeled data "
              f"(n={res['long_narrative']['baseline']['n']}, positives={res['long_narrative']['baseline']['n_positive']}); "
              "AUC undefined, as in the README."]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "reproduced_hand_label_eval.md").write_text("\n".join(lines) + "\n")
    (out / "reproduced_hand_label_eval.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
