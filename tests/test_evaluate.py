import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src.evaluate import EvaluateError


def test_score_from_proba_basic():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    r = ev.score_from_proba(p, y, 0.5)
    assert r["n"] == 4 and r["n_positive"] == 2 and r["n_negative"] == 2
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["auc"] == pytest.approx(0.75)


@pytest.mark.parametrize("y", [np.zeros(4, dtype=int), np.ones(4, dtype=int)])
def test_score_from_proba_single_class_auc_is_none(y):
    r = ev.score_from_proba(np.array([0.1, 0.9, 0.2, 0.8]), y, 0.5)
    assert r["auc"] is None


def test_score_from_proba_no_predicted_positives_zero_not_nan():
    r = ev.score_from_proba(np.array([0.1, 0.2, 0.1, 0.3]), np.array([1, 0, 0, 1]), 0.5)
    assert r["precision"] == 0.0 and r["recall"] == 0.0


# ---- paired bootstrap ------------------------------------------------------

def _data(n=200, seed=0, fused_gain=0.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    base = rng.random(n) * 0.5 + y * 0.3
    fused = np.clip(base + rng.normal(0, 0.05, n) + fused_gain * y, 0, 1)
    return y, base, fused


def test_bootstrap_identical_models_delta_is_exactly_zero():
    y, base, _ = _data()
    r = ev.bootstrap_auc_ci(y, base, base.copy(), n_bootstrap=200, seed=1)
    assert r["delta_ci"] == (0.0, 0.0)
    assert r["delta_median"] == 0.0
    assert r["pct_delta_positive"] == 0.0
    assert r["n_valid"] == 200


def test_bootstrap_clearly_better_fused_has_positive_ci():
    y, base, fused = _data(n=400, fused_gain=0.4)
    r = ev.bootstrap_auc_ci(y, base, fused, n_bootstrap=300, seed=1)
    lo, hi = r["delta_ci"]
    assert lo > 0 and hi >= lo
    assert r["pct_delta_positive"] > 0.99


def test_bootstrap_ci_ordering_and_deterministic_with_seed():
    y, base, fused = _data(fused_gain=0.1)
    a = ev.bootstrap_auc_ci(y, base, fused, 150, seed=7)
    b = ev.bootstrap_auc_ci(y, base, fused, 150, seed=7)
    c = ev.bootstrap_auc_ci(y, base, fused, 150, seed=8)
    assert a == b
    assert a != c
    for key in ("baseline_ci", "fused_ci", "delta_ci"):
        assert a[key][0] <= a[key][1]


def test_bootstrap_uses_same_indices_for_both_models():
    # If fused == base shifted by a constant, AUC is rank-based => delta must be 0
    # on every resample only if the SAME rows are used for both models.
    y, base, _ = _data()
    r = ev.bootstrap_auc_ci(y, base, base + 0.123, 100, seed=3)
    assert r["delta_ci"] == (0.0, 0.0)


def test_bootstrap_single_class_returns_none_fields():
    y = np.zeros(10, dtype=int)
    p = np.linspace(0, 1, 10)
    r = ev.bootstrap_auc_ci(y, p, p, 50, seed=0)
    assert r["n_valid"] == 0 and r["delta_ci"] is None and r["delta_median"] is None


def test_bootstrap_skips_single_class_resamples_and_counts_them():
    y = np.array([1] + [0] * 5)  # one positive: many resamples miss it
    rng = np.random.default_rng(0)
    p = rng.random(6)
    r = ev.bootstrap_auc_ci(y, p, p, 300, seed=0)
    assert 0 < r["n_valid"] < 300
    assert r["n_bootstrap"] == 300


# ---- loaders / feature helpers --------------------------------------------

def test_load_hand_labeled_errors_and_types(tmp_path):
    with pytest.raises(EvaluateError, match="not found"):
        ev.load_hand_labeled(tmp_path / "x.csv")
    blank = tmp_path / "blank.csv"
    blank.write_text("complaint_id,escalated\n1,1\n2,\n")
    with pytest.raises(EvaluateError, match="blank"):
        ev.load_hand_labeled(blank)
    empty = tmp_path / "empty.csv"
    empty.write_text("complaint_id,escalated\n")
    with pytest.raises(EvaluateError, match="empty"):
        ev.load_hand_labeled(empty)
    ok = tmp_path / "ok.csv"
    ok.write_text("complaint_id,escalated\n001,1\n002,0\n")
    df = ev.load_hand_labeled(ok)
    assert df["escalated"].dtype.kind == "i"
    assert list(df["complaint_id"]) == ["001", "002"]  # ids stay strings (leading zeros kept)


def test_add_narrative_features_buckets():
    words = lambda n: " ".join(["w"] * n)  # noqa: E731
    df = pd.DataFrame({"complaint_what_happened": [None, "", words(10), words(180), words(181)]})
    out = ev.add_narrative_features(df, word_threshold=180)
    assert list(out["narrative_length_bucket"]) == ["no_narrative", "no_narrative", "short", "short", "long"]
    assert list(out["has_narrative"]) == [False, False, True, True, True]
    assert "has_narrative" not in df.columns  # input not mutated


def test_build_eval_matrices_alignment_and_ground_truth_source():
    structured = pd.DataFrame({"complaint_id": ["a", "b", "c"], "escalated": [0, 0, 0],
                               "f": [1.0, 2.0, 3.0]})
    emb = pd.DataFrame({"complaint_id": ["a", "b", "c"], "emb_0": [10.0, 20.0, 30.0]})
    hand = pd.DataFrame({"complaint_id": ["c", "a"], "escalated": [1, 0]})  # different order
    Xb, Xf, y = ev.build_eval_matrices(hand, structured, emb)
    assert list(Xb.index) == ["c", "a"] == list(Xf.index) == list(y.index)
    assert list(Xb["f"]) == [3.0, 1.0]
    assert list(Xf.columns) == ["f", "emb_0"]
    assert list(y) == [1, 0]  # hand label, not the proxy label (all 0 in structured)


def test_build_eval_matrices_missing_ids_raise():
    structured = pd.DataFrame({"complaint_id": ["a"], "escalated": [0], "f": [1.0]})
    emb = pd.DataFrame({"complaint_id": ["a"], "emb_0": [1.0]})
    hand = pd.DataFrame({"complaint_id": ["a", "zzz"], "escalated": [0, 1]})
    with pytest.raises(EvaluateError, match="no matching structured"):
        ev.build_eval_matrices(hand, structured, emb)


def test_compute_comparison_runs_all_subgroups_end_to_end():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    n = 60
    ids = [str(i) for i in range(n)]
    y = pd.Series(rng.integers(0, 2, n), index=pd.Index(ids, name="complaint_id"))
    Xb = pd.DataFrame({"f": rng.normal(size=n) + y.values}, index=y.index)
    Xf = Xb.assign(e=rng.normal(size=n))
    narr = [None if i % 3 == 0 else ("w " * (5 if i % 2 else 200)).strip() for i in range(n)]
    hand = ev.add_narrative_features(
        pd.DataFrame({"complaint_id": ids, "complaint_what_happened": narr}), 180)
    mb = LogisticRegression().fit(Xb, y)
    mf = LogisticRegression().fit(Xf, y)
    res = ev.compute_comparison(mb, mf, Xb, Xf, y, hand, 0.5, n_bootstrap=50, seed=1)
    assert set(res) == {"overall", "has_narrative", "no_narrative", "short_narrative", "long_narrative"}
    assert res["overall"]["baseline"]["n"] == n
    assert (res["has_narrative"]["baseline"]["n"] + res["no_narrative"]["baseline"]["n"]) == n
