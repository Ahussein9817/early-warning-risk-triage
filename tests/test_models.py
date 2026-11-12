import numpy as np
import pandas as pd
import pytest

from src import models
from src.models import ModelsError

CFG = {"seed": 42, "split": {"test_size": 0.25, "stratify_on": "escalated"},
       "models": {"logistic_regression": {"max_iter": 200, "class_weight": "balanced", "C": 0.1}}}


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    n = 40
    ids = [str(i) for i in range(n)]
    y = np.array([1, 0, 0, 0] * 10)
    s = pd.DataFrame({"complaint_id": ids, "escalated": y,
                      "f1": rng.normal(size=n) + y, "f2": rng.normal(size=n)})
    e = pd.DataFrame({"complaint_id": ids, "emb_0": rng.normal(size=n), "emb_1": rng.normal(size=n)})
    return s, e


def test_split_is_disjoint_complete_stratified_and_deterministic(frames):
    s, _ = frames
    tr, te = models.make_train_test_split(s, CFG)
    assert set(tr).isdisjoint(te)
    assert set(tr) | set(te) == set(s["complaint_id"])
    assert len(te) == 10
    pos_te = s.set_index("complaint_id").loc[te, "escalated"].sum()
    assert pos_te in (2, 3)  # 25% base rate of 10 test rows, stratified
    assert models.make_train_test_split(s, CFG) == (tr, te)


def test_baseline_and_fused_use_identical_rows_in_identical_order(frames):
    s, e = frames
    tr, te = models.make_train_test_split(s, CFG)
    Xb_tr, Xb_te, yb_tr, yb_te = models.build_baseline_matrices(s, tr, te)
    Xf_tr, Xf_te, yf_tr, yf_te = models.build_fused_matrices(s, e, tr, te)
    assert list(Xb_tr.index) == list(Xf_tr.index) == tr
    assert list(Xb_te.index) == list(Xf_te.index) == te
    assert (yb_te == yf_te).all()
    assert list(Xb_tr.columns) == ["f1", "f2"]
    assert list(Xf_tr.columns) == ["f1", "f2", "emb_0", "emb_1"]  # embeddings appended


def test_matrices_exclude_id_and_label(frames):
    s, e = frames
    tr, te = models.make_train_test_split(s, CFG)
    for X in (models.build_baseline_matrices(s, tr, te)[0], models.build_fused_matrices(s, e, tr, te)[0]):
        assert "escalated" not in X.columns and "complaint_id" not in X.columns


def test_fused_raises_when_embedding_row_missing(frames):
    s, e = frames
    tr, te = models.make_train_test_split(s, CFG)
    with pytest.raises(ModelsError, match="no matching text embedding"):
        models.build_fused_matrices(s, e.iloc[:-3], tr, te)


def test_fused_raises_on_duplicate_embedding_ids(frames):
    s, e = frames
    tr, te = models.make_train_test_split(s, CFG)
    dup = pd.concat([e, e.iloc[[0]]])
    with pytest.raises(pd.errors.MergeError):
        models.build_fused_matrices(s, dup, tr, te)


def test_train_model_uses_config_hyperparameters_and_predicts(frames):
    s, _ = frames
    tr, te = models.make_train_test_split(s, CFG)
    X_tr, X_te, y_tr, _ = models.build_baseline_matrices(s, tr, te)
    m = models.train_model(X_tr, y_tr, CFG)
    assert m.C == 0.1 and m.class_weight == "balanced"
    assert m.coef_.shape == (1, 2)
    assert m.predict_proba(X_te).shape == (len(te), 2)


def test_save_and_load_model_round_trip(frames, tmp_path):
    import joblib
    s, _ = frames
    tr, te = models.make_train_test_split(s, CFG)
    X_tr, X_te, y_tr, _ = models.build_baseline_matrices(s, tr, te)
    m = models.train_model(X_tr, y_tr, CFG)
    p = tmp_path / "m" / "model.joblib"
    models.save_model(m, p)
    np.testing.assert_allclose(joblib.load(p).predict_proba(X_te), m.predict_proba(X_te))


def test_loaders_raise_on_missing_or_empty(tmp_path):
    with pytest.raises(ModelsError, match="not found"):
        models.load_structured_features(tmp_path / "a.csv")
    with pytest.raises(ModelsError, match="not found"):
        models.load_text_embeddings(tmp_path / "b.csv")
    p = tmp_path / "e.csv"
    p.write_text("complaint_id,escalated\n")
    with pytest.raises(ModelsError, match="empty"):
        models.load_structured_features(p)
