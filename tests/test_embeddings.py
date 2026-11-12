import numpy as np
import pytest

from src import embeddings
from src.embeddings import EmbeddingsError


class FakeModel:
    """Deterministic stand-in for SentenceTransformer: vector depends on text length."""
    calls = []

    def __init__(self, name):
        self.name = name

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        FakeModel.calls.append(list(texts))
        return np.array([[len(t)] * 4 for t in texts], dtype=np.float32)


@pytest.fixture(autouse=True)
def fake_st(monkeypatch):
    import sentence_transformers
    FakeModel.calls = []
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)


def _recs(texts):
    return [{"complaint_id": str(i), "complaint_what_happened": t} for i, t in enumerate(texts)]


def test_empty_and_none_narratives_get_zero_vectors_and_are_not_dropped():
    out = embeddings.generate_embeddings(_recs(["abc", "", None, "hello"]), "m", 4)
    assert out.shape == (4, 4)
    assert (out[1] == 0).all() and (out[2] == 0).all()
    assert (out[0] == 3).all() and (out[3] == 5).all()


def test_only_non_empty_narratives_are_sent_to_model_in_order():
    embeddings.generate_embeddings(_recs(["abc", "", "hello"]), "m", 4)
    assert FakeModel.calls == [["abc", "hello"]]


def test_all_empty_narratives_never_instantiates_model_and_returns_zeros():
    out = embeddings.generate_embeddings(_recs(["", None]), "m", 4)
    assert FakeModel.calls == []
    assert not out.any()


def test_build_text_embeddings_alignment_with_complaint_ids():
    recs = _recs(["aa", "", "bbbb"])
    df = embeddings.build_text_embeddings(recs, {"embedding": {"model_name": "m", "dimension": 4}})
    assert list(df["complaint_id"]) == ["0", "1", "2"]
    assert list(df.columns[:3]) == ["complaint_id", "emb_0", "emb_1"]
    assert df.shape == (3, 5)
    assert df.loc[2, "emb_0"] == 4 and df.loc[1, "emb_0"] == 0


def test_load_labeled_data_missing(tmp_path):
    with pytest.raises(EmbeddingsError, match="not found"):
        embeddings.load_labeled_data(tmp_path / "x.json")


def test_nan_narrative_from_dataframe_is_treated_as_missing():
    out = embeddings.generate_embeddings(_recs(["abc", float("nan")]), "m", 4)
    assert (out[1] == 0).all() and (out[0] == 3).all()
    assert FakeModel.calls == [["abc"]]
