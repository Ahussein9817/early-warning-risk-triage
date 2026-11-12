import pandas as pd
import pytest

from src import features
from src.features import FeaturesError


def _records():
    comps = ["A"] * 3 + ["B"] * 2 + ["C"]
    return [
        {"complaint_id": str(i), "escalated": i % 2, "company": c,
         "sub_product": "Credit card debt" if i % 2 else "Other debt",
         "issue": "i1", "sub_issue": "s1", "state": None if i == 0 else "TX",
         "company_response": "LEAK", "timely": "Yes", "product": "Debt collection"}
        for i, c in enumerate(comps)
    ]


CFG = {"features": {"categorical_fields": ["sub_product", "issue", "sub_issue", "state"],
                    "company_top_n": 2}}


def test_compute_top_companies_ordering_and_n():
    assert features.compute_top_companies(_records(), 2) == ["A", "B"]
    assert features.compute_top_companies(_records(), 1) == ["A"]


def test_bucket_company():
    assert features.bucket_company("A", ["A"]) == "A"
    assert features.bucket_company("Z", ["A"]) == "Other"


def test_build_structured_features_excludes_label_inputs_and_constant_product():
    df, top = features.build_structured_features(_records(), CFG)
    assert top == ["A", "B"]
    cols = " ".join(df.columns)
    assert "company_response" not in cols
    assert "timely" not in cols
    assert "product_" not in cols.replace("sub_product_", "")
    assert {"complaint_id", "escalated"} <= set(df.columns)


def test_company_bucketing_creates_other_column_and_one_hot_rows_sum_to_one():
    df, _ = features.build_structured_features(_records(), CFG)
    comp_cols = [c for c in df.columns if c.startswith("company_bucketed_")]
    assert set(comp_cols) == {"company_bucketed_A", "company_bucketed_B", "company_bucketed_Other"}
    assert (df[comp_cols].astype(int).sum(axis=1) == 1).all()
    assert df.loc[df["complaint_id"] == "5", "company_bucketed_Other"].item()


def test_null_state_is_all_zero_across_state_columns():
    df, _ = features.build_structured_features(_records(), CFG)
    state_cols = [c for c in df.columns if c.startswith("state_")]
    row = df.loc[df["complaint_id"] == "0", state_cols].astype(int)
    assert row.sum(axis=1).item() == 0


def test_row_count_and_ids_preserved():
    df, _ = features.build_structured_features(_records(), CFG)
    assert len(df) == 6
    assert list(df["complaint_id"]) == [str(i) for i in range(6)]


def test_save_features_round_trip(tmp_path):
    df, _ = features.build_structured_features(_records(), CFG)
    p = tmp_path / "o" / "f.csv"
    features.save_features(df, p)
    assert pd.read_csv(p).shape == df.shape


def test_load_labeled_data_errors(tmp_path):
    with pytest.raises(FeaturesError, match="not found"):
        features.load_labeled_data(tmp_path / "nope.json")
    p = tmp_path / "e.json"
    p.write_text("[]")
    with pytest.raises(FeaturesError, match="empty"):
        features.load_labeled_data(p)
