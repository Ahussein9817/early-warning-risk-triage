import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import reproduce_eval as re_  # noqa: E402

NAMES = ["sub_product_A", "sub_product_B", "state_TX", "company_bucketed_Acme", "company_bucketed_Other"]


def _raw():
    return pd.DataFrame({
        "sub_product": ["A", "Z", None], "state": ["TX", "TX", None],
        "company": ["Acme", "Unknown Co", "Acme"]})


def test_baseline_matrix_one_hot_unknown_category_and_company_bucketing():
    X = re_.build_baseline_matrix(_raw(), NAMES, ["sub_product", "state"])
    assert list(X.columns) == NAMES
    assert X.loc[0].tolist() == [1, 0, 1, 1, 0]
    assert X.loc[1].tolist() == [0, 0, 1, 0, 1]   # unseen sub_product -> zeros; unseen company -> Other
    assert X.loc[2].tolist() == [0, 0, 0, 1, 0]   # null fields -> all zero


def test_fused_matrix_appends_embeddings_in_model_column_order():
    Xb = re_.build_baseline_matrix(_raw(), NAMES, ["sub_product", "state"])
    emb = np.arange(6, dtype=float).reshape(3, 2)
    names = NAMES + ["emb_0", "emb_1"]
    Xf = re_.build_fused_matrix(Xb, emb, names)
    assert list(Xf.columns) == names
    assert Xf["emb_1"].tolist() == [1.0, 3.0, 5.0]
    assert not Xf.isnull().any().any()
