import json

import pytest

from src import label
from src.label import LabelError

RELIEF = ["Closed with monetary relief", "Closed with non-monetary relief", "Closed with relief"]


@pytest.mark.parametrize("record,expected", [
    ({"company_response": "Closed with monetary relief", "timely": "Yes"}, 1),
    ({"company_response": "Closed with relief", "timely": "Yes"}, 1),
    ({"company_response": "Closed with explanation", "timely": "No"}, 1),
    ({"company_response": "Closed with explanation", "timely": "Yes"}, 0),
    ({"company_response": "In progress", "timely": "Yes"}, 0),
    ({"company_response": None, "timely": "No"}, 1),
    ({"company_response": "Closed with relief", "timely": None}, 1),
    ({"company_response": None, "timely": "Yes"}, 0),
])
def test_compute_escalated_label_rule(record, expected):
    assert label.compute_escalated_label(record, RELIEF) == expected


def test_both_inputs_missing_defaults_to_zero_and_warns(caplog):
    with caplog.at_level("WARNING"):
        out = label.compute_escalated_label({"complaint_id": "9"}, RELIEF)
    assert out == 0
    assert "complaint_id=9" in caplog.text


def test_label_ignores_narrative_presence():
    # label must stay independent of the text signal under test
    a = {"company_response": "Closed with explanation", "timely": "Yes", "has_narrative": True,
         "complaint_what_happened": "long text"}
    b = {**a, "has_narrative": False, "complaint_what_happened": None}
    assert label.compute_escalated_label(a, RELIEF) == label.compute_escalated_label(b, RELIEF)


def test_apply_labels_adds_field_without_mutating_input():
    recs = [{"company_response": "Closed with relief", "timely": "Yes"}]
    out = label.apply_labels(recs, RELIEF)
    assert out[0]["escalated"] == 1
    assert "escalated" not in recs[0]


def test_load_raw_data_errors(tmp_path):
    with pytest.raises(LabelError, match="not found"):
        label.load_raw_data(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(LabelError, match="not valid JSON"):
        label.load_raw_data(bad)
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(LabelError, match="empty"):
        label.load_raw_data(empty)
    notlist = tmp_path / "dict.json"
    notlist.write_text(json.dumps({"a": 1}))
    with pytest.raises(LabelError, match="list of records"):
        label.load_raw_data(notlist)


def test_load_raw_data_ok(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps([{"complaint_id": "1"}]))
    assert label.load_raw_data(p) == [{"complaint_id": "1"}]
