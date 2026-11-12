from unittest.mock import MagicMock

import pytest
import requests

from src import ingest
from src.ingest import IngestError


def _hit(cid, ts=1000, extra=None):
    src = {"complaint_id": cid, "company": "X", "timely": "Yes"}
    src.update(extra or {})
    return {"_source": src, "sort": [ts + int(cid), cid]}


def _resp(hits):
    return {"hits": {"hits": hits}}


def _ok(payload):
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


FETCH_ARGS = dict(
    base_url="http://api", product="Debt collection", page_size=2, search_after=None,
    sort="created_date_asc", date_received_min="2023-01-01", date_received_max="2023-03-31",
)


# ---- fetch_page: retries, backoff, params ---------------------------------

def test_fetch_page_success_sends_expected_params(monkeypatch):
    get = MagicMock(return_value=_ok(_resp([])))
    monkeypatch.setattr(ingest.requests, "get", get)
    ingest.fetch_page(**FETCH_ARGS)
    params = get.call_args.kwargs["params"]
    assert params["product"] == "Debt collection"
    assert params["sort"] == "created_date_asc"
    assert params["size"] == 2
    assert "search_after" not in params  # first page


def test_fetch_page_includes_search_after_cursor(monkeypatch):
    get = MagicMock(return_value=_ok(_resp([])))
    monkeypatch.setattr(ingest.requests, "get", get)
    ingest.fetch_page(**{**FETCH_ARGS, "search_after": "123_456"})
    assert get.call_args.kwargs["params"]["search_after"] == "123_456"
    assert "frm" not in get.call_args.kwargs["params"]


def test_fetch_page_retries_with_exponential_backoff_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)
    get = MagicMock(side_effect=[requests.exceptions.Timeout("t"),
                                 requests.exceptions.ConnectionError("c"),
                                 _ok(_resp([]))])
    monkeypatch.setattr(ingest.requests, "get", get)
    out = ingest.fetch_page(**FETCH_ARGS, max_retries=3, retry_backoff_seconds=5)
    assert out == _resp([])
    assert get.call_count == 3
    assert sleeps == [5, 10]


def test_fetch_page_raises_ingest_error_after_exhausting_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)
    get = MagicMock(side_effect=requests.exceptions.Timeout("boom"))
    monkeypatch.setattr(ingest.requests, "get", get)
    with pytest.raises(IngestError, match="failed after 3 attempts"):
        ingest.fetch_page(**FETCH_ARGS, max_retries=3, retry_backoff_seconds=1)
    assert get.call_count == 3
    assert len(sleeps) == 2  # no sleep after the final attempt


def test_fetch_page_http_error_is_retried_and_reported(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    bad = MagicMock()
    bad.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
    monkeypatch.setattr(ingest.requests, "get", MagicMock(return_value=bad))
    with pytest.raises(IngestError, match="503"):
        ingest.fetch_page(**FETCH_ARGS, max_retries=2, retry_backoff_seconds=0)


# ---- validate_response ----------------------------------------------------

@pytest.mark.parametrize("payload,msg", [
    ({}, "empty response"),
    (None, "empty response"),
    ({"foo": 1}, "hits.hits"),
    ({"hits": []}, "hits.hits"),
    ({"hits": {"total": 1}}, "hits.hits"),
    ({"hits": {"hits": "nope"}}, "not a list"),
])
def test_validate_response_rejects_malformed(payload, msg):
    with pytest.raises(IngestError, match=msg):
        ingest.validate_response(payload)


def test_validate_response_accepts_empty_hits_list():
    ingest.validate_response(_resp([]))  # no exception


# ---- normalize_record -----------------------------------------------------

def test_normalize_record_keeps_only_required_and_fills_missing_with_none():
    hit = {"_source": {"a": 1, "b": 2, "extra": 3}}
    assert ingest.normalize_record(hit, ["a", "c"]) == {"a": 1, "c": None}


def test_normalize_record_handles_missing_source():
    assert ingest.normalize_record({}, ["a"]) == {"a": None}


# ---- fetch_stratum: cursor pagination --------------------------------------

def _stratum_kwargs(cap):
    return dict(
        base_url="http://api", product="p", page_size=2, sort="s",
        date_received_min="a", date_received_max="b", cap=cap,
        required_fields=["complaint_id"], timeout_seconds=1, max_retries=1,
        retry_backoff_seconds=0,
    )


def test_fetch_stratum_follows_search_after_cursor_until_short_page(monkeypatch):
    pages = [_resp([_hit("1"), _hit("2")]), _resp([_hit("3")])]
    calls = []

    def fake_fetch_page(base_url, product, page_size, search_after, *a, **k):
        calls.append(search_after)
        return pages[len(calls) - 1]

    monkeypatch.setattr(ingest, "fetch_page", fake_fetch_page)
    recs = ingest.fetch_stratum(**_stratum_kwargs(cap=100))
    assert [r["complaint_id"] for r in recs] == ["1", "2", "3"]
    assert calls[0] is None
    assert calls[1] == "1002_2"  # f"{sort[0]}_{sort[1]}" of the previous page's last hit


def test_fetch_stratum_respects_cap_mid_page(monkeypatch):
    monkeypatch.setattr(ingest, "fetch_page",
                        lambda *a, **k: _resp([_hit("1"), _hit("2")]))
    recs = ingest.fetch_stratum(**_stratum_kwargs(cap=1))
    assert len(recs) == 1


def test_fetch_stratum_stops_on_empty_hits(monkeypatch):
    monkeypatch.setattr(ingest, "fetch_page", lambda *a, **k: _resp([]))
    assert ingest.fetch_stratum(**_stratum_kwargs(cap=10)) == []


def test_fetch_stratum_missing_sort_cursor_raises(monkeypatch):
    full_page = _resp([{"_source": {"complaint_id": "1"}}, {"_source": {"complaint_id": "2"}}])
    monkeypatch.setattr(ingest, "fetch_page", lambda *a, **k: full_page)
    with pytest.raises(IngestError, match="sort"):
        ingest.fetch_stratum(**_stratum_kwargs(cap=10))


def test_fetch_stratum_malformed_page_raises(monkeypatch):
    monkeypatch.setattr(ingest, "fetch_page", lambda *a, **k: {"unexpected": 1})
    with pytest.raises(IngestError):
        ingest.fetch_stratum(**_stratum_kwargs(cap=10))


# ---- fetch_all_complaints / save / main ------------------------------------

def test_fetch_all_complaints_concatenates_strata(monkeypatch):
    seen = []

    def fake_stratum(base_url, product, page_size, sort, start, end, cap, *a):
        seen.append((start, end, cap))
        return [{"complaint_id": start}]

    monkeypatch.setattr(ingest, "fetch_stratum", fake_stratum)
    cfg = {"cfpb": {
        "base_url": "u", "product": "p", "page_size": 10, "required_fields": ["complaint_id"],
        "sort": "s", "request_timeout_seconds": 1, "max_retries": 1, "retry_backoff_seconds": 0,
        "sampling": {"records_per_stratum": 5, "strata": [
            {"start": "2023-01-01", "end": "2023-03-31"},
            {"start": "2023-04-01", "end": "2023-06-30"}]},
    }}
    out = ingest.fetch_all_complaints(cfg)
    assert [r["complaint_id"] for r in out] == ["2023-01-01", "2023-04-01"]
    assert seen[0][2] == 5


def test_save_raw_data_round_trip(tmp_path):
    import json
    path = tmp_path / "nested" / "raw.json"
    ingest.save_raw_data([{"complaint_id": "1"}], path)
    assert json.loads(path.read_text()) == [{"complaint_id": "1"}]


def test_main_exits_1_with_clear_message_on_ingest_error(monkeypatch, capsys):
    def boom(*a, **k):
        raise IngestError("API down")

    monkeypatch.setattr(ingest, "run_ingest", boom)
    with pytest.raises(SystemExit) as e:
        ingest.main()
    assert e.value.code == 1
    assert "Ingest failed: API down" in capsys.readouterr().err
