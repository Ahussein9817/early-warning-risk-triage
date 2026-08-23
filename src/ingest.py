"""Section 1, Ingest.

Pulls debt-collection complaints from the public CFPB Consumer Complaint
Database API, validates the response, normalizes records to the required
fields, and caches the result to disk as raw data for downstream sections.

Acceptance criteria this module is responsible for (see BUILD_INSTRUCTIONS.md):
- Given the debt-collection CFPB pull, ingestion produces a raw dataset with
  all required fields and a logged record count.
- Given an empty or malformed API response, ingestion exits with a clear
  error message rather than crashing with a raw traceback.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

from src.config import load_config

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Raised when the CFPB API returns an empty/malformed response, or a
    request fails after all retries are exhausted."""


def fetch_page(
    base_url: str,
    product: str,
    page_size: int,
    search_after: str | None,
    sort: str,
    date_received_min: str,
    date_received_max: str,
    timeout_seconds: int = 30,
    max_retries: int = 3,
    retry_backoff_seconds: float = 5,
) -> dict[str, Any]:
    """Fetch one page of results from the CFPB API, retrying transient
    network failures (timeouts, connection errors) with exponential backoff.
    Raises IngestError with a clear message if all retries are exhausted.

    `sort`, `date_received_min`, and `date_received_max` are required (not
    optional) because the API's default order is not a stable global order:
    without an explicit sort and a fixed date range, re-running ingestion
    later is not guaranteed to return the same set of records, which breaks
    reproducibility.

    Pagination uses `search_after` cursoring (`{timestamp_ms}_{complaint_id}`,
    taken from the previous page's last hit's "sort" field), not an frm/offset
    parameter, the CFPB API's `frm` parameter is a known, unresolved bug
    (https://github.com/cfpb/cfpb.github.io/issues/292): it is silently
    ignored, so requests with different `frm` values return identical results.
    `search_after=None` fetches the first page.
    """
    params = {
        "product": product,
        "size": page_size,
        "sort": sort,
        "date_received_min": date_received_min,
        "date_received_max": date_received_max,
    }
    if search_after is not None:
        params["search_after"] = search_after
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(base_url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                wait = retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "CFPB API request failed (attempt %d/%d): %s. Retrying in %.0fs.",
                    attempt, max_retries, e, wait,
                )
                time.sleep(wait)

    raise IngestError(
        f"CFPB API request failed after {max_retries} attempts "
        f"(date range {date_received_min}..{date_received_max}): {last_error}"
    )


def validate_response(response_json: dict[str, Any]) -> None:
    """Raise IngestError with a clear message if `response_json` is empty or
    missing the expected structure (e.g. no "hits" key).
    """
    if not response_json:
        raise IngestError("CFPB API returned an empty response body.")

    hits_wrapper = response_json.get("hits")
    if not isinstance(hits_wrapper, dict) or "hits" not in hits_wrapper:
        raise IngestError(
            "CFPB API response is missing the expected 'hits.hits' structure. "
            f"Top-level keys received: {list(response_json.keys())}"
        )

    if not isinstance(hits_wrapper["hits"], list):
        raise IngestError(
            "CFPB API response 'hits.hits' is not a list as expected "
            f"(got {type(hits_wrapper['hits']).__name__})."
        )


def normalize_record(hit: dict[str, Any], required_fields: list[str]) -> dict[str, Any]:
    """Extract `required_fields` from one raw CFPB API hit into a flat dict."""
    source = hit.get("_source", {})
    return {field: source.get(field) for field in required_fields}


def fetch_stratum(
    base_url: str,
    product: str,
    page_size: int,
    sort: str,
    date_received_min: str,
    date_received_max: str,
    cap: int,
    required_fields: list[str],
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[dict[str, Any]]:
    """Paginate through one date-bounded stratum until `cap` records are
    collected or the stratum is exhausted, returning normalized records.
    """
    records: list[dict[str, Any]] = []
    search_after: str | None = None

    while len(records) < cap:
        response_json = fetch_page(
            base_url, product, page_size, search_after, sort,
            date_received_min, date_received_max,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        validate_response(response_json)

        hits = response_json["hits"]["hits"]
        if not hits:
            break

        for hit in hits:
            records.append(normalize_record(hit, required_fields))
            if len(records) >= cap:
                break

        if len(hits) < page_size:
            break

        last_sort = hits[-1].get("sort")
        if not last_sort or len(last_sort) < 2:
            raise IngestError(
                "CFPB API response is missing the 'sort' cursor needed to "
                "fetch the next page (expected on every hit)."
            )
        search_after = f"{last_sort[0]}_{last_sort[1]}"

    return records


def fetch_all_complaints(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch up to `records_per_stratum` complaints from each configured
    time stratum, returning the concatenated, normalized record list.

    Stratified (rather than one contiguous pull) so the dataset spans the
    full configured date range instead of clustering in the earliest days
    of it, see config.yaml `cfpb.sampling` for the stratum boundaries and
    rationale.
    """
    cfpb_cfg = config["cfpb"]
    base_url = cfpb_cfg["base_url"]
    product = cfpb_cfg["product"]
    page_size = cfpb_cfg["page_size"]
    required_fields = cfpb_cfg["required_fields"]
    sort = cfpb_cfg["sort"]
    timeout_seconds = cfpb_cfg["request_timeout_seconds"]
    max_retries = cfpb_cfg["max_retries"]
    retry_backoff_seconds = cfpb_cfg["retry_backoff_seconds"]

    sampling_cfg = cfpb_cfg["sampling"]
    records_per_stratum = sampling_cfg["records_per_stratum"]
    strata = sampling_cfg["strata"]

    records: list[dict[str, Any]] = []
    for stratum in strata:
        stratum_records = fetch_stratum(
            base_url, product, page_size, sort,
            stratum["start"], stratum["end"], records_per_stratum,
            required_fields, timeout_seconds, max_retries, retry_backoff_seconds,
        )
        logger.info(
            "Fetched %d/%d records for stratum %s..%s.",
            len(stratum_records), records_per_stratum, stratum["start"], stratum["end"],
        )
        records.extend(stratum_records)

    logger.info("Fetched %d complaint records from the CFPB API across %d strata.",
                len(records), len(strata))
    return records


def save_raw_data(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write `records` to `output_path` as JSON and log the record count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Saved %d records to %s", len(records), output_path)


def run_ingest(config_path: Path | None = None) -> Path:
    """Orchestrate the ingest section: load config, pull complaints, cache
    raw data, return the path written.
    """
    config = load_config(config_path) if config_path is not None else load_config()
    records = fetch_all_complaints(config)
    output_path = Path(config["data"]["raw_dir"]) / config["data"]["raw_filename"]
    save_raw_data(records, output_path)
    return output_path


def main() -> None:
    """CLI entry point: run ingest, print a clear message and exit(1) on
    IngestError instead of a raw traceback.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output_path = run_ingest()
    except IngestError as e:
        print(f"Ingest failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Ingest complete: raw data written to {output_path}")


if __name__ == "__main__":
    main()
