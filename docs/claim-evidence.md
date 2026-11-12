# Claim-to-evidence table (early-warning-risk-triage)

Evidence types: **README-reported**, **regenerated** (from committed artifacts), **test**, **CI**, **not run**. State on branch `improve/portfolio-refinement` (head 57a86e2, CI run 43156063417 green). The training label is a **proxy** (company concession behaviour), not verified ground truth.

| Claim | Status | Evidence type | Where | Caveat |
|---|---|---|---|---|
| CFPB ingestion with retry/backoff and cursor paging | VERIFIED (code) | test (mocked HTTP) | `src/ingest.py`, `tests/test_ingest.py` | Live API not re-pulled |
| Hand-labeled evaluation table (n=106): AUCs, deltas, bootstrap CIs | VERIFIED | regenerated | `scripts/reproduce_eval.py`, `results/reproduced_hand_label_eval.md` | Matches README to 4 decimals; single rater, n=106 |
| Fusing text with structured fields shows no reliable benefit on the hand-labeled sample (`has_narrative` CI crosses zero) | VERIFIED | regenerated | same | Underpowered; README calls it inconclusive |
| Proxy-label dataset statistics (n=5,600; 26.6% proxy rate) | UNSUPPORTED (not reproducible from the repo) | README-reported | README | Raw/processed data and split rows not committed |
| Large-scale proxy-label complement (n=1,120) | UNSUPPORTED (not reproducible) | README-reported | README | Scored against the proxy, whose base rate differs from the hand labels (26.6% vs 78.3%) |
| Dashboard renders and predicts headlessly | VERIFIED (local) | test (opt-in `RUN_SLOW=1`) | `tests/test_dashboard_smoke.py` | Downloads MiniLM |
| Dockerfile builds and serves the dashboard | VERIFIED (in CI only) | CI | run 43153694669, `docker` job | Not built locally |
| Live dashboard at early-warning-risk-triage.streamlit.app | PARTIAL (unconfirmed) | README-reported; one HTTP request on 2026-11-12 | README | The request was redirected (303) to a Streamlit sign-in page, so public access could not be confirmed; the owner should check the app's sharing setting |
| 74 tests, ruff, CI; line coverage 44% | VERIFIED | CI | run 43156063417 | Four modules have no unit tests |
