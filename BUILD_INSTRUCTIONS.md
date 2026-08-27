## Overview

Build a working pipeline for an **early-warning triage model** that tests whether fusing unstructured complaint narratives with structured case fields improves prediction of case escalation, versus structured data alone. This is a research prototype for a single-semester project, demonstrating a pattern (predict a negative outcome before it happens, using every available signal) that recurs across insurance, fintech, and healthcare, it is not built for any single company and makes no claim of production readiness.

Data source: the **CFPB Consumer Complaint Database** public API, scoped to a single product category: **Debt Collection**.


## Status (as of 2026-08-22)

All seven pipeline sections are complete and verified against real data (not just "should work", each was tested end-to-end, with several real bugs and design gaps caught and fixed along the way, documented inline below and in README.md Methodology).

- [x] **Section 1, Ingest.** 5,600 records, stratified across 14 quarters (2023-01-01 to 2026-05-21), reproducible (byte-identical across independent runs). Caught and fixed two live-API bugs (see Inputs).
- [x] **Section 2, Label.** Proxy rule `company_response` relief category OR `timely=="No"`, logged when applied. 26.6% escalated on the full dataset.
- [x] **Section 3, Structured features.** 120 one-hot columns; `company` bucketed top-20 + "Other" (see Requirements).
- [x] **Section 4, Text embeddings.** `all-MiniLM-L6-v2`, zero-imputed for no-narrative rows (verified 3,125/3,125 exact zero vectors), bit-exact reproducible.
- [x] **Section 5, Models.** Logistic regression, `C=0.1` L2 regularization (see Architecture), same train/test split verified identical for both models.
- [x] **Section 6, Evaluate.** Scored against a 106-row hand-labeled sample. Headline finding is a base-rate divergence between the proxy label and hand-judged ground truth (see Architecture and Acceptance criteria).
- [x] **Section 7, Dashboard.** Streamlit app, live MiniLM inference, verified in-browser. Deployed live to Streamlit Community Cloud (revised from HuggingFace Spaces, see `OPEN_DECISIONS.md`): https://early-warning-risk-triage.streamlit.app. Deployment needed a Python version fix (3.14, the platform default at the time, lacked prebuilt wheels for pandas/PyYAML, forcing slow source builds; redeployed pinned to Python 3.12 via Advanced settings, since `runtime.txt` is currently unreliable on this platform).

All deliverables complete: README's Problem section is written, and the dashboard is deployed live at https://early-warning-risk-triage.streamlit.app.

## Goal

Produce a numerical, falsifiable answer to: *does adding complaint-narrative text embeddings to a structured-feature classifier improve escalation prediction on the CFPB debt-collection complaint dataset, and by how much?*

The baseline-vs-fused comparison is the deliverable, not optional polish added at the end, without it, the project only claims text was used, not that it helped.

## Inputs/Outputs

**Inputs**

- CFPB API pull, `product = "Debt collection"`, sorted `created_date_asc`. **Stratified sampling** (see `cfpb.sampling` in `config.yaml`): 400 records from each of 14 fixed quarterly strata spanning 2023-01-01 through 2026-05-21, for 5,600 records total spanning multiple years rather than one contiguous block. An initial non-stratified design (a single pull from `date_received_min = "2023-01-01"` capped at 5,000) was tried first and only covered ~4 weeks (2023-01-01 to 2023-01-27) at the observed volume of ~185 debt-collection complaints/day, too thin a slice, hence the switch to stratification. The 2026-05-21 cutoff (not "today") is deliberate: very recent complaints often still show `company_response = "In progress"`, which would corrupt the label rule below, and a fixed cutoff keeps the pull reproducible regardless of when the pipeline is re-run. Required fields (verified 2026-08-21 against the live API, see note below): `product`, `sub_product`, `issue`, `sub_issue`, `company_response`, `timely`, `state`, `complaint_what_happened`, `has_narrative`, `date_received`, `company`, `complaint_id`.
  - **Correction from the original draft of this spec:** `timely_response` and `consumer_complaint_narrative` were renamed in the live API to `timely` and `complaint_what_happened` respectively. `consumer_disputed` does not exist in the live API at all, CFPB discontinued the "Consumer disputed?" field on April 24, 2017 ([CFPB API release notes](https://cfpb.github.io/api/ccdb/release-notes.html)). This directly affects the Label section's rule below; resolved 2026-08-21 by swapping in `timely` (see Architecture).
  - **Reproducibility correction:** without an explicit `sort` and a fixed `date_received_min`, the CFPB API's default result order is not a stable global order. A fresh ingest run at a later date is not guaranteed to return the same records, violating the reproducibility requirement below. `sort=created_date_asc` + a fixed `date_received_min` fixes this: the first `max_records` complaints starting from that date are the same set regardless of when the pipeline is run, since historical CFPB records aren't retroactively altered.
  - **Pagination correction:** the CFPB API's documented `frm` offset parameter is a known, unresolved bug ([cfpb/cfpb.github.io#292](https://github.com/cfpb/cfpb.github.io/issues/292)): it's silently ignored, so every "page" returned the same first 100 records regardless of offset (verified empirically: an initial ingest run produced only 100 unique complaint_ids repeated 50x, masquerading as 5,000 records). Fixed by switching to the API's `search_after` cursor pagination: each request after the first passes `search_after={last_hit_sort_timestamp}_{last_hit_complaint_id}`, taken from the previous page's final hit. Verified: 5,000/5,000 unique complaint_ids across two independent runs, byte-identical output both times.
- Hand-labeled sample: **106 complaints actually labeled**, out of a 150-row template (resolved 2026-08-21, within the 100-200 range), manually reviewed by you (not the agent), stored as `complaint_id, escalated (0/1), labeler_notes`. Used only for held-out evaluation, not training. Sampled uniformly at random (seeded) from the model **test split only** (never train, see Section 5), via `src/sample_for_labeling.py` → `data/labeled/hand_labeled_TEMPLATE.csv`. The exported template deliberately excludes `company_response`, `timely`, and the proxy `escalated` label, those are the proxy rule's own inputs/output, so seeing them while hand-labeling would reproduce the proxy's answer rather than provide an independent validation signal. Composition: 55 no-narrative, 33 short-narrative, 18 long-narrative (long_narrative is 18/18 escalated, a single-class subgroup, see Acceptance criteria).
  - **Consistency-recheck correction (2026-08-22):** an initial pass found `sub_issue="Debt is not yours"` (14 rows) escalated only 7.1% of the time vs. `"Debt was result of identity theft"` (100%) and `"Sued you without properly notifying you of lawsuit"` (100%), despite being substantively the same kind of claim. 19 rows were rechecked against this pattern; 18 were corrected from `escalated=0` to `escalated=1`. Documented as a labeling-consistency limitation in README.md Methodology, not silently corrected without a record.

**Outputs**

- Engineered structured feature matrix and text-embedding matrix (keyed by `complaint_id`)
- Two trained model artifacts: baseline (structured-only) and fused (structured + text)
- A comparison report: precision, recall, and AUC for both models on the same held-out labeled sample, with a plain-language verdict on whether fusion helped
- A Streamlit dashboard showing a complaint's text and structured fields going in, and both models' predictions and confidence scores, side by side

## Requirements

- Product category is fixed to Debt Collection, and this and all other tunable values (seed, paths, API params) live in one config file, never hardcoded elsewhere.
- Runs locally via `streamlit run`; no cloud account or paid service required to reproduce results.
- Public, keyless CFPB API only, no credentials to manage in this phase.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~80MB). **Resolved and documented** in README.md Methodology → Text embeddings and here: chosen over `all-mpnet-base-v2` (higher quality, ~4-5x slower on CPU, ~420MB) and `BAAI/bge-small-en-v1.5` (comparable size/quality but needs a query-instruction prefix convention) for fast CPU inference and a small footprint, given no guaranteed GPU and a semester timeline.
  - **Accepted limitation: MiniLM's 256-token max input length truncates long narratives.** 28.7% of narratives with text (711/2,475, word-count proxy; 25.2% confirmed by direct tokenizer measurement) are long enough to risk truncation to roughly the opening ~200 words. Accepted rather than mitigated with chunk-and-average (rejected: adds complexity, dilutes rather than clarifies signal). This is a measurement limitation, not evidence against the hypothesis, escalation-relevant language (regulatory threats, repeated-contact complaints, "unresolved after months" language) often appears later in long narratives, so the fused model's true benefit on long narratives may be understated. This is exactly why Section 6 splits the comparison by narrative length (see Acceptance criteria).
- **Company bucketing (Section 3): top-20 + "Other", resolved 2026-08-21.** `company` has 689 unique values on 5,600 records; one-hot encoding all of them would add 689 mostly-sparse columns. N=20 chosen from a full cumulative-coverage sweep (see README.md Methodology for the table): covers 52.96% of records, each included company has ≥45 complaints (workable across the train/test split), and coverage-per-column drops sharply past N≈20. Verified programmatically: an out-of-top-20 company resolves to exactly `company_bucketed_Other=1`, no double-counting, no silent mishandling.
- Streamlit dashboard is deployed to Streamlit Community Cloud (public URL) in addition to running locally via `streamlit run`; no paid service required. **Deployment target resolved** (revised 2026-08-23 from HuggingFace Spaces after HF deprecated free-tier Streamlit hosting, see `OPEN_DECISIONS.md`); actually creating the public deployment is still pending.

## Architecture

Ingest (CFPB API pull, cached raw data) → Label (derive proxy escalation label as `escalated = 1 if company_response in {"Closed with monetary relief", "Closed with non-monetary relief", "Closed with relief"} OR timely == "No", else 0`; isolated in its own module so the rule is auditable; uses only structured fields, deliberately excluding `has_narrative`/`complaint_what_happened` so the label isn't correlated with the text signal the fused model is being tested on) → Structured features (Stage 1 input; `sub_product`/`issue`/`sub_issue`/`state` one-hot, `company` bucketed top-20+"Other", see Requirements) → Text embeddings (Stage 2 addition; MiniLM, zero-imputed for no-narrative rows, truncated for long narratives, see Requirements) → Baseline model (Stage 1) and Fused model (Stage 2), logistic regression, `class_weight="balanced"`, `C=0.1` L2 regularization (chosen 2026-08-21: three `company_bucketed` features showed perfect separation and unstable coefficients at C=1.0; C=0.1 nearly halved them for <1% AUC cost, see README.md Methodology for the full sweep; applied to both models so regularization strength isn't itself a confound in the comparison), trained on the same split (verified identical row indices, not just identical split size) → Evaluate (Stage 3: both models scored against the same held-out hand-labeled sample, see Section 6 findings below) → Dashboard (Streamlit, shows a complaint moving through the pipeline live, see Section 7 acceptance criteria).

Constraint on the pipeline: complaints with no narrative (roughly 74% of CFPB complaints) are retained in baseline training data and only excluded/zero-imputed for the text-input stage, never silently dropped from the dataset.

### Section 6 findings (part of the spec's record, not just README color)

- **Headline finding: the proxy label and hand-labeled ground truth diverge sharply in base rate**, proxy (training, n=5,600): 26.6% escalated; hand-labeled (evaluation, n=106): 78.3% escalated. Traced to a concrete mechanism: companies concede far more readily on cheap procedural sub_issues (`"Notification didn't disclose it was an attempt to collect a debt"`: 54.6% proxy-escalated, n=491) than on substantive claims like identity theft (28.0%, n=751) or "debt is not yours" (24.7%, n=1,475).
- **AUC is the primary comparison metric, not pooled precision/recall at the 0.5 threshold.** Both models were trained to predict positive at ~26.6% (the proxy rate); scored against a 78.3%-positive ground truth, a fixed 0.5 threshold mechanically depresses recall regardless of actual discrimination quality. AUC is threshold-independent and unaffected by this mismatch.
- **`has_narrative` is the primary comparison subgroup, not the pooled/overall number.** The hand-labeled ground truth itself correlates narrative presence with escalation, so a pooled fused-model AUC advantage could reflect detecting *whether* a narrative exists rather than *what it says*. Within `has_narrative` (n=51), every row already has a narrative, isolating genuine content signal: fused AUC 0.6875 vs. baseline 0.6701 (+0.0174), modest, directionally positive, not decisive at this sample size.
- `overall` and `no_narrative` AUC are *below* 0.5 (worse than random), verified as a real finding, not a pipeline bug (complaint_id alignment, label polarity, and a manual spot-check all confirmed clean).
- **Added 2026-08-23: paired bootstrap CIs (2,000 resamples, `src/evaluate.py:bootstrap_auc_ci`).** The `has_narrative` point estimate's 95% CI is [-0.12, +0.16], crossing zero: not distinguishable from noise at n=51 alone. `overall`/`no_narrative`'s CIs sit entirely below zero (well-supported). This materially revised the reported verdict, see README Results.
- **Added 2026-08-23: large-scale complement on the proxy-labeled test split (`src/large_scale_comparison.py`, n=1,120, ~20x the hand-labeled sample).** Built because the hand-labeled sample can't scale and its own bootstrap CIs are wide. Proxy label is deliberately independent of `has_narrative` (Section 2), so this comparison isn't confounded the way the hand-labeled one turned out to be. Result: `has_narrative` +0.0052 AUC, fused wins 86.1% of resamples, smaller effect than the hand-labeled point estimate but far more precisely measured and directionally consistent. `no_narrative` sits almost exactly at zero (mechanistically expected: fused's input there is a constant zero vector), which does not replicate the hand-labeled sample's below-random finding for that subgroup, informative disagreement between the two measurements, see README Results for the triangulated verdict.

## Acceptance criteria

- [x] Given the debt-collection CFPB pull, ingestion produces a raw dataset with all required fields and a logged record count.
- [x] The proxy-label rule is printed/logged when applied, not silently baked in.
- [x] Given a complaint with an empty narrative, the fused-model pipeline does not crash and the row is not dropped from the structured comparison.
- [x] Given the same input data and seed, two full pipeline runs produce identical train/test splits and identical (or floating-point-tolerance-equal) metrics.
- [x] Evaluation outputs precision, recall, and AUC for both baseline and fused models on the same held-out sample, in one report (`data/processed/comparison_report.md`), AUC reported as primary, precision/recall as secondary/threshold-sensitive (see Section 6 findings above for why).
- [x] Evaluation additionally splits the baseline-vs-fused comparison by (a) has_narrative and (b) narrative length (short vs. long, relative to MiniLM's 256-token truncation point), added 2026-08-21 during Section 4, to surface whether narrative truncation (see Requirements note on the embedding model) is masking the fused model's true benefit on long narratives specifically, rather than burying it as an unremarked footnote. `has_narrative` is the headline subgroup (see Section 6 findings).
- [x] Given an empty or malformed API response, ingestion exits with a clear error message rather than crashing with a raw traceback.
- [ ] README documents the problem, data scope, proxy-label limitation, methodology, results, and explicit limitations (proxy label is not verified ground truth; CFPB complaints are not literal claims/records; not deployment-ready)., Methodology/Results/Limitations done; Problem section still stubbed to project_description.md.

### Section 7 (Dashboard) acceptance criteria

- [x] The Streamlit app runs and displays a complaint moving through the pipeline with both models' outputs visible, built and verified locally (live prediction, both narrative and no-narrative paths tested in-browser).
- [x] Confidence scores are captioned with the Section 6 calibration finding (not calibrated against the hand-labeled evaluation's true escalation rate) directly next to each prediction, not only in a footer.
- [x] "Escalated" uses a reserved risk color (red), distinct from the app's action/accent color, not a green/positive-reading color.
- [x] An explicit "models agree"/"models disagree" indicator compares baseline and fused directly, not just two independent cards.
- [x] Confidence is shown as a filled meter (`st.progress`), not a bare percentage.
- [x] Model and embedding-model loading confirmed wrapped in `st.cache_resource`, verified not reloading on every Predict click.
- [x] Company dropdown correctness verified programmatically: an out-of-top-20 company resolves to exactly `company_bucketed_Other=1` in the feature vector, not silently mishandled.
- [x] No-narrative state is captioned with the specific Section 6 finding for that subgroup (fused offers little advantage over baseline there), not a generic message.
- [x] Visual system: real color palette (`.streamlit/config.toml`), bordered panels per logical section, `st.metric` with a fused-vs-baseline delta, typographic hierarchy distinguishing subtitle/body/labels/disclaimer, spacing that matches actual content rather than leftover space.
- [x] Deployed to Streamlit Community Cloud: https://early-warning-risk-triage.streamlit.app

## Examples

**Input** (one raw CFPB record with narrative):

```json
{
  "complaint_id": "1234567",
  "product": "Debt collection",
  "sub_product": "Credit card debt",
  "issue": "Attempts to collect debt not owed",
  "company_response": "Closed with explanation",
  "consumer_disputed": "Yes",
  "consumer_complaint_narrative": "I have never had an account with this company...",
  "state": "TN",
  "date_received": "2024-03-11"
}
```

**Output** (one row of the comparison report):

| complaint_id | baseline_pred | baseline_confidence | fused_pred | fused_confidence | true_label |
|---|---|---|---|---|---|
| 1234567 | 0 | 0.61 | 1 | 0.78 | 1 |

## Resources

- CFPB Consumer Complaint Database API docs: https://cfpb.github.io/api/ccdb/
- HuggingFace `sentence-transformers` model hub, for selecting and justifying the embedding model
- Project description: `project_description.md` (this folder), full problem statement, label strategy, and stated limitations this spec implements
