---
title: Early-Warning Risk Triage
emoji: ⚠️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.62.0"
app_file: dashboard/app.py
pinned: false
---

# Early-Warning Risk Triage Model

Tests whether fusing unstructured complaint-narrative text with structured case fields improves prediction of case escalation, on real CFPB debt-collection complaints — versus structured data alone.

## Problem

Systems that predict risk before it becomes a real problem — a complaint escalating, a claim turning severe, a case going bad — commonly rely on structured data alone (categories, flags, history) and skip the unstructured text sitting right next to it (the complaint itself, a note, a narrative). Fusing the two is not a novel idea — it's an established pattern in healthcare triage, insurance claims processing, and support-ticket escalation prediction. What's largely missing is a transparent, reproducible demonstration of *how much* the text actually helps: most evidence is vendor claims or results from proprietary systems, not a controlled baseline-vs-fused comparison anyone can check.

This project runs that comparison in the open, on public data: the CFPB Consumer Complaint Database, scoped to debt-collection complaints. It builds a baseline model (structured fields only) and a fused model (structured fields + narrative text embeddings), trains both on the identical data and split, and scores both against a hand-labeled evaluation sample — producing a falsifiable, numerical answer to *does the text help, and by how much* — rather than a claim that text was used.

It's a research prototype for a single-semester project, not built for any one company and not deployment-ready; see Explicit Limitations below for what that means concretely. Full problem statement and motivation: `project_description.md`.

## Data

CFPB Consumer Complaint Database public API, scoped to `product = "Debt collection"`. See Methodology below for the exact sampling design.

## Methodology

### Ingestion and sampling

5,600 complaints, stratified evenly (400 each) across 14 quarterly windows from 2023-01-01 through 2026-05-21, rather than one contiguous pull — an initial single-block design only covered ~4 weeks of data at the observed complaint volume (~185/day), which was too thin a slice. The 2026-05-21 cutoff is fixed (not "today") for two reasons: very recent complaints often haven't received a final `company_response` yet ("In progress"), which would corrupt the label rule below, and a fixed cutoff keeps the pull reproducible regardless of when the pipeline is re-run.

Two live-API issues were found and fixed during ingestion:
- The API's documented `frm` offset parameter is a known, unresolved bug ([cfpb/cfpb.github.io#292](https://github.com/cfpb/cfpb.github.io/issues/292)) — it's silently ignored. Fixed by switching to the API's `search_after` cursor pagination.
- Several field names in the original project spec didn't match the live API: `consumer_complaint_narrative`→`complaint_what_happened`, `timely_response`→`timely`. `consumer_disputed` doesn't exist at all — CFPB discontinued it on April 24, 2017.

Verified: 5,600/5,600 unique `complaint_id`s, byte-identical output across independent runs.

### Proxy escalation label

CFPB provides no ground-truth "this case escalated" field. The original design proxied it from `consumer_disputed` + `company_response`; since `consumer_disputed` doesn't exist in the live API, the rule uses `timely` in its place:

```
escalated = 1 if company_response in {"Closed with monetary relief",
                                        "Closed with non-monetary relief",
                                        "Closed with relief"}
             OR timely == "No"
           else 0
```

Deliberately excludes `has_narrative`/`complaint_what_happened` — the label must not be correlated with whether a complaint has text, or the fused-vs-baseline comparison this whole project exists to run would be contaminated (the fused model would look better partly *because* the label was defined by text presence, not because text genuinely helped).

Resulting distribution on the full dataset: **1,488/5,600 escalated (26.6%)**.

### Structured features (baseline model input)

Categorical fields one-hot encoded: `sub_product` (11 values), `issue` (7), `sub_issue` (27), `state` (54 + null→all-zero for the 6 records missing it). `product` is excluded — constant across the dataset (filtered to Debt collection only), zero information. `company_response` and `timely` are excluded — they're the label rule's own inputs, so including them as features would let the model reverse-engineer the label instead of genuinely predicting escalation.

**`company` bucketing (top-20 + "Other").** `company` has 689 unique values on 5,600 records — one-hot encoding all of them would add 689 mostly-sparse columns (most companies appear only 1-5 times), risking overfitting on rare companies and dwarfing every other structured feature by column count.

Cumulative coverage by rank (companies ranked by complaint count, descending):

| Rank (N) | Cumulative coverage | Count at that rank |
|---|---|---|
| 1 | 6.98% | 391 |
| 5 | 28.73% | 219 |
| 10 | 42.45% | 104 |
| **20** | **52.96%** | **45** |
| 25 | 56.30% | 34 |
| 30 | 59.02% | 28 |
| 40 | 63.52% | 24 |
| 60 | 70.27% | 16 |
| 100 | 78.18% | 8 |
| 340 | 93.12% | 2 |
| 689 (all) | 100.00% | 1 |

**N=20 chosen.** Reasoning:
- Coverage-per-column drops sharply after N≈20: companies 1-10 contribute ~4 coverage points each on average; companies 20-30 contribute ~0.6 points each — most of the recoverable coverage is already captured by N=20, and pushing further mainly adds columns, not signal.
- Sample size per bucket: with a 20% held-out test split, a company with 45 complaints (the N=20 cutoff) nets roughly ~36 train / ~9 test examples — thin, but workable. A company with 28 complaints (the N=30 cutoff) nets roughly ~22 train / ~6 test, and with the label's 26.6% positive rate, that's potentially only 5-6 escalated examples total — too sparse to learn a stable signal from rather than noise, which reintroduces the overfitting problem bucketing exists to solve.
- Dimensionality stays proportionate to the other structured features: 21 company columns (20 + Other) sits between `sub_issue` (27) and `state` (54), rather than dominating.

Result: **52.96% of records retain their actual company identity; the remaining 47.04% fall into "Other."** This is a real limitation — most of the company signal, if it exists, is only usable for about half the dataset — and is inherent to the long-tail company distribution, not something a different N meaningfully fixes (even N=30 only reaches 59.02%).

Output: `data/processed/structured_features.csv` — 5,600 rows × 120 feature columns (11+7+27+54+21) plus `complaint_id` and `escalated`.

### Text embeddings

`sentence-transformers/all-MiniLM-L6-v2` (384-dim; chosen for fast CPU inference and small footprint — see Requirements in `BUILD_INSTRUCTIONS.md` for the full tradeoff vs. larger models). Complaints with no narrative (55.8%, 3,125/5,600) get an exact zero vector rather than being dropped — verified directly (3,125/3,125 no-narrative rows all-zero; 2,475/2,475 narrative rows non-zero and unit-normalized). Embedding generation is bit-exact reproducible across runs (max diff = 0.0 between two independent full runs).

Output: `data/processed/text_embeddings.csv` — 5,600 rows × 384 dims + `complaint_id`.

### Models

Logistic regression for both baseline and fused models (chosen over gradient boosting for interpretability, well-calibrated probabilities for the dashboard's confidence scores, and lower overfitting risk given 504 features on ~4,480 training rows). `class_weight="balanced"` for the 26.6%-positive imbalance. Both models trained on the identical stratified train/test split (4,480 train / 1,120 test, `seed=42`) — verified directly that baseline and fused share identical row indices and labels, not just the same split *size*.

**L2 regularization (C=0.1).** Initial baseline model (C=1.0, scikit-learn's default) showed three `company_bucketed` features with implausibly large coefficients (Kriya Capital +3.64, CL Holdings -3.46, Transworld Systems -2.78) — investigation found all three show **perfect separation**: every complaint against that company in this dataset falls on the same side of the label (Kriya Capital 104/104 escalated; CL Holdings 0/202 escalated; Transworld Systems 0/88 escalated). Perfect separation is a known logistic regression pathology — the coefficient doesn't converge to a stable value, it inflates toward whatever the solver's stopping criteria happen to allow.

C was swept over {1.0, 0.5, 0.1} on the baseline model:

| C | Kriya Capital | CL Holdings | Transworld | precision | recall | AUC (internal test split) |
|---|---|---|---|---|---|---|
| 1.0 | +3.6445 | -3.4610 | -2.7828 | 0.5587 | 0.6544 | 0.7940 |
| 0.5 | +3.1129 | -2.9671 | -2.2690 | 0.5607 | 0.6510 | 0.7941 |
| **0.1** | **+1.8489** | **-1.7821** | **-1.1587** | **0.5706** | **0.6376** | **0.7878** |

C=0.1 chosen: nearly halves all three coefficients for a <1% relative AUC cost (0.7940→0.7878) and a slight precision gain. C=0.5 barely regularized anything (~15% shrinkage). Applied to **both** baseline and fused models, not just baseline — the fused model shares the same `company_bucketed` columns, so using different regularization strength between the two would itself become a confound in the baseline-vs-fused comparison.

Post-regularization internal test-split AUC (sanity check only, not the official result): **baseline 0.7878, fused 0.7900**. The official baseline-vs-fused comparison is Section 6, scored against the separate hand-labeled sample, not this number.

Output: `models/baseline_model.joblib`, `models/fused_model.joblib`.

### Hand-labeling rubric

The hand-labeled sample (106 of a 150-row template actually labeled, sampled uniformly at random from the model's test split, seeded — see `src/sample_for_labeling.py`) was labeled without access to `company_response`, `timely`, or the proxy `escalated` label, so the labeler's judgment would be independent of the proxy rule rather than reproduce it.

Working definition applied: *"escalated" means the complaint, based only on what's described (issue/sub_issue/narrative), reflects a case where the company was genuinely at fault and should be expected to make it right, or where the company's handling was clearly inadequate* — not a prediction of the actual outcome, a judgment of whether the complaint itself describes a serious, legitimate problem versus a minor or ambiguous one.

Signals used, when a narrative was present: concrete factual claims of wrongdoing (wrong debt amount, debt not theirs, no verification provided, threats of legal action, identity theft) versus vague/purely emotional complaints with no clear factual claim; evidence of persistence or harm (repeated contact after being told to stop, months unresolved, credit damage, wage garnishment threats); specificity and internal consistency of the account.

When no narrative was present (about half the sample), `issue`/`sub_issue` text was the only available signal — a weaker basis than a narrative, and treated as such.

**Consistency recheck.** After initial labeling (106/150 rows), a review of no-narrative rows found an inconsistency: `sub_issue="Debt is not yours"` (14 rows) was escalated only 7.1% of the time, while `sub_issue="Debt was result of identity theft"` (7 rows) and `"Sued you without properly notifying you of lawsuit"` (2 rows) were escalated 100% of the time — despite "debt is not yours" and "identity theft" being substantively the same kind of claim (the debt isn't legitimately the consumer's). 19 rows were rechecked against this pattern; 18 were corrected from `escalated=0` to `escalated=1`. Post-correction: no-narrative escalated rate rose from 36.4% to 63.6%; has-narrative rate stayed at ~94%.

**Remaining limitation, not fully resolved by the recheck.** The hand-labeled ground truth was constructed with a labeler who relied heavily on narrative-described identity theft and threat-of-legal-action language for escalation calls, falling back to `sub_issue` category for no-narrative complaints. Since the baseline model structurally never sees narrative content, part of what the ground truth calls "correct" may be information baseline cannot access by design — this could both understate baseline's true structured-data performance and inflate fused's apparent advantage for reasons connected to labeling criteria rather than genuine embedding-based understanding of complaint content. This is a limitation of the evaluation ground truth's construction, not a fixable modeling issue.

### Evaluation

Both models scored against the hand-labeled sample (n=106) — the real held-out ground truth, never used in training. Full report: `data/processed/comparison_report.md`.

**Headline finding: the proxy label and hand-labeled ground truth diverge sharply in base rate** — proxy (training, n=5,600): 26.6% escalated; hand-labeled (evaluation, n=106): 78.3% escalated. This is direct evidence the proxy (company concession behavior) and genuine complaint severity (human judgment) are related but distinct constructs, not a validation nuisance. Traced to a concrete mechanism: in the training data, companies concede far more readily on cheap procedural sub_issues (`"Notification didn't disclose it was an attempt to collect a debt"`: 54.6% proxy-escalated, n=491) than on substantive claims like identity theft (28.0%, n=751) or "debt is not yours" (24.7%, n=1,475) — likely because procedural fixes are cheap to grant while substantive claims are more often denied outright. The models learned this pattern, so they can rank procedural complaints as *more* likely to escalate than identity-theft complaints — the opposite of genuine severity.

**Consequence for reading these results:** since both models were trained to predict "positive" at ~26.6% (the proxy rate), scoring them against a 78.3%-positive ground truth with a fixed 0.5 threshold mechanically depresses recall — a calibration artifact, not a discrimination failure. **AUC (threshold-independent) is the primary metric; precision/recall at 0.5 are secondary.**

| Subgroup | n | Baseline AUC | Fused AUC | Delta |
|---|---|---|---|---|
| overall | 106 | 0.4531 | 0.4280 | -0.0251 |
| **has_narrative (headline)** | **51** | **0.6701** | **0.6875** | **+0.0174** |
| no_narrative | 55 | 0.4257 | 0.4229 | -0.0029 |
| short_narrative | 33 | 0.7056 | 0.7222 | +0.0167 |
| long_narrative | 18 | undefined (single-class: 18/18 escalated) | undefined | n/a |

`overall` and `no_narrative` are below 0.5 (worse than random) — this traces directly to the sub_issue-concession-rate mechanism above, not to a pipeline defect (verified: complaint_id alignment between predictions and ground truth confirmed exact match; model `classes_=[0,1]` on both models, no label inversion; hand-labeled `escalated` column matches evaluation input row-for-row).

### Dashboard

`dashboard/app.py` — Streamlit. Structured fields are dropdowns populated from the actual trained categories (so every selection maps to a real one-hot column, verified directly — an out-of-top-20 company resolves to exactly `company_bucketed_Other=1`, no silent mishandling); the narrative is free text, embedded live with MiniLM on every prediction — a genuine hosted-inference demo, not a lookup table. A "Load a random real example" button pulls from the 106-row hand-labeled sample and shows the human-judged ground truth alongside both models' live predictions for direct comparison. Verified locally end-to-end: narrative and no-narrative (zero-vector) paths both produce distinct, sensible predictions from both models.

**Design system:** a real palette via `.streamlit/config.toml` (indigo action color reserved for the Predict button, separate red/green risk colors for Escalated/Not-Escalated reserved for that meaning only), bordered `st.container` panels instead of free-floating fields, `st.metric` with a fused-vs-baseline confidence delta, and an explicit "models agree/disagree" indicator above the two prediction cards. The Section 6 calibration finding (78.3% true rate vs. 26.6% training rate) is surfaced directly under each confidence score, not just in the footer; a no-narrative note ties directly back to the evaluation finding that fused offers little advantage in that subgroup.

Deployed via HuggingFace Spaces' GitHub-sync (this repo's README.md carries the required Space YAML frontmatter, `app_file: dashboard/app.py`) rather than a second, separately-maintained repo. `dashboard/app.py` doesn't import from `src/` — its small amount of feature-encoding logic is duplicated locally rather than importing the full pipeline package, keeping the deployed Space's dependency surface limited to what it actually runs.

## Results

**Verdict: modest, directionally positive evidence that fusion helps, too small and too thin a sample to call decisive.** On the deconfounded `has_narrative` subgroup — the one comparison where a fused-model advantage can't be explained by simply detecting whether a narrative exists — fused beats baseline by +0.0174 AUC (0.6875 vs 0.6701), consistent within the short_narrative subset (+0.0167). `long_narrative` (n=18) can't be assessed for AUC at all — every hand-labeled long-narrative complaint was scored escalated, so there's no negative class to rank against, which is itself informative about what got labeled escalated but not about model quality.

The more consequential finding is the base-rate divergence itself: the proxy label used for training and genuine human-judged severity disagree by a factor of ~3x in prevalence, traced to a specific, plausible mechanism (companies concede cheaply on procedural issues, deny substantively on serious ones). This means confidence in either model's *absolute* quality should be low regardless of the fused-vs-baseline comparison — the comparison itself (which model discriminates better, given the same imperfect proxy-trained starting point) is more trustworthy than either model's standalone numbers.

**What would strengthen this result:** a larger hand-labeled sample (n=51 in the headline subgroup is thin — a ±0.017 AUC delta is a fragile signal at this size) and/or a training label rule that incorporates severity signals beyond company concession behavior.

## Explicit Limitations

- **Three company-level features showed perfect separation before regularization.** `company_bucketed_Kriya Capital, LLC` (104 complaints, 100% escalated), `CL Holdings LLC` (202 complaints, 0% escalated), and `TRANSWORLD SYSTEMS INC` (88 complaints, 0% escalated) each fell entirely on one side of the label in this dataset. This is a real pattern worth noting — not a confirmed effect. At these sample sizes, perfect separation cannot be distinguished from small-sample coincidence using this dataset alone; it could reflect a genuine, consistent difference in how these companies handle debt-collection complaints, or it could be an artifact of this specific 5,600-row, 2023-2026 stratified sample (e.g., a company's complaints clustering in one particular quarter for unrelated reasons). L2 regularization (C=0.1) was applied to reduce the resulting coefficient instability (see Methodology → Models), but this does not resolve the underlying ambiguity about whether the pattern is real.
- **Long narratives are truncated by the embedding model.** `all-MiniLM-L6-v2` has a 256-token max input length. 28.7% of narratives with text (711/2,475) are long enough to risk truncation, meaning the embedding reflects roughly the opening ~200 words only. Escalation-relevant language (regulatory threats, repeated-contact complaints, "unresolved after months" language) often appears later in longer narratives — so this is a **measurement limitation, not evidence against the hypothesis**: it may cause the fused model's true benefit to be understated on long narratives specifically, rather than reflecting an actual absence of textual signal. Section 6 (Evaluate) splits the comparison by narrative length in addition to has-narrative, specifically to surface whether this limitation shows up in the results.
- The proxy escalation label is an approximation, not verified ground truth — validated only against a small hand-labeled sample (see `BUILD_INSTRUCTIONS.md`).
- Observed narrative rate on this dataset (44.2%) is notably higher than CFPB's stated all-products average (~26%); this is likely product-specific (debt collection) rather than representative of CFPB complaints broadly.
- Company identity is only usable as a feature for ~53% of records; the rest fall into a generic "Other" bucket.
- CFPB complaints are complaints filed against companies, not literal insurance claims, clinical records, or loan applications — this project demonstrates a generalizable pattern, not a domain-specific production system.
- This is a research prototype for a single semester; no claim is made that it is deployment-ready.
