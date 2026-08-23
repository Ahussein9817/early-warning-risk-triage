# Early-Warning Risk Model: Fusing Structured and Unstructured Data for Predictive Triage

## Overview

Fusing structured data with unstructured text for risk triage is not a novel idea: it's an established pattern across healthcare triage, insurance claims processing, and support-ticket escalation prediction. What's largely missing from that landscape is a transparent, reproducible demonstration of *how much* the text actually helps: most evidence is vendor claims or proprietary systems, not a controlled baseline-vs-fused comparison anyone can check. This project builds and evaluates a model that combines both, and directly tests whether the combination produces a measurably better prediction than either source alone, in the open, on public data.

The project is not built for any single company. It's a public demonstration of a pattern that recurs across insurance, fintech, and healthcare: predicting a negative outcome before it happens, using every available signal rather than a subset of it.

## The Problem

When something is about to go wrong, a customer complaint escalating, a patient's condition worsening, a claim turning into a dispute, there is usually both:

- **Structured data**: account history, product type, prior flags, categorical fields
- **Unstructured data**: the actual written narrative, notes, or complaint text

The pattern of fusing both is well established in industry and research (e.g. hybrid ED triage models, NLP over insurance adjuster notes, ML-based support-ticket escalation prediction). What's rarely shown is the effect size, out in the open, with a controlled ablation: almost none of the public discussion is backed by a visible baseline-vs-fused comparison with real numbers, almost none of it runs on open/reproducible data, and almost none of it is transparent about label quality. This project asks a specific, falsifiable question on one public case: **does fusing text with structured data actually improve prediction, and by how much?**

## Data Source

The **CFPB Consumer Complaint Database**, a free, public API maintained by the U.S. Consumer Financial Protection Bureau. It contains real, unfiltered consumer complaints against financial companies.

- Structured fields: product, sub-product, issue, sub-issue, company response, timely-response flag, disputed flag, state
- Roughly 26% of complaints include a written consumer narrative
- Scope is narrowed to a single product category (e.g., debt collection or mortgage complaints) to keep the data coherent and the problem well-defined

## The Label Problem

The CFPB dataset has no ground-truth label for "this case escalated" or "this case was high-risk." This is a real limitation, addressed directly rather than glossed over:

- The proxy signal is `company_response` (whether the company had to grant relief) combined with `timely` (whether the company missed its response deadline). The original design called for the "disputed" flag as a second signal, but CFPB discontinued that field in the live API on April 24, 2017, `timely` was substituted, keeping the same two-independent-structured-signals design intent. Neither signal touches the complaint narrative, so the label stays independent of the text feature the fused model is being tested on.
- A small hand-labeled sample (roughly 100-200 complaints) is created for evaluation, to validate the proxy against actual human judgment on a subset

This means reported performance numbers are **indicative, not validated ground truth**, a limitation stated upfront in the project's documentation, not discovered by a future critic.

## Methodology

**Stage 1, Baseline model.** Structured features only (product, sub-product, issue category, prior flags), using a standard classifier (logistic regression or gradient boosting) to predict escalation likelihood.

**Stage 2, Fused model.** The same structured features, plus text embeddings generated from the complaint narrative (via a HuggingFace embedding model), retrained on the combined input.

**Stage 3, Comparison.** The fused model is evaluated against the baseline on the same held-out labeled sample, using metrics appropriate for an imbalanced classification problem (precision, recall, AUC). The explicit goal is a numerical answer to whether, and how much, the text signal improves prediction.

### Non-negotiable requirement

The baseline-vs-fused comparison must be built in and reported as a core part of the project, not treated as optional polish added at the end. Without this comparison, the project is only a claim that text was used, not evidence that it helped.

## Deliverables

- A GitHub repository containing the full pipeline: data ingestion, extraction, baseline model, fused model, and evaluation
- A deployed Streamlit dashboard demonstrating a complaint moving through the pipeline live, with model output and confidence visible
- A README documenting the problem, data, methodology, results, and explicit limitations

## Why This Project

- **It extends real, provable experience.** Time-series forecasting work from a prior role in banking is extended into a new modality (text) rather than claiming an entirely new skill without evidence of it.
- **It mirrors a recurring, real business problem.** The same underlying pattern shows up independently across multiple companies and domains: real-time cash-flow monitoring in fintech, disease-progression prediction in value-based healthcare, and claims-severity assessment in insurance. All are versions of "predict the negative outcome before it happens, using every available signal." This project demonstrates the general pattern rather than any one company's specific system.
- **It fills a transparency gap, not an invention gap.** The technique itself (fuse structured + text signals) is already used in production systems across these domains. What's missing publicly is a reproducible, auditable comparison showing the actual size of the improvement, most claims of "NLP improves triage" are asserted by vendors, not demonstrated with an open baseline-vs-fused ablation on data anyone can inspect.
- **It's honestly scoped for a single semester.** The data is public and immediately accessible, the label strategy is defined upfront, and the technical scope (three stages: baseline, fusion, comparison) is achievable without waiting on data access approvals or proprietary systems.

## Explicit Limitations

Stated directly, not left for someone else to discover:

- The proxy label is an approximation of true risk/escalation, not verified ground truth. Reported performance is indicative, not a validated benchmark.
- CFPB complaints are complaints filed against companies, not literal insurance claims, clinical records, or loan applications. The project demonstrates a generalizable pattern, not a domain-specific production system ready for deployment.
- No claim is made that this system could be deployed as-is in any real company's environment; it is a research prototype built to test a specific hypothesis about combining structured and unstructured signals.
