# Reproduced hand-labeled evaluation (independent re-run from committed artifacts)

Compare against README 'Evaluation'. Bootstrap: 2,000 paired resamples, seed from config.

| Subgroup | n | metric | README | Reproduced | abs diff |
|---|---|---|---|---|---|
| overall | 106 | n | 106 | 106.0000 | 0.0000 |
| overall | 106 | baseline AUC | 0.4531 | 0.4531 | 0.0000 |
| overall | 106 | fused AUC | 0.428 | 0.4280 | 0.0000 |
| overall | 106 | delta | -0.0251 | -0.0251 | 0.0000 |
| overall | 106 | delta CI low | -0.0539 | -0.0539 | 0.0000 |
| overall | 106 | delta CI high | -0.0003 | -0.0003 | 0.0000 |
| has_narrative | 51 | n | 51 | 51.0000 | 0.0000 |
| has_narrative | 51 | baseline AUC | 0.6701 | 0.6701 | 0.0000 |
| has_narrative | 51 | fused AUC | 0.6875 | 0.6875 | 0.0000 |
| has_narrative | 51 | delta | 0.0174 | 0.0174 | 0.0000 |
| has_narrative | 51 | delta CI low | -0.12 | -0.1200 | 0.0000 |
| has_narrative | 51 | delta CI high | 0.16 | 0.1600 | 0.0000 |
| no_narrative | 55 | n | 55 | 55.0000 | 0.0000 |
| no_narrative | 55 | baseline AUC | 0.4257 | 0.4257 | 0.0000 |
| no_narrative | 55 | fused AUC | 0.4229 | 0.4229 | 0.0000 |
| no_narrative | 55 | delta | -0.0029 | -0.0029 | 0.0000 |
| no_narrative | 55 | delta CI low | -0.0117 | -0.0117 | 0.0000 |
| no_narrative | 55 | delta CI high | 0.0 | 0.0000 | 0.0000 |
| short_narrative | 33 | n | 33 | 33.0000 | 0.0000 |
| short_narrative | 33 | baseline AUC | 0.7056 | 0.7056 | 0.0000 |
| short_narrative | 33 | fused AUC | 0.7222 | 0.7222 | 0.0000 |
| short_narrative | 33 | delta | 0.0167 | 0.0167 | 0.0000 |
| short_narrative | 33 | delta CI low | -0.1935 | -0.1935 | 0.0000 |
| short_narrative | 33 | delta CI high | 0.2009 | 0.2009 | 0.0000 |

long_narrative: single-class in the hand-labeled data (n=18, positives=18); AUC undefined, as in the README.
