# PromptWall evaluation report

_Generated 2026-10-11T20:31:13+00:00 by `promptwall evaluate`._

## Prompt-injection classifier

Thresholds are calibrated on benign validation prompts for a **1% false-positive budget** on every source (worst case), unless stated otherwise. Cells show detection rate / false-positive rate on the test data.

| Experiment | Threshold | SPML (unseen chatbots) | deepset | jackhhao (jailbreaks) | Dolly (benign) |
|---|---:|---:|---:|---:|---:|
| multi-source | 0.534 | 98.7% / 1.7% | 45.0% / 0.0% | 94.9% / 0.8% | FPR 0.03% |
| single-source | 0.537 | 98.8% / 2.0% | 15.2% / 1.0% | 83.2% / 6.7% | FPR 0.00% |
| multi-source, pooled calibration | 0.369 | 99.1% / 4.0% | 55.0% / 0.0% | 97.1% / 6.6% | FPR 0.20% |
| multi-source, word n-grams only | 0.748 | 96.4% / 0.3% | 28.3% / 0.0% | 89.0% / 1.6% | FPR 0.00% |
| multi-source, char n-grams only | 0.706 | 96.8% / 0.6% | 23.3% / 0.0% | 89.7% / 0.0% | FPR 0.00% |

In the single-source experiment, deepset and jackhhao are entirely unseen sources (all rows are used); in the other experiments their held-out test split is used.

![Detection across sources](figures/cross_source.png)

![ROC](figures/roc.png)

## PII detector (synthetic benchmark)

| Entity | Precision | Recall | Support |
|---|---:|---:|---:|
| EMAIL | 100.0% | 100.0% | 253 |
| PHONE | 100.0% | 100.0% | 291 |
| CREDIT_CARD | 98.2% | 100.0% | 224 |
| IBAN | 100.0% | 100.0% | 230 |
| FR_NIR | 100.0% | 100.0% | 254 |
| IP_ADDRESS | 100.0% | 100.0% | 252 |
| **All** | **99.73%** | **100.00%** | |

## Latency

| Measure | p50 | p95 |
|---|---:|---:|
| Classifier, one prompt | 1.88 ms | 4.57 ms |
| Full guard (input + output scanners) | 2.30 ms | 5.50 ms |
| Proxied request, instant mock upstream | 3.32 ms | 6.76 ms |

## Data

| Source/split | Rows | Injections |
|---|---:|---:|
| deepset/test | 116 | 60 |
| deepset/train | 436 | 162 |
| deepset/val | 110 | 41 |
| dolly/test | 2,989 | 0 |
| dolly/train | 8,903 | 0 |
| dolly/val | 2,937 | 0 |
| jackhhao/test | 258 | 136 |
| jackhhao/train | 828 | 415 |
| jackhhao/val | 205 | 102 |
| spml/test | 3,148 | 2,499 |
| spml/train | 11,183 | 8,784 |
| spml/val | 1,586 | 1,258 |
