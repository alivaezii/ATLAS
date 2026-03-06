
<img width="1278" height="363" alt="3113f537-dadf-4ecc-ae0e-23178e17a761" src="https://github.com/user-attachments/assets/ebac5d28-27f8-459b-b2fa-e64370f1b59d" />

<p align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![ML Evaluation](https://img.shields.io/badge/ML-Evaluation%20Protocol-purple)
![Reproducibility](https://img.shields.io/badge/reproducibility-focused-success)
![Auditability](https://img.shields.io/badge/auditability-ALAV%20verified-blue)
![Trustworthy AI](https://img.shields.io/badge/trustworthy%20AI-evaluation%20framework-cyan)

</p>



## Overview

ATLAS is a research framework for **leakage‑resilient machine learning
evaluation**.\
It enforces strict information‑flow constraints so that **validation and
test data cannot influence model development**, helping ensure reliable
and reproducible performance estimates.

The framework formalizes a **Split‑Before‑Fit protocol**, provides
automated leakage auditing, and introduces a quantitative **Leakage Risk
Score (LRS)** for evaluation governance.


## Key Components

### 1. Split‑Before‑Fit Protocol

Evaluation pipelines must follow:

1.  Define **train / validation / test splits before modeling**
2.  **Fit all operators on train only**
3.  Use **validation for model selection**
4.  Use the **test set only once for final reporting**



### 2. ALAV --- Automated Leakage Auditing Verifier

ALAV automatically audits pipeline artifacts and detects protocol
violations.

Checks include:

-   split overlap detection
-   preprocessing scope verification
-   test‑reuse detection
-   duplicate leakage detection
-   temporal/group leakage checks
-   cache contamination checks

Output status:

PASS / WARN / FAIL

### 3. Leakage Risk Score (LRS)

ATLAS quantifies evaluation risk using a **Leakage Risk Score
(0--100)**.

Risk levels:

| Score | Interpretation |
|------|----------------|
| 0-19 | Low |
| 20-39 | Medium |
| 40-69 | High |
| 70-100 | Critical |

Computed using surrogate indicators:

-   Duplicate Overlap Rate (DOR)
-   Preprocessing Leakage Indicator (PLI)
-   Test‑Reuse Optimism Proxy (TOP)



## Conceptual Pipeline

    Data → Split → Train → Select → Evaluate

The **evaluation stage is protected by the ATLAS trust layer**,
preventing information leakage from test data.



## Example Usage

``` python
from atlas import Protocol, Auditor

protocol = Protocol()
protocol.split(data)

model = protocol.train(model, train_data)
protocol.select(model, validation_data)

results = protocol.evaluate(model, test_data)

Auditor.run(protocol)
```



## Reproducibility Artifacts

ATLAS produces machine‑auditable artifacts such as:

    data/audit/split_manifest.json
    data/audit/operator_log.csv
    data/audit/duplicate_report.csv
    data/audit/alav_report.json

These allow independent verification of evaluation integrity.

