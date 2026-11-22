# Decision Log: 30-Day Readmission Triage

**For:** Care-Management Lead, Quality & Compliance, Clinical Informatics  
**Scope:** Synthetic Lakeshore admissions; discharge-time scoring; holdout Jan–Mar 2024

> **Decision:** Rank eligible discharges for outreach, with 5% as the default weekly capacity and a dashboard adjustment range of 1–15%. Based on the recorded 2023 validation results, select Logistic Regression: it leads or ties on every reported measure.

## 1. Decision objective and trade-off

The score supports a **weekly outreach-priority list**, not a diagnosis or a prediction that a readmission is preventable. A false positive uses scarce nurse time and can displace a higher-risk patient; a false negative may leave a high-risk patient without outreach. Therefore, assess ranking at the 5% default capacity (**Precision@5%, Lift@5%, and Recall@5%**) and probability quality (**Brier score and ECE**). ROC-AUC is secondary.

## 2. Data audit and feature handling

The datasets are **synthetic, not audited EHR records**: 60,000 training admissions (2021–2023) and 8,000 unlabeled holdout admissions (Jan–Mar 2024). The generator deliberately injects data issues; findings below describe those generated data, not measured hospital-system defects.

| Finding | Handling |
|---|---|
| `post_discharge_call_logged` is recorded after discharge and can leak treatment/outcome information. | Exclude it from model features. |
| `weekend_discharge` can conflict with the discharge date (~1.5% injected). | Recompute from `discharge_date`. |
| Corrupted length of stay (~0.4%) and impossible BMI/heart-rate values (~0.3%). | Recompute invalid length of stay from dates; treat out-of-range BMI, heart rate, and systolic BP as missing. |
| Lab/vital missingness (~3–4% per field overall, higher for one-day stays). | Preserve a missing-count signal; use imputation for Logistic Regression and native missing-value handling for HistGradientBoosting. |

**Open data questions:** Confirm whether prior admissions include out-of-network care and whether discharge disposition is finalized when scoring occurs.

## 3. Validation design

Use a **temporal split**: train on 2021–2022 and validate on 2023; then refit the selected candidate on 2021–2023 before scoring the Jan–Mar 2024 holdout. This avoids training on future dates. Repeat patients can cross time boundaries by design, reflecting returning-patient operations; results are not a patient-disjoint evaluation. The synthetic generator yields **50.25% of unique patients with multiple admissions**, which should be checked against real operations. It also applies a modest 2024 risk shift, so holdout performance may differ.

## 4. Model comparison and selection

Metrics below are from the saved 2023 validation report. Lower Brier/ECE is better; higher remaining metrics are better.

| Candidate | Precision@5% | Lift@5% | Recall@5% | Brier | ECE | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | **63.60%** | **2.61x** | **13.06%** | **0.1641** | **0.0070** | **0.7028** |
| HistGradientBoosting (uncalibrated) | 63.60% | 2.61x | 13.06% | 0.1653 | 0.0110 | 0.6982 |
| HistGradientBoosting (isotonic-calibrated) | 62.49% | 2.57x | 12.83% | 0.1656 | 0.0092 | 0.6968 |

**Selection:** Logistic Regression leads or ties on every reported validation measure; the results do not support the earlier claim that calibrated HistGradientBoosting won. The reproducible pipeline selects by highest Precision@5%, then lowest Brier score, then lowest ECE; on these metrics it selects Logistic Regression. Run `python code/run_assessment.py` before submission to regenerate the model artifact and predictions using that selection rule.

## 5. Probability use, deployment, and monitoring

The baseline’s **2023 ECE is 0.0070** and Brier score is **0.1641**; its calibration bin near 35.5% predicted risk had 35.2% observed readmission. These aggregate results are encouraging but do not prove that every individual score is a reliable probability, nor establish calibration across sites or patient groups. Show scores as estimated risk, not certainty; validate calibration on later labeled data before operational reliance.

The intended action is to call the highest-ranked patients **within available capacity**. A call is not guaranteed to prevent readmission. Monitor input drift, weekly call-list size, and observed-versus-predicted outcomes; audit calibration and outreach rates across age, sex, insurance, and site. Where feasible, retain a small randomized comparison among eligible patients to measure outreach impact and avoid learning only from treated outcomes.
