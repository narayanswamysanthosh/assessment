import os
import numpy as np
import pandas as pd


def generate_hospital_data(seed: int = 42, output_dir: str = "synthetic_data_generator/data") -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset sizes and temporal boundaries
    n_train = 60_000
    n_holdout = 8_000
    n_total = n_train + n_holdout

    train_start = pd.Timestamp("2021-01-01")
    train_end = pd.Timestamp("2023-12-31")
    holdout_start = pd.Timestamp("2024-01-01")
    holdout_end = pd.Timestamp("2024-03-31")

    # 2. Generate patient pool so ~30% of patients appear in multiple admissions
    n_unique_patients = 50_500
    patient_pool = np.array([f"PAT_{i:06d}" for i in range(1, n_unique_patients + 1)])

    # Assign patient-level baseline risk & demographics
    patient_base_age = rng.integers(18, 92, size=n_unique_patients)
    patient_sex = rng.choice(["F", "M"], size=n_unique_patients, p=[0.52, 0.48])
    patient_base_comorb = rng.poisson(2.2, size=n_unique_patients)
    patient_base_bmi = np.clip(rng.normal(28.5, 6.2, size=n_unique_patients), 15.0, 58.0)
    patient_frailty = rng.normal(0.0, 0.85, size=n_unique_patients)

    # Use skewed weights to simulate patients with differing admission frequency.
    weights = rng.pareto(a=2.5, size=n_unique_patients) + 0.5
    weights /= weights.sum()
    sampled_patient_idx = rng.choice(n_unique_patients, size=n_total, replace=True, p=weights)

    # 3. Assign admission dates (60k in 2021-2023, 8k in Jan-Mar 2024)
    train_days = (train_end - train_start).days
    holdout_days = (holdout_end - holdout_start).days

    train_admit_offsets = rng.integers(0, train_days + 1, size=n_train)
    holdout_admit_offsets = rng.integers(0, holdout_days + 1, size=n_holdout)

    admit_dates = pd.DatetimeIndex(
        [train_start + pd.Timedelta(days=int(d)) for d in train_admit_offsets]
        + [holdout_start + pd.Timedelta(days=int(d)) for d in holdout_admit_offsets]
    )

    df = pd.DataFrame({
        "patient_idx": sampled_patient_idx,
        "patient_id": patient_pool[sampled_patient_idx],
        "admit_date": admit_dates,
        "is_holdout": np.array([0] * n_train + [1] * n_holdout),
    })
    df = df.sort_values(["patient_id", "admit_date"]).reset_index(drop=True)

    # 4. Generate clinical & administrative features
    p_idx = df["patient_idx"].values
    years_elapsed = (df["admit_date"].dt.year - 2021).clip(lower=0)

    df["hospital_site"] = rng.choice(
        ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E", "Site_F"],
        size=n_total,
        p=[0.24, 0.20, 0.18, 0.15, 0.13, 0.10],
    )
    df["age"] = np.clip(patient_base_age[p_idx] + years_elapsed, 18, 100)
    df["sex"] = patient_sex[p_idx]

    df["insurance_type"] = np.where(
        df["age"] >= 65,
        rng.choice(
            ["Medicare", "Commercial", "Medicaid", "Self-pay"],
            size=n_total,
            p=[0.82, 0.12, 0.05, 0.01],
        ),
        rng.choice(
            ["Commercial", "Medicaid", "Self-pay", "Medicare"],
            size=n_total,
            p=[0.58, 0.26, 0.11, 0.05],
        ),
    )

    df["admission_source"] = rng.choice(
        ["Emergency", "Referral", "Transfer"],
        size=n_total,
        p=[0.66, 0.24, 0.10],
    )

    dx_groups = [
        "Heart_Failure",
        "COPD_Asthma",
        "Pneumonia_Sepsis",
        "GI_Disorder",
        "Renal_Failure",
        "Diabetes_Metabolic",
        "Arrhythmia",
        "Orthopedic",
        "Stroke_Neuro",
        "Oncology",
        "Behavioral_Health",
        "Other_Medical",
    ]
    df["primary_dx_group"] = rng.choice(dx_groups, size=n_total)

    df["comorbidity_count"] = np.clip(
        patient_base_comorb[p_idx] + rng.poisson(0.3 * years_elapsed),
        0,
        15,
    )

    # Length of stay (days) driven by acuity and comorbidities
    acuity = (
        0.25 * df["comorbidity_count"]
        + 0.45 * (df["admission_source"] == "Emergency").astype(int)
        + 0.60 * (df["admission_source"] == "Transfer").astype(int)
        + 0.50 * patient_frailty[p_idx]
    )
    raw_los = np.round(rng.lognormal(mean=1.1 + 0.12 * acuity, sigma=0.55))
    df["los_days"] = np.clip(raw_los, 1, 45).astype(int)
    df["discharge_date"] = df["admit_date"] + pd.to_timedelta(df["los_days"], unit="D")

    # Historical utilization
    prior_adm_rate = np.clip(
        0.35 + 0.22 * df["comorbidity_count"] + 0.35 * np.maximum(0, patient_frailty[p_idx]),
        0.05,
        6.0,
    )
    df["n_prior_admissions_12m"] = rng.poisson(prior_adm_rate)

    ed_rate = np.clip(
        0.45 + 0.25 * df["n_prior_admissions_12m"] + 0.20 * (df["insurance_type"] == "Medicaid"),
        0.05,
        8.0,
    )
    df["n_ed_visits_6m"] = rng.poisson(ed_rate)

    df["num_medications"] = np.clip(
        rng.poisson(4.0 + 1.4 * df["comorbidity_count"] + 0.15 * df["los_days"]),
        0,
        30,
    )
    df["bmi"] = np.round(patient_base_bmi[p_idx] + rng.normal(0, 0.6, size=n_total), 1)

    # Labs and Vitals before discharge
    renal_flag = (df["primary_dx_group"] == "Renal_Failure").astype(float)
    hf_flag = (df["primary_dx_group"] == "Heart_Failure").astype(float)

    df["lab_creatinine_last"] = np.round(
        np.clip(
            rng.lognormal(0.05 + 0.45 * renal_flag + 0.04 * df["comorbidity_count"], 0.35),
            0.4,
            9.5,
        ),
        2,
    )
    df["lab_sodium_last"] = np.round(
        np.clip(rng.normal(138.5 - 1.8 * hf_flag - 0.3 * acuity, 3.8), 118.0, 155.0),
        1,
    )
    df["lab_hemoglobin_last"] = np.round(
        np.clip(
            rng.normal(12.6 - 0.35 * df["comorbidity_count"] - 0.5 * renal_flag, 1.9),
            6.0,
            18.0,
        ),
        1,
    )
    df["systolic_bp_last"] = np.round(
        np.clip(rng.normal(126.0 + 0.18 * (df["age"] - 50) - 2.0 * hf_flag, 18.0), 75.0, 210.0),
        0,
    )
    df["heart_rate_last"] = np.round(
        np.clip(rng.normal(78.0 + 2.5 * acuity, 14.0), 42.0, 150.0),
        0,
    )

    # Discharge disposition & follow-up
    disp_probs = np.column_stack([
        np.full(n_total, 0.58),
        0.18 + 0.02 * (df["age"] > 70),
        0.14 + 0.03 * (df["age"] > 75) + 0.02 * (df["los_days"] > 7),
        np.full(n_total, 0.10),
    ])
    disp_probs /= disp_probs.sum(axis=1, keepdims=True)
    disp_choices = ["Home", "Home with services", "Skilled nursing", "Rehab"]
    df["discharge_disposition"] = [disp_choices[rng.choice(4, p=p)] for p in disp_probs]

    # True weekend flag (Saturday=5, Sunday=6)
    df["weekend_discharge"] = (df["discharge_date"].dt.dayofweek >= 5).astype(int)

    followup_prob = np.clip(
        0.68 - 0.18 * df["weekend_discharge"] - 0.10 * (df["insurance_type"] == "Self-pay"),
        0.15,
        0.90,
    )
    df["followup_visit_scheduled"] = rng.binomial(1, followup_prob)

    # 5. Compute ground-truth 30-day readmission risk
    drift_2024 = 0.12 * df["is_holdout"]
    logit_risk = (
        -2.75
        + 0.34 * df["n_prior_admissions_12m"]
        + 0.24 * df["n_ed_visits_6m"]
        + 0.14 * df["comorbidity_count"]
        + 0.035 * df["los_days"]
        + 0.03 * df["num_medications"]
        + 0.38 * (df["lab_creatinine_last"] > 1.8).astype(float)
        + 0.32 * (df["lab_hemoglobin_last"] < 10.0).astype(float)
        + 0.28 * (df["lab_sodium_last"] < 134.0).astype(float)
        + 0.25 * (df["discharge_disposition"] == "Skilled nursing").astype(float)
        + 0.22 * df["weekend_discharge"]
        - 0.30 * df["followup_visit_scheduled"]
        + 0.25 * (df["insurance_type"] == "Medicaid").astype(float)
        + 0.40 * patient_frailty[p_idx]
        + drift_2024
        + rng.normal(0, 0.25, size=n_total)
    )
    true_prob = 1.0 / (1.0 + np.exp(-logit_risk))
    df["readmitted_30d"] = rng.binomial(1, true_prob)

    # 6. Target/Treatment Leakage Column: post_discharge_call_logged
    leak_logit = -2.2 + 1.8 * df["readmitted_30d"] + 0.35 * df["n_prior_admissions_12m"]
    df["post_discharge_call_logged"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-leak_logit)))

    # 7. Inject realistic hospital extract messiness (~4% missing labs/vitals + audit anomalies)
    lab_vital_cols = [
        "lab_creatinine_last",
        "lab_sodium_last",
        "lab_hemoglobin_last",
        "systolic_bp_last",
        "heart_rate_last",
    ]
    for col in lab_vital_cols:
        miss_prob = np.where(df["los_days"] == 1, 0.075, 0.032)
        mask = rng.random(n_total) < miss_prob
        df.loc[mask, col] = np.nan

    # Anomaly A: ~0.4% corrupted/negative los_days
    corrupt_los_idx = rng.choice(n_total, size=int(0.004 * n_total), replace=False)
    df.loc[corrupt_los_idx, "los_days"] = rng.choice([-1, 0, 999], size=len(corrupt_los_idx))

    # Anomaly B: ~1.5% mismatch between weekend_discharge flag and actual discharge_date
    corrupt_wknd_idx = rng.choice(n_total, size=int(0.015 * n_total), replace=False)
    df.loc[corrupt_wknd_idx, "weekend_discharge"] = 1 - df.loc[corrupt_wknd_idx, "weekend_discharge"]

    # Anomaly C: ~0.3% impossible physiological outliers
    corrupt_vital_idx = rng.choice(n_total, size=int(0.003 * n_total), replace=False)
    half = len(corrupt_vital_idx) // 2
    df.loc[corrupt_vital_idx[:half], "bmi"] = -99.0
    df.loc[corrupt_vital_idx[half:], "heart_rate_last"] = 0.0

    # 8. Split into Training (2021-2023) and Holdout (Jan-Mar 2024) & assign clean IDs
    train_df = (
        df[df["is_holdout"] == 0]
        .sort_values(["admit_date", "patient_id"])
        .reset_index(drop=True)
    )
    holdout_df = (
        df[df["is_holdout"] == 1]
        .sort_values(["admit_date", "patient_id"])
        .reset_index(drop=True)
    )

    train_df["admission_id"] = [f"ADM_TR_{i:06d}" for i in range(1, len(train_df) + 1)]
    holdout_df["admission_id"] = [f"ADM_HO_{i:06d}" for i in range(1, len(holdout_df) + 1)]

    schema_cols_26 = [
        "admission_id",
        "patient_id",
        "hospital_site",
        "admit_date",
        "discharge_date",
        "age",
        "sex",
        "insurance_type",
        "admission_source",
        "primary_dx_group",
        "comorbidity_count",
        "los_days",
        "n_prior_admissions_12m",
        "n_ed_visits_6m",
        "num_medications",
        "bmi",
        "lab_creatinine_last",
        "lab_sodium_last",
        "lab_hemoglobin_last",
        "systolic_bp_last",
        "heart_rate_last",
        "discharge_disposition",
        "weekend_discharge",
        "followup_visit_scheduled",
        "post_discharge_call_logged",
        "readmitted_30d",
        ]
    schema_cols_25 = [c for c in schema_cols_26 if c != "readmitted_30d"]
    train_out = train_df[schema_cols_26]
    holdout_out = holdout_df[schema_cols_25]
    train_path = os.path.join(output_dir, "train_admissions.csv")
    holdout_path = os.path.join(output_dir, "holdout_admissions.csv")
    train_out.to_csv(train_path, index=False)
    holdout_out.to_csv(holdout_path, index=False)
    repeat_rate = (df["patient_id"].value_counts() > 1).mean()
    print(f"Generated Training Set: {train_out.shape} -> {train_path}")
    print(f"Generated Holdout Set:  {holdout_out.shape} -> {holdout_path}")
    print(f"Base 30d Readmission Rate (Train): {train_out['readmitted_30d'].mean():.2%}")
    print(f"Unique patients with multiple admissions: {repeat_rate:.2%}")
    print("Missing rates in labs/vitals (Train):")
    print(train_out[lab_vital_cols].isna().mean().round(4).to_dict())
if __name__ == "__main__":
    generate_hospital_data()

