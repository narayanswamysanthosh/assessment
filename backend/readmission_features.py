import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HospitalFeatureCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_cols = [
            "age",
            "comorbidity_count",
            "los_days_clean",
            "n_prior_admissions_12m",
            "n_ed_visits_6m",
            "num_medications",
            "bmi",
            "lab_creatinine_last",
            "lab_sodium_last",
            "lab_hemoglobin_last",
            "systolic_bp_last",
            "heart_rate_last",
            "weekend_discharge_clean",
            "followup_visit_scheduled",
            "prior_utilization_total",
            "high_creatinine_flag",
            "low_hemoglobin_flag",
            "low_sodium_flag",
            "missing_vitals_count",
        ]
        self.cat_cols = [
            "hospital_site",
            "sex",
            "insurance_type",
            "admission_source",
            "primary_dx_group",
            "discharge_disposition",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        admit_dt = pd.to_datetime(df["admit_date"])
        discharge_dt = pd.to_datetime(df["discharge_date"])

        computed_los = (discharge_dt - admit_dt).dt.days.clip(lower=1)
        bad_los = (df["los_days"] <= 0) | (df["los_days"] > 180) | df["los_days"].isna()
        df["los_days_clean"] = np.where(bad_los, computed_los, df["los_days"])
        df["weekend_discharge_clean"] = (discharge_dt.dt.dayofweek >= 5).astype(int)

        df.loc[(df["bmi"] < 12.0) | (df["bmi"] > 80.0), "bmi"] = np.nan
        df.loc[(df["heart_rate_last"] < 25.0) | (df["heart_rate_last"] > 220.0), "heart_rate_last"] = np.nan
        df.loc[(df["systolic_bp_last"] < 50.0) | (df["systolic_bp_last"] > 260.0), "systolic_bp_last"] = np.nan

        lab_vital_cols = [
            "lab_creatinine_last",
            "lab_sodium_last",
            "lab_hemoglobin_last",
            "systolic_bp_last",
            "heart_rate_last",
        ]
        df["missing_vitals_count"] = df[lab_vital_cols].isna().sum(axis=1)
        df["prior_utilization_total"] = df["n_prior_admissions_12m"] + df["n_ed_visits_6m"]
        df["high_creatinine_flag"] = (df["lab_creatinine_last"] > 1.8).astype(float)
        df["low_hemoglobin_flag"] = (df["lab_hemoglobin_last"] < 10.0).astype(float)
        df["low_sodium_flag"] = (df["lab_sodium_last"] < 134.0).astype(float)

        return df[self.num_cols + self.cat_cols]