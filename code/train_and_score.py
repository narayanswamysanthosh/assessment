import os
import json
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.readmission_features import HospitalFeatureCleaner


# 2. Evaluation Metrics Aligned with Care-Management Capacity (Top 5%)
def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_frac: float = 0.05) -> float:
    n_top = max(1, int(np.floor(len(y_true) * k_frac)))
    top_idx = np.argsort(y_prob)[::-1][:n_top]
    return float(np.mean(y_true[top_idx]))


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_frac: float = 0.05) -> float:
    n_top = max(1, int(np.floor(len(y_true) * k_frac)))
    top_idx = np.argsort(y_prob)[::-1][:n_top]
    return float(np.sum(y_true[top_idx]) / max(1, np.sum(y_true)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            ece += (np.sum(mask) / len(y_true)) * abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
    return float(ece)


def build_pipelines():
    cleaner = HospitalFeatureCleaner()

    # Preprocessor for Linear Baseline (impute + scale + one-hot)
    linear_prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                ]),
                cleaner.num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cleaner.cat_cols,
            ),
        ]
    )

    baseline_model = Pipeline([
        ("clean", cleaner),
        ("prep", linear_prep),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

    # Preprocessor for Gradient Boosted Trees (handles NaNs natively + one-hot)
    tree_prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", cleaner.num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cleaner.cat_cols),
        ]
    )

    base_gbdt = Pipeline([
        ("clean", cleaner),
        ("prep", tree_prep),
        (
            "clf",
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=2.0,
                random_state=42,
            ),
        ),
    ])

    # Evaluate whether isotonic calibration improves probability quality.
    calibrated_gbdt = CalibratedClassifierCV(
        estimator=base_gbdt,
        method="isotonic",
        cv=3,
    )

    return baseline_model, base_gbdt, calibrated_gbdt


def run_pipeline(
    train_path: str = "synthetic_data_generator/data/train_admissions.csv",
    holdout_path: str = "synthetic_data_generator/data/holdout_admissions.csv",
    predictions_path: str = "predictions.csv",
    model_dir: str = "code/artifacts",
) -> None:
    os.makedirs(model_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)

    # 3. Temporal Validation Split (Train: 2021-2022, Validation: 2023)
    # Matches real deployment (predicting future discharges from historical records)
    admit_years = pd.to_datetime(train_df["admit_date"]).dt.year
    tr_mask = admit_years <= 2022
    va_mask = admit_years == 2023

    X_tr, y_tr = train_df[tr_mask].copy(), train_df.loc[tr_mask, "readmitted_30d"].values
    X_va, y_va = train_df[va_mask].copy(), train_df.loc[va_mask, "readmitted_30d"].values

    overlap_patients = len(set(X_tr["patient_id"]).intersection(set(X_va["patient_id"])))
    print(f"Temporal Split -> Train (2021-2022): {len(X_tr):,} rows | Val (2023): {len(X_va):,} rows")
    print(f"Repeat patients across Train/Val boundary (realistic hospital traffic): {overlap_patients:,}")
    print(f"Validation Base Readmission Rate: {y_va.mean():.2%}\n")

    baseline_model, uncal_gbdt, cal_gbdt = build_pipelines()

    models = {
        "1. Baseline (Logistic Regression)": baseline_model,
        "2. Uncalibrated HistGBDT": uncal_gbdt,
        "3. Calibrated HistGBDT": cal_gbdt,
    }

    report = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        va_probs = model.predict_proba(X_va)[:, 1]

        p_at_5 = precision_at_k(y_va, va_probs, k_frac=0.05)
        r_at_5 = recall_at_k(y_va, va_probs, k_frac=0.05)
        lift_5 = p_at_5 / y_va.mean()
        brier = brier_score_loss(y_va, va_probs)
        ece = expected_calibration_error(y_va, va_probs)
        auc = roc_auc_score(y_va, va_probs)

        report[name] = {
            "precision_at_5pct": round(p_at_5, 4),
            "lift_at_5pct": round(lift_5, 2),
            "recall_at_5pct": round(r_at_5, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "roc_auc": round(auc, 4),
        }

        print(f"--- {name} ---")
        print(f"  Precision@5% (Nurse Call Hit Rate): {p_at_5:.2%} ({lift_5:.2f}x lift over random)")
        print(f"  Recall@5% (Readmissions Caught):    {r_at_5:.2%}")
        print(f"  Brier Score (lower is better):      {brier:.4f}")
        print(f"  Calibration Error (ECE):            {ece:.4f}")
        print(f"  ROC-AUC (reference only):           {auc:.4f}\n")

    selected_model_name = min(
        report,
        key=lambda name: (
            -report[name]["precision_at_5pct"],
            report[name]["brier_score"],
            report[name]["ece"],
        ),
    )
    print(
        "Selected model (highest Precision@5%, then lowest Brier score and ECE): "
        f"{selected_model_name}"
    )

    va_probs_selected = models[selected_model_name].predict_proba(X_va)[:, 1]
    prob_true, prob_pred = calibration_curve(
        y_va,
        va_probs_selected,
        n_bins=8,
        strategy="quantile",
    )
    print("Calibration Check on 2023 Validation (Predicted Mean vs. Actual Readmission Rate):")
    for pred_m, true_m in zip(prob_pred, prob_true):
        print(f"  Predicted: {pred_m:.1%}  ->  Actual: {true_m:.1%}")

    # 4. Refit the selected model on all 2021-2023 data and score the holdout.
    print(f"\nRefitting {selected_model_name} on all 60,000 training admissions...")
    X_full = train_df.copy()
    y_full = train_df["readmitted_30d"].values

    final_candidates = dict(zip(models, build_pipelines()))
    final_model = final_candidates[selected_model_name]
    final_model.fit(X_full, y_full)
    model_artifact_path = os.path.join(model_dir, "readmission_model.joblib")
    metrics_artifact_path = os.path.join(model_dir, "validation_metrics.json")
    joblib.dump(final_model, model_artifact_path)
    with open(metrics_artifact_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # Score the holdout set and enforce the output contract.
    holdout_scores = final_model.predict_proba(holdout_df)[:, 1]
    holdout_scores = np.clip(holdout_scores, 0.0, 1.0)
    preds_df = pd.DataFrame({
        "admission_id": holdout_df["admission_id"],
        "risk_score": np.round(holdout_scores, 6),
    })
    # Integrity checks before writing predictions.csv
    assert len(preds_df) == len(holdout_df), "Row count mismatch!"
    assert list(preds_df.columns) == ["admission_id", "risk_score"], "Invalid columns!"
    assert preds_df["risk_score"].isna().sum() == 0, "NaN predictions detected!"
    assert ((preds_df["risk_score"] >= 0.0) & (preds_df["risk_score"] <= 1.0)).all(), "Scores outside [0, 1]!"
    preds_df.to_csv(predictions_path, index=False)
    top_5_cutoff = np.quantile(preds_df["risk_score"], 0.95)
    print(f"\nSaved {len(preds_df):,} holdout predictions -> {predictions_path}")
    print(f"Holdout Top-5% Risk Score Cutoff: {top_5_cutoff:.1%} (Mean score: {preds_df['risk_score'].mean():.2%})")


if __name__ == "__main__":
    run_pipeline()