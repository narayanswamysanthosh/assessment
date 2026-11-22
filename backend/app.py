import os
import sys
from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncGenerator, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import uvicorn
# Ensure the project root is importable for module-based and direct launches.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "code", "artifacts", "readmission_model.joblib")
DEFAULT_HOLDOUT_PATH = os.path.join(ROOT_DIR, "synthetic_data_generator", "data", "holdout_admissions.csv")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)

MODEL = None


def load_model() -> None:
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python code/run_assessment.py` first."
        )
    MODEL = joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    load_model()
    yield


app = FastAPI(
    title="Lakeshore Health Network — 30-Day Readmission Scoring API",
    description="Scores hospital admissions at discharge and builds the weekly nurse call list.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 1. Strict Input & Output Schemas (Pydantic Validation)
# ---------------------------------------------------------------------------
class AdmissionRecord(BaseModel):
    admission_id: str = Field(..., min_length=1, description="Unique admission ID")
    patient_id: str = Field(..., min_length=1, description="Pseudonymous patient ID")
    hospital_site: Literal["Site_A", "Site_B", "Site_C", "Site_D", "Site_E", "Site_F"]
    admit_date: date
    discharge_date: date
    age: int = Field(..., ge=0, le=120)
    sex: Literal["F", "M"]
    insurance_type: Literal["Commercial", "Medicare", "Medicaid", "Self-pay"]
    admission_source: Literal["Emergency", "Referral", "Transfer"]
    primary_dx_group: Literal[
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
    comorbidity_count: int = Field(..., ge=0, le=30)
    los_days: int = Field(..., description="Length of stay in days")
    n_prior_admissions_12m: int = Field(..., ge=0, le=50)
    n_ed_visits_6m: int = Field(..., ge=0, le=50)
    num_medications: int = Field(..., ge=0, le=60)
    bmi: float = Field(..., description="Body mass index at admission")
    lab_creatinine_last: Optional[float] = None
    lab_sodium_last: Optional[float] = None
    lab_hemoglobin_last: Optional[float] = None
    systolic_bp_last: Optional[float] = None
    heart_rate_last: Optional[float] = None
    discharge_disposition: Literal["Home", "Home with services", "Skilled nursing", "Rehab"]
    weekend_discharge: int = Field(..., ge=0, le=1)
    followup_visit_scheduled: int = Field(..., ge=0, le=1)
    # Optional if sent from historical files; ignored by model to prevent leakage
    post_discharge_call_logged: Optional[int] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def check_dates(self) -> "AdmissionRecord":
        if self.discharge_date < self.admit_date:
            raise ValueError("discharge_date cannot be earlier than admit_date")
        return self


class BatchScoreRequest(BaseModel):
    admissions: List[AdmissionRecord] = Field(..., min_length=1, max_length=25_000)


class ScoredAdmission(BaseModel):
    admission_id: str
    patient_id: str
    hospital_site: str
    discharge_date: str
    risk_score: float
    risk_tier: str
    reasons: List[str]


class CallListResponse(BaseModel):
    total_admissions_evaluated: int
    capacity_pct: float
    call_quota: int
    score_cutoff: float
    disclaimer: str
    call_list: List[ScoredAdmission]


# ---------------------------------------------------------------------------
# 2. Plain-Language Explanation Helper for the Care-Management Lead
# ---------------------------------------------------------------------------
def build_plain_language_reasons(row: pd.Series) -> List[str]:
    reasons: List[str] = []

    if row["n_prior_admissions_12m"] >= 2:
        reasons.append(f"{int(row['n_prior_admissions_12m'])} hospital admissions in the past 12 months")
    elif row["n_prior_admissions_12m"] == 1:
        reasons.append("1 prior hospital admission in the past 12 months")

    if row["n_ed_visits_6m"] >= 2:
        reasons.append(f"{int(row['n_ed_visits_6m'])} emergency visits in the past 6 months")

    if row["comorbidity_count"] >= 4:
        reasons.append(f"High chronic condition burden ({int(row['comorbidity_count'])} comorbidities)")

    if pd.notna(row["lab_creatinine_last"]) and row["lab_creatinine_last"] > 1.8:
        reasons.append(f"Elevated discharge creatinine ({row['lab_creatinine_last']:.1f} mg/dL)")

    if pd.notna(row["lab_hemoglobin_last"]) and row["lab_hemoglobin_last"] < 10.0:
        reasons.append(f"Low discharge hemoglobin ({row['lab_hemoglobin_last']:.1f} g/dL)")

    if pd.notna(row["lab_sodium_last"]) and row["lab_sodium_last"] < 134.0:
        reasons.append(f"Low discharge sodium ({row['lab_sodium_last']:.0f} mmol/L)")

    if int(row["followup_visit_scheduled"]) == 0:
        reasons.append("No follow-up appointment scheduled at discharge")

    if pd.to_datetime(row["discharge_date"]).day_of_week >= 5:
        reasons.append("Discharged over the weekend")

    if row["discharge_disposition"] == "Skilled nursing":
        reasons.append("Discharged to skilled nursing facility")

    if row["num_medications"] >= 10:
        reasons.append(f"Complex medication regimen ({int(row['num_medications'])} discharge meds)")

    if not reasons:
        reasons.append("Combined clinical profile and primary diagnosis risk")

    return reasons[:4]  # Keep top 4 concise bullets for nurse readability


def score_dataframe(df: pd.DataFrame) -> List[ScoredAdmission]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    probs = np.clip(MODEL.predict_proba(df)[:, 1], 0.0, 1.0)
    results: List[ScoredAdmission] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        score = round(float(probs[idx]), 4)
        if score >= 0.35:
            tier = "High"
        elif score >= 0.18:
            tier = "Moderate"
        else:
            tier = "Low"

        results.append(
            ScoredAdmission(
                admission_id=str(row["admission_id"]),
                patient_id=str(row["patient_id"]),
                hospital_site=str(row["hospital_site"]),
                discharge_date=str(row["discharge_date"]),
                risk_score=score,
                risk_tier=tier,
                reasons=build_plain_language_reasons(row),
            )
        )
    return results


# ---------------------------------------------------------------------------
# 3. Required Endpoints: Health, Score One, Score Batch, Ranked Top-K Call List
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_path": os.path.basename(MODEL_PATH),
    }


@app.post("/score/one", response_model=ScoredAdmission)
def score_one(admission: AdmissionRecord):
    df = pd.DataFrame([admission.model_dump()])
    return score_dataframe(df)[0]


@app.post("/score/batch", response_model=List[ScoredAdmission])
def score_batch(payload: BatchScoreRequest):
    # Keep inference bounded to 5,000 records per chunk.
    records = [rec.model_dump() for rec in payload.admissions]
    all_scored: List[ScoredAdmission] = []
    chunk_size = 5_000
    for start in range(0, len(records), chunk_size):
        chunk_df = pd.DataFrame(records[start : start + chunk_size])
        all_scored.extend(score_dataframe(chunk_df))
    return all_scored


@app.get("/call-list", response_model=CallListResponse)
def get_weekly_call_list(
    capacity_pct: float = Query(
        default=5.0,
        ge=1.0,
        le=25.0,
        description="Percentage of discharged patients the care team can call (default 5%).",
    ),
    week_start: Optional[str] = Query(
        default=None,
        description="Optional YYYY-MM-DD filter to slice a specific discharge week from holdout.",
    ),
    hospital_site: Optional[str] = Query(
        default=None,
        description="Optional hospital site filter (e.g., Site_A).",
    ),
):
    if not os.path.exists(DEFAULT_HOLDOUT_PATH):
        raise HTTPException(status_code=404, detail="Holdout admissions dataset not found.")

    df = pd.read_csv(DEFAULT_HOLDOUT_PATH)

    if hospital_site:
        df = df[df["hospital_site"] == hospital_site].copy()

    if week_start:
        start_dt = pd.to_datetime(week_start)
        end_dt = start_dt + pd.Timedelta(days=6)
        discharge_dt = pd.to_datetime(df["discharge_date"])
        df = df[(discharge_dt >= start_dt) & (discharge_dt <= end_dt)].copy()

    if df.empty:
        return CallListResponse(
            total_admissions_evaluated=0,
            capacity_pct=capacity_pct,
            call_quota=0,
            score_cutoff=0.0,
            disclaimer=(
                "Each score estimates baseline 30-day readmission risk; it is a triage aid, not a "
                "diagnosis or guarantee. Listed factors are rule-based review cues, not explanations "
                "of the model's prediction or evidence that outreach will prevent readmission."
            ),
            call_list=[],
            )
    scored = score_dataframe(df)
    scored.sort(key=lambda x: x.risk_score, reverse=True)
    quota = int(np.floor(len(scored) * (capacity_pct / 100.0)))
    top_calls = scored[:quota]
    cutoff = top_calls[-1].risk_score if top_calls else 0.0
    return CallListResponse(
    total_admissions_evaluated=len(scored),
    capacity_pct=capacity_pct,
    call_quota=quota,
    score_cutoff=cutoff,
    disclaimer=(
    "Each score estimates baseline 30-day readmission risk; it is a triage aid, not a "
    "diagnosis or guarantee. Listed factors are rule-based review cues, not explanations "
    "of the model's prediction or evidence that outreach will prevent readmission."
    ),
    call_list=top_calls,
    )
if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
