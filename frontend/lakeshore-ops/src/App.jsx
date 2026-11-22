import React, { useState, useEffect } from "react";
import "./App.css";

const SITES = [
  "All Sites",
  "Site_A",
  "Site_B",
  "Site_C",
  "Site_D",
  "Site_E",
  "Site_F",
];

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/$/, "");

const formatDate = (value) => {
  if (!value) return "";
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
};

const INITIAL_ADMISSION = {
  admission_id: "",
  patient_id: "",
  hospital_site: "Site_A",
  admit_date: "2024-01-01",
  discharge_date: "2024-01-05",
  age: 65,
  sex: "F",
  insurance_type: "Medicare",
  admission_source: "Emergency",
  primary_dx_group: "Heart_Failure",
  comorbidity_count: 3,
  los_days: 4,
  n_prior_admissions_12m: 1,
  n_ed_visits_6m: 1,
  num_medications: 8,
  bmi: 28,
  lab_creatinine_last: "",
  lab_sodium_last: "",
  lab_hemoglobin_last: "",
  systolic_bp_last: "",
  heart_rate_last: "",
  discharge_disposition: "Home",
  weekend_discharge: 0,
  followup_visit_scheduled: 1,
};

function SinglePatientView() {
  const [admission, setAdmission] = useState(INITIAL_ADMISSION);
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateField = (event) => {
    const { name, value, type, checked } = event.target;
    if (type === "checkbox") {
      setAdmission((previous) => ({ ...previous, [name]: checked ? 1 : 0 }));
      return;
    }

    const numericFields = new Set([
      "age",
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
    ]);

    setAdmission((previous) => ({
      ...previous,
      [name]: numericFields.has(name)
        ? value === ""
          ? ""
          : Number(value)
        : value,
    }));
  };

  const scorePatient = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setScore(null);

    const payload = Object.fromEntries(
      Object.entries(admission).map(([key, value]) => [
        key,
        value === "" ? null : value,
      ]),
    );

    try {
      const response = await fetch(`${API_BASE_URL}/score/one`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const details = await response.json().catch(() => null);
        throw new Error(details?.detail || `API error (${response.status})`);
      }
      setScore(await response.json());
    } catch (err) {
      setError(err.message || "Unable to score the patient.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section>
      <form className="single-score-card" onSubmit={scorePatient}>
        <h2>Single Patient Entry</h2>
        <p className="helper-text">
          Enter discharge information and send it to the score-one API.
        </p>

        <div className="form-grid">
          <div className="control-group">
            <label htmlFor="admission_id">Admission ID</label>
            <input id="admission_id" name="admission_id" value={admission.admission_id} onChange={updateField} required />
          </div>
          <div className="control-group">
            <label htmlFor="patient_id">Patient ID</label>
            <input id="patient_id" name="patient_id" value={admission.patient_id} onChange={updateField} required />
          </div>
          <div className="control-group">
            <label htmlFor="hospital_site">Hospital Site</label>
            <select id="hospital_site" name="hospital_site" value={admission.hospital_site} onChange={updateField}>
              {SITES.filter((site) => site !== "All Sites").map((site) => <option key={site}>{site}</option>)}
            </select>
          </div>
          <div className="control-group">
            <label htmlFor="admit_date">Admit Date</label>
            <input id="admit_date" name="admit_date" type="date" value={admission.admit_date} onChange={updateField} required />
          </div>
          <div className="control-group">
            <label htmlFor="discharge_date">Discharge Date</label>
            <input id="discharge_date" name="discharge_date" type="date" value={admission.discharge_date} onChange={updateField} required />
          </div>
          <div className="control-group">
            <label htmlFor="age">Age</label>
            <input id="age" name="age" type="number" min="0" max="120" value={admission.age} onChange={updateField} required />
          </div>
          <div className="control-group">
            <label htmlFor="sex">Sex</label>
            <select id="sex" name="sex" value={admission.sex} onChange={updateField}>
              <option value="F">F</option><option value="M">M</option>
            </select>
          </div>
          <div className="control-group">
            <label htmlFor="insurance_type">Insurance</label>
            <select id="insurance_type" name="insurance_type" value={admission.insurance_type} onChange={updateField}>
              {["Commercial", "Medicare", "Medicaid", "Self-pay"].map((value) => <option key={value}>{value}</option>)}
            </select>
          </div>
          <div className="control-group">
            <label htmlFor="admission_source">Admission Source</label>
            <select id="admission_source" name="admission_source" value={admission.admission_source} onChange={updateField}>
              {["Emergency", "Referral", "Transfer"].map((value) => <option key={value}>{value}</option>)}
            </select>
          </div>
          <div className="control-group">
            <label htmlFor="primary_dx_group">Primary Diagnosis Group</label>
            <select id="primary_dx_group" name="primary_dx_group" value={admission.primary_dx_group} onChange={updateField}>
              {["Heart_Failure", "COPD_Asthma", "Pneumonia_Sepsis", "GI_Disorder", "Renal_Failure", "Diabetes_Metabolic", "Arrhythmia", "Orthopedic", "Stroke_Neuro", "Oncology", "Behavioral_Health", "Other_Medical"].map((value) => <option key={value}>{value}</option>)}
            </select>
          </div>
          {[
            ["comorbidity_count", "Comorbidity Count", 0, 30],
            ["los_days", "Length of Stay (days)", 0, null],
            ["n_prior_admissions_12m", "Prior Admissions (12 months)", 0, 50],
            ["n_ed_visits_6m", "ED Visits (6 months)", 0, 50],
            ["num_medications", "Discharge Medications", 0, 60],
            ["bmi", "BMI", 0, null],
          ].map(([name, label, min, max]) => (
            <div className="control-group" key={name}>
              <label htmlFor={name}>{label}</label>
              <input id={name} name={name} type="number" min={min} max={max ?? undefined} step={name === "bmi" ? "0.1" : "1"} value={admission[name]} onChange={updateField} required />
            </div>
          ))}
          {[
            ["lab_creatinine_last", "Creatinine", "0.1"],
            ["lab_sodium_last", "Sodium", "1"],
            ["lab_hemoglobin_last", "Hemoglobin", "0.1"],
            ["systolic_bp_last", "Systolic BP", "1"],
            ["heart_rate_last", "Heart Rate", "1"],
          ].map(([name, label, step]) => (
            <div className="control-group" key={name}>
              <label htmlFor={name}>{label} (Optional)</label>
              <input id={name} name={name} type="number" step={step} value={admission[name]} onChange={updateField} />
            </div>
          ))}
          <div className="control-group">
            <label htmlFor="discharge_disposition">Discharge Disposition</label>
            <select id="discharge_disposition" name="discharge_disposition" value={admission.discharge_disposition} onChange={updateField}>
              {["Home", "Home with services", "Skilled nursing", "Rehab"].map((value) => <option key={value}>{value}</option>)}
            </select>
          </div>
          <label className="checkbox-group">
            <input name="followup_visit_scheduled" type="checkbox" checked={admission.followup_visit_scheduled === 1} onChange={updateField} />
            Follow-up visit scheduled
          </label>
        </div>

        <button className="score-submit-btn" type="submit" disabled={loading || !admission.admission_id || !admission.patient_id}>
          {loading ? "Scoring..." : "Score Patient"}
        </button>
      </form>

      {error && <div className="state-msg error" role="alert">{error}</div>}
      {score && (
        <section className="single-result-card" aria-live="polite">
          <h2>Readmission Risk Result</h2>
          <div className="single-result-grid">
            <div><span className="kpi-label">Risk Score</span><strong>{(score.risk_score * 100).toFixed(1)}%</strong></div>
            <div><span className="kpi-label">Risk Tier</span><strong>{score.risk_tier}</strong></div>
            <div><span className="kpi-label">Hospital Site</span><strong>{score.hospital_site}</strong></div>
            <div><span className="kpi-label">Discharge Date</span><strong>{formatDate(score.discharge_date)}</strong></div>
          </div>
          <div className="result-reasons">
            <span className="kpi-label">Clinical Signals to Review</span>
            <p className="helper-text">
              These rule-based signals do not explain the model score.
            </p>
            <ul>{score.reasons.map((reason, index) => <li key={index}>{reason}</li>)}</ul>
          </div>
        </section>
      )}
    </section>
  );
}

export default function App() {
  const [activeView, setActiveView] = useState("dashboard");
  const [capacityPct, setCapacityPct] = useState(5.0);
  const [selectedSite, setSelectedSite] = useState("All Sites");
  const [weekStart, setWeekStart] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [calledIds, setCalledIds] = useState({});

  const fetchCallList = async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        capacity_pct: capacityPct.toString(),
      });
      if (selectedSite !== "All Sites") {
        params.append("hospital_site", selectedSite);
      }
      if (weekStart) {
        params.append("week_start", weekStart);
      }

      const response = await fetch(
        `${API_BASE_URL}/call-list?${params.toString()}`,
      );
      if (!response.ok) {
        throw new Error(
          `API error (${response.status}): Could not load call list.`,
        );
      }
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err.message || "Unable to connect to scoring service.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCallList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capacityPct, selectedSite, weekStart]);

  const toggleCalled = (admissionId) => {
    setCalledIds((prev) => ({
      ...prev,
      [admissionId]: !prev[admissionId],
    }));
  };

  const completedCount = data
    ? data.call_list.filter((item) => calledIds[item.admission_id]).length
    : 0;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div>
          <h1>Lakeshore Health Network</h1>
          <p className="subtitle">
            Care-Management Weekly Post-Discharge Call Prioritization
          </p>
        </div>
        <div className="status-pill">
          Outreach Progress: <strong>{completedCount}</strong> /{" "}
          {data ? data.call_quota : 0} Called
        </div>
      </header>

      <nav className="view-nav" aria-label="Application views">
        <button
          type="button"
          className={activeView === "dashboard" ? "active" : ""}
          onClick={() => setActiveView("dashboard")}
        >
          Weekly Call List
        </button>
        <button
          type="button"
          className={activeView === "single" ? "active" : ""}
          onClick={() => setActiveView("single")}
        >
          Single Patient Entry
        </button>
      </nav>

      {activeView === "single" ? (
        <SinglePatientView />
      ) : (
        <>

      {/* Mandatory Visible Disclaimer for Clinical Trust */}
      <section className="disclaimer-box" role="alert">
        <strong>How to Read These Scores (Important):</strong>{" "}
        {data?.disclaimer ||
          "Each score estimates the probability (0% to 100%) that a patient will have an unplanned readmission within 30 days based on discharge records. It is a triage aid—not a clinical diagnosis—and does not guarantee a patient will or will not return, nor that a phone call alone will prevent readmission."}
      </section>

      {/* Controls: Staffing Capacity Slider + Site & Week Filters */}
      <section className="controls-card">
        <div className="control-group slider-group">
          <label htmlFor="capacity-slider">
            Team Staffing Capacity: <strong>{capacityPct.toFixed(1)}%</strong>{" "}
            of Discharges
          </label>
          <input
            id="capacity-slider"
            type="range"
            min="1"
            max="15"
            step="0.5"
            value={capacityPct}
            onChange={(e) => setCapacityPct(parseFloat(e.target.value))}
          />
          <span className="helper-text">
            Defaults to 5%. Adjust from 1% to 15% to match this week&apos;s
            available outreach capacity.
          </span>
        </div>

        <div className="control-group">
          <label htmlFor="site-select">Hospital Site</label>
          <select
            id="site-select"
            value={selectedSite}
            onChange={(e) => setSelectedSite(e.target.value)}
            aria-label="Hospital site"
          >
            {SITES.map((site) => (
              <option key={site} value={site}>
                {site}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="week-input">Discharge Week Starting (Optional)</label>
          <div className="week-row">
            <input
              id="week-input"
              type="date"
              min="2024-01-01"
              max="2024-03-31"
              value={weekStart}
              onChange={(e) => setWeekStart(e.target.value)}
              aria-label="Discharge week starting date"
            />
            {weekStart && (
              <button
                type="button"
                className="clear-btn"
                onClick={() => setWeekStart("")}
              >
                All Q1
              </button>
            )}
          </div>
        </div>
      </section>

      {(selectedSite !== "All Sites" || weekStart) && (
        <div className="active-filters" aria-live="polite">
          <strong>Viewing:</strong>
          {selectedSite !== "All Sites" && (
            <span>Hospital site: {selectedSite}</span>
          )}
          {weekStart && (
            <span>
              Discharge week: {formatDate(weekStart)} through{" "}
              {formatDate(
                new Date(
                  new Date(`${weekStart}T00:00:00`).getTime() +
                    6 * 24 * 60 * 60 * 1000,
                ).toISOString().slice(0, 10),
              )}
            </span>
          )}
        </div>
      )}

      {/* Summary KPIs */}
      {data && (
        <section className="kpi-grid">
          <div className="kpi-card">
            <span className="kpi-label">Discharges Evaluated</span>
            <span className="kpi-value">
              {data.total_admissions_evaluated.toLocaleString()}
            </span>
          </div>
          <div className="kpi-card highlight">
            <span className="kpi-label">
              Patients on Call List ({data.capacity_pct}%)
            </span>
            <span className="kpi-value">
              {data.call_quota.toLocaleString()}
            </span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Minimum Risk Cutoff</span>
            <span className="kpi-value">
              {(data.score_cutoff * 100).toFixed(1)}%
            </span>
          </div>
        </section>
      )}

      {/* Ranked Call List Table */}
      <section className="table-card">
        {loading && (
          <div className="state-msg">Scoring discharged patients...</div>
        )}
        {error && <div className="state-msg error">{error}</div>}

        {!loading && !error && data && (
          <>
            {data.call_list.length === 0 ? (
              <div className="state-msg">
                {data.total_admissions_evaluated === 0
                  ? "No discharges found for the selected filter."
                  : "No calls fit the selected capacity for this filtered group."}
              </div>
            ) : (
              <table className="call-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Patient / Admission</th>
                    <th>Site & Discharge</th>
                    <th>30-Day Return Risk</th>
                    <th>Clinical Signals to Review</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {data.call_list.map((row, idx) => {
                    const isCalled = !!calledIds[row.admission_id];
                    const pct = (row.risk_score * 100).toFixed(1);
                    return (
                      <tr
                        key={row.admission_id}
                        className={isCalled ? "row-done" : ""}
                      >
                        <td className="rank-col">#{idx + 1}</td>
                        <td>
                          <div className="patient-id">{row.patient_id}</div>
                          <div className="adm-id">{row.admission_id}</div>
                        </td>
                        <td>
                          <div>{row.hospital_site}</div>
                          <div className="adm-id">
                            Discharged {formatDate(row.discharge_date)}
                          </div>
                        </td>
                        <td>
                          <span
                            className={`risk-badge tier-${row.risk_tier.toLowerCase()}`}
                          >
                            {pct}% ({row.risk_tier})
                          </span>
                        </td>
                        <td>
                          <ul className="reasons-list">
                            {row.reasons.map((reason, rIdx) => (
                              <li key={rIdx}>{reason}</li>
                            ))}
                          </ul>
                        </td>
                        <td>
                          <button
                            type="button"
                            className={`action-btn ${isCalled ? "called" : ""}`}
                            onClick={() => toggleCalled(row.admission_id)}
                          >
                            {isCalled ? "Called ✓" : "Mark Called"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </>
        )}
      </section>
        </>
      )}
    </div>
  );
}
