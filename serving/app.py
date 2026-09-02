"""Self-contained deployment service (no MLflow dependency).

Loads the bundled champion snapshot (model.joblib + meta.json) and serves:
  GET  /          -> the demo frontend (index.html)
  GET  /health    -> model version + threshold
  POST /predict   -> probability, decision, and (with ?explain=true) a SHAP receipt

Input validation rejects impossible values (422). None -> NaN coercion at the boundary.
"""
import os, json, pathlib
import numpy as np
import pandas as pd
import shap
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).parent
FIELD_TO_COLUMN = {
    "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines", "age": "age",
    "times_30_59_days_late": "NumberOfTime30-59DaysPastDueNotWorse", "debt_ratio": "DebtRatio",
    "monthly_income": "MonthlyIncome", "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "times_90_days_late": "NumberOfTimes90DaysLate", "real_estate_loans": "NumberRealEstateLoansOrLines",
    "times_60_89_days_late": "NumberOfTime60-89DaysPastDueNotWorse", "dependents": "NumberOfDependents",
}
MODEL_COLS = list(FIELD_TO_COLUMN.values())
COLUMN_TO_FIELD = {c: f for f, c in FIELD_TO_COLUMN.items()}


class BorrowerFeatures(BaseModel):
    revolving_utilization: float = Field(..., ge=0, examples=[0.5])
    age: int = Field(..., ge=0, le=120, examples=[45])
    times_30_59_days_late: int = Field(..., ge=0, examples=[0])
    debt_ratio: float = Field(..., ge=0, examples=[0.3])
    monthly_income: float | None = Field(None, ge=0, examples=[6000])
    open_credit_lines: int = Field(..., ge=0, examples=[8])
    times_90_days_late: int = Field(..., ge=0, examples=[0])
    real_estate_loans: int = Field(..., ge=0, examples=[1])
    times_60_89_days_late: int = Field(..., ge=0, examples=[0])
    dependents: float | None = Field(None, ge=0, examples=[2])


model = joblib.load(HERE / "model.joblib")
meta = json.load(open(HERE / "meta.json"))
THRESHOLD = meta["threshold"]
_bg = pd.read_csv(HERE / "background.csv")[MODEL_COLS]
_predict = lambda arr: model.predict_proba(pd.DataFrame(arr, columns=MODEL_COLS))[:, 1]
explainer = shap.explainers.Permutation(_predict, _bg.values)
explainer(_bg.values[:1], max_evals=201, silent=True)   # warm up

app = FastAPI(title="Credit distress model")


@app.get("/", response_class=HTMLResponse)
def home():
    return (HERE / "index.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model"], "version": meta["version"], "threshold": THRESHOLD}


@app.post("/predict")
def predict(features: BorrowerFeatures, explain: bool = False):
    row = {c: getattr(features, f) for f, c in FIELD_TO_COLUMN.items()}
    X = pd.DataFrame([row], columns=MODEL_COLS).apply(pd.to_numeric, errors="coerce")
    proba = float(model.predict_proba(X)[0, 1])
    out = {"probability": round(proba, 4), "threshold": THRESHOLD,
           "decision": "flag" if proba >= THRESHOLD else "clear"}
    if explain:
        sv = explainer(X.values, max_evals=201, silent=True)
        out["explanation"] = {
            "base_value": round(float(sv.base_values[0]), 4),
            "contributions": [{"feature": COLUMN_TO_FIELD[MODEL_COLS[i]],
                               "value": None if pd.isna(X.iloc[0, i]) else float(X.iloc[0, i]),
                               "shap": round(float(sv.values[0][i]), 4)}
                              for i in np.argsort(-np.abs(sv.values[0]))],
        }
    return out
