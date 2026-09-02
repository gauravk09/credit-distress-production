"""Live prediction server.

Wraps the trained pipeline behind an HTTP endpoint so anything can ask for a
prediction. The SAME pipeline (fill -> scale -> model) that we trained runs here,
loaded from disk once at startup.

Run:  uvicorn src.serve:app --port 8000
Then: POST /predict with the 10 borrower features (snake_case, see BorrowerFeatures).
"""
import os
import sys
import json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(__file__))   # so MLflow can find clipper.PercentileClipper
import clipper  # noqa: F401  (registers PercentileClipper for unpickling)
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_NAME = "credit-distress"
MODEL_ALIAS = "champion"

LOG_PATH = "logs/requests.jsonl"   # one JSON object per line, append-only

# Snake-case API field  ->  exact model column name (order matters for the model).
FIELD_TO_COLUMN = {
    "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines",
    "age": "age",
    "times_30_59_days_late": "NumberOfTime30-59DaysPastDueNotWorse",
    "debt_ratio": "DebtRatio",
    "monthly_income": "MonthlyIncome",
    "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "times_90_days_late": "NumberOfTimes90DaysLate",
    "real_estate_loans": "NumberRealEstateLoansOrLines",
    "times_60_89_days_late": "NumberOfTime60-89DaysPastDueNotWorse",
    "dependents": "NumberOfDependents",
}
MODEL_COLS = list(FIELD_TO_COLUMN.values())
COLUMN_TO_FIELD = {col: field for field, col in FIELD_TO_COLUMN.items()}


class BorrowerFeatures(BaseModel):
    # Validation at the boundary: reject the physically IMPOSSIBLE (ge=0, sane age).
    # Unusual-but-real values (huge debt ratio, high utilization) are allowed through
    # on purpose -- they exist in the data; only the impossible is rejected with a 422.
    revolving_utilization: float = Field(..., ge=0, examples=[0.5])
    age: int = Field(..., ge=0, le=120, examples=[45])
    times_30_59_days_late: int = Field(..., ge=0, examples=[0])
    debt_ratio: float = Field(..., ge=0, examples=[0.3])
    monthly_income: float | None = Field(None, ge=0, examples=[6000])   # may be missing -> imputed
    open_credit_lines: int = Field(..., ge=0, examples=[8])
    times_90_days_late: int = Field(..., ge=0, examples=[0])
    real_estate_loans: int = Field(..., ge=0, examples=[1])
    times_60_89_days_late: int = Field(..., ge=0, examples=[0])
    dependents: float | None = Field(None, ge=0, examples=[2])           # may be missing -> imputed


# Load the @champion model from the MLflow registry ONCE at startup.
# Promoting a new model = move the @champion alias in MLflow; no code change here.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
_mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
THRESHOLD = float(_mv.tags["threshold"])   # threshold travels with the model version
MODEL_VERSION = _mv.version

# Build the SHAP explainer ONCE at startup (background = typical borrowers).
# Model-agnostic (sampling) so it survives a champion swap to any model type.
_bg = pd.read_csv("data/credit.csv")[MODEL_COLS].sample(50, random_state=0)
_predict_fn = lambda arr: model.predict_proba(pd.DataFrame(arr, columns=MODEL_COLS))[:, 1]
explainer = shap.explainers.Permutation(_predict_fn, _bg.values)  # bounded cost per request
explainer(_bg.values[:1], max_evals=201, silent=True)             # warm up (avoid ~5s first-call)

app = FastAPI(title="Credit distress model")


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD,
            "model": f"{MODEL_NAME}@{MODEL_ALIAS}", "version": MODEL_VERSION}


@app.post("/predict")
def predict(features: BorrowerFeatures, explain: bool = False):
    # Rebuild a one-row DataFrame with the exact column names/order the model wants.
    row = {col: getattr(features, field) for field, col in FIELD_TO_COLUMN.items()}
    X = pd.DataFrame([row], columns=list(FIELD_TO_COLUMN.values()))
    # Coerce None -> NaN at the serving boundary so ANY champion (with or without a
    # clipper) gets clean float columns. Model-agnostic: survives a champion swap.
    X = X.apply(pd.to_numeric, errors="coerce")

    proba = float(model.predict_proba(X)[0, 1])
    decision = "flag" if proba >= THRESHOLD else "clear"

    _log_request(row, proba, decision)
    result = {"probability": round(proba, 4), "threshold": THRESHOLD, "decision": decision}
    if explain:                       # SHAP only when asked -> score-only calls stay fast
        result["explanation"] = _explain(X)
    return result


def _explain(X):
    """SHAP receipt for this one borrower: base value + per-feature contributions
    (snake_case names), sorted by impact. base + sum(shap) == probability."""
    sv = explainer(X.values, max_evals=201, silent=True)
    base = float(sv.base_values[0])
    vals = sv.values[0]
    contribs = [
        {"feature": COLUMN_TO_FIELD[MODEL_COLS[i]],
         "value": float(X.iloc[0, i]) if pd.notna(X.iloc[0, i]) else None,
         "shap": round(float(vals[i]), 4)}
        for i in np.argsort(-np.abs(vals))
    ]
    return {"base_value": round(base, 4), "contributions": contribs}


def _log_request(row, proba, decision):
    """Append one line: the raw inputs + what we predicted + when."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": row,                       # model-column-named inputs
        "probability": round(proba, 4),
        "decision": decision,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
