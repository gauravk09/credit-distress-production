"""Streamlit demo for the credit-distress champion (free on Streamlit Community Cloud).

Loads the bundled champion snapshot, scores a borrower, shows a SHAP waterfall.
Rejects physically-impossible inputs (negatives, absurd age).
"""
import json, pathlib
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

HERE = pathlib.Path(__file__).parent
COLS = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]
LABELS = ["Revolving utilization", "Age", "Times 30–59 days late", "Debt ratio", "Monthly income",
          "Open credit lines", "Times 90+ days late", "Real-estate loans", "Times 60–89 days late",
          "Dependents"]
DEFAULTS = [0.35, 42, 0, 0.4, 5000.0, 7, 0, 1, 0, 1]


@st.cache_resource
def load():
    model = joblib.load(HERE / "model.joblib")
    meta = json.load(open(HERE / "meta.json"))
    bg = pd.read_csv(HERE / "background.csv")[COLS]
    predict = lambda arr: model.predict_proba(pd.DataFrame(arr, columns=COLS))[:, 1]
    explainer = shap.explainers.Permutation(predict, bg.values)
    explainer(bg.values[:1], max_evals=201, silent=True)   # warm up
    return model, meta, explainer


model, meta, explainer = load()
THRESHOLD = meta["threshold"]

st.title("Credit Distress Model")
st.caption("Calibrated XGBoost + SHAP explanations · champion v%s · threshold %.1f%%"
           % (meta["version"], THRESHOLD * 100))

vals, cols = [], st.columns(2)
for i, (label, default) in enumerate(zip(LABELS, DEFAULTS)):
    with cols[i % 2]:
        vals.append(st.number_input(label, value=default, step=1.0))

if st.button("Score borrower", type="primary"):
    if any(v < 0 for v in vals):
        st.error("Inputs must be ≥ 0 — impossible values are rejected, not silently clamped.")
    elif vals[1] > 120:
        st.error("Age must be ≤ 120.")
    else:
        X = pd.DataFrame([dict(zip(COLS, vals))], columns=COLS)
        proba = float(model.predict_proba(X)[0, 1])
        decision = "🚩 FLAG" if proba >= THRESHOLD else "✅ CLEAR"
        st.metric(f"Risk of distress — {decision}", f"{proba*100:.1f}%",
                  help=f"threshold {THRESHOLD*100:.1f}%")

        sv = explainer(X.values, max_evals=201, silent=True)
        order = np.argsort(np.abs(sv.values[0]))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.barh([LABELS[i] for i in order], [sv.values[0][i] for i in order],
                color=["#d62728" if sv.values[0][i] >= 0 else "#2ca02c" for i in order])
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title("Why (SHAP): red raises risk, green lowers")
        ax.set_xlabel("contribution to probability")
        fig.tight_layout()
        st.pyplot(fig)
