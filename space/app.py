"""Gradio demo for the credit-distress champion (free HF Space, no Docker).

Loads the bundled champion snapshot, scores a borrower, and shows a SHAP waterfall.
Rejects physically-impossible inputs (negatives, absurd age).
"""
import json, pathlib
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import spaces   # ZeroGPU: the app must expose at least one @spaces.GPU function

HERE = pathlib.Path(__file__).parent
COLS = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]
LABELS = ["Revolving utilization", "Age", "Times 30–59 days late", "Debt ratio", "Monthly income",
          "Open credit lines", "Times 90+ days late", "Real-estate loans", "Times 60–89 days late",
          "Dependents"]

model = joblib.load(HERE / "model.joblib")
meta = json.load(open(HERE / "meta.json"))
THRESHOLD = meta["threshold"]
_bg = pd.read_csv(HERE / "background.csv")[COLS]
_predict = lambda arr: model.predict_proba(pd.DataFrame(arr, columns=COLS))[:, 1]
explainer = shap.explainers.Permutation(_predict, _bg.values)
explainer(_bg.values[:1], max_evals=201, silent=True)   # warm up


@spaces.GPU   # a GPU function must EXIST for ZeroGPU to start; this stub is never called.
def _zerogpu_stub():
    return None


def score(*vals):   # real inference runs on CPU (sklearn/xgboost/shap need no GPU)
    if any(v is not None and v < 0 for v in vals):
        raise gr.Error("Inputs must be ≥ 0 (impossible values are rejected, not clamped).")
    if vals[1] is not None and vals[1] > 120:
        raise gr.Error("Age must be ≤ 120.")

    X = pd.DataFrame([dict(zip(COLS, vals))], columns=COLS).apply(pd.to_numeric, errors="coerce")
    proba = float(model.predict_proba(X)[0, 1])
    decision = "🚩 FLAG" if proba >= THRESHOLD else "✅ CLEAR"
    verdict = (f"## {proba*100:.1f}% risk — {decision}\n"
               f"threshold {THRESHOLD*100:.1f}%")

    sv = explainer(X.values, max_evals=201, silent=True)
    order = np.argsort(np.abs(sv.values[0]))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([LABELS[i] for i in order], [sv.values[0][i] for i in order],
            color=["#d62728" if sv.values[0][i] >= 0 else "#2ca02c" for i in order])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title("Why (SHAP): red raises risk, green lowers")
    ax.set_xlabel("contribution to probability")
    fig.tight_layout()
    return verdict, fig


DEFAULTS = [0.35, 42, 0, 0.4, 5000, 7, 0, 1, 0, 1]
RISKY = [0.95, 24, 3, 0.8, 2000, 3, 1, 0, 2, 2]

with gr.Blocks(title="Credit Distress Model") as demo:
    gr.Markdown("# Credit Distress Model\n"
                "Calibrated XGBoost + SHAP explanations. Enter a borrower and score them.")
    inputs = []
    for r in range(0, 10, 5):
        with gr.Row():
            for i in range(r, r + 5):
                inputs.append(gr.Number(label=LABELS[i], value=DEFAULTS[i]))
    btn = gr.Button("Score borrower", variant="primary")
    with gr.Row():
        verdict = gr.Markdown()
        plot = gr.Plot()
    btn.click(score, inputs=inputs, outputs=[verdict, plot])
    gr.Examples(examples=[DEFAULTS, RISKY], inputs=inputs, label="Try: typical / risky")

if __name__ == "__main__":
    demo.launch()
