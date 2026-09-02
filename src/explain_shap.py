"""Compute the SHAP receipt for ONE borrower on our live (clipped) model.

- background = a sample of training rows (defines what "off / typical" means)
- explain the risky borrower we've been using
- print the itemized contributions and save a waterfall plot
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_splits
import clipper  # noqa: F401  (needed to unpickle PercentileClipper)

model = joblib.load("src/model_clipped.joblib")
X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()
cols = list(model.feature_names_in_)

# what the model outputs: probability of distress
f = lambda X: model.predict_proba(X)[:, 1]

# background: 100 typical borrowers -> defines the "average" the receipt starts from
background = X_tr.sample(100, random_state=0)[cols]

# the borrower to explain (the risky one from our API tests)
borrower = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": 0.95, "age": 24,
    "NumberOfTime30-59DaysPastDueNotWorse": 3, "DebtRatio": 0.8, "MonthlyIncome": 2000,
    "NumberOfOpenCreditLinesAndLoans": 3, "NumberOfTimes90DaysLate": 1,
    "NumberRealEstateLoansOrLines": 0, "NumberOfTime60-89DaysPastDueNotWorse": 2,
    "NumberOfDependents": 2,
}], columns=cols)

explainer = shap.Explainer(f, background)
sv = explainer(borrower)

base = float(sv.base_values[0])
contribs = sv.values[0]
pred = float(f(borrower)[0])

print(f"average borrower (base) : {base:.3f}")
order = np.argsort(-np.abs(contribs))
for i in order:
    sign = "+" if contribs[i] >= 0 else "-"
    print(f"  {sign} {cols[i]:<38} {contribs[i]:+.3f}   (value={borrower.iloc[0, i]})")
print(f"= this borrower          : {pred:.3f}")
print(f"check base+sum           : {base + contribs.sum():.3f}")

os.makedirs("explain", exist_ok=True)
shap.plots.waterfall(sv[0], show=False)
plt.tight_layout(); plt.savefig("explain/shap_borrower.png", dpi=110, bbox_inches="tight")
print("saved explain/shap_borrower.png")
