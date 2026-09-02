"""LIME explanation for the SAME risky borrower, to compare against SHAP.

LIME jitters the borrower, scores the copies with the real model, and fits a
simple local line. Its slopes are the explanation.
"""
import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from data import load_splits
import clipper  # noqa: F401

model = joblib.load("src/model_clipped.joblib")
X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()
cols = list(model.feature_names_in_)

# LIME hands us plain numpy rows; wrap so our pipeline gets a named DataFrame.
def predict_fn(arr):
    return model.predict_proba(pd.DataFrame(arr, columns=cols))

explainer = LimeTabularExplainer(
    training_data=X_tr[cols].values,
    feature_names=cols,
    class_names=["no_distress", "distress"],
    mode="classification",
    discretize_continuous=False,   # our count features have zero-spread bins -> discretizer errors
    random_state=0,
)

borrower = {
    "RevolvingUtilizationOfUnsecuredLines": 0.95, "age": 24,
    "NumberOfTime30-59DaysPastDueNotWorse": 3, "DebtRatio": 0.8, "MonthlyIncome": 2000,
    "NumberOfOpenCreditLinesAndLoans": 3, "NumberOfTimes90DaysLate": 1,
    "NumberRealEstateLoansOrLines": 0, "NumberOfTime60-89DaysPastDueNotWorse": 2,
    "NumberOfDependents": 2,
}
row = np.array([borrower[c] for c in cols], dtype=float)

exp = explainer.explain_instance(row, predict_fn, num_features=10, num_samples=5000)
print("LIME local reasons (weight toward 'distress'):")
for feature, weight in exp.as_list():
    sign = "+" if weight >= 0 else "-"
    print(f"  {sign} {feature:<45} {weight:+.3f}")
