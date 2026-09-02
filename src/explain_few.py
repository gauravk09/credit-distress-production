"""SHAP + LIME for a few borrowers on the CALIBRATED champion (v3).
Prints a per-borrower comparison and saves a SHAP waterfall for each."""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

from data import load_splits

model = joblib.load("src/model_calibrated.joblib")
X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()
cols = list(X_tr.columns)
SHORT = {c: c.replace("NumberOfTime", "late").replace("PastDueNotWorse", "")
          .replace("RevolvingUtilizationOfUnsecuredLines", "utilization")
          .replace("NumberOfTimes90DaysLate", "late90")
          .replace("NumberOfOpenCreditLinesAndLoans", "open_lines")
          .replace("NumberRealEstateLoansOrLines", "realestate")
          .replace("NumberOfDependents", "dependents")
          .replace("MonthlyIncome", "income").replace("DebtRatio", "debt") for c in cols}

f = lambda X: model.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]
background = X_tr.sample(80, random_state=0)[cols]
shap_expl = shap.explainers.Permutation(f, background.values)
lime_expl = LimeTabularExplainer(X_tr[cols].values, feature_names=cols,
                                 class_names=["no", "distress"], mode="classification",
                                 discretize_continuous=False, random_state=0)
lime_fn = lambda X: model.predict_proba(pd.DataFrame(X, columns=cols))

BORROWERS = {
    "risky":      [0.95, 24, 3, 0.8, 2000, 3, 1, 0, 2, 2],
    "safe":       [0.05, 55, 0, 0.15, 12000, 9, 0, 2, 0, 1],
    "borderline": [0.55, 40, 1, 0.45, 5000, 6, 0, 1, 0, 1],
}

for name, vals in BORROWERS.items():
    row = pd.DataFrame([dict(zip(cols, vals))], columns=cols)
    prob = float(f(row.values)[0])

    sv = shap_expl(row.values, max_evals=201, silent=True)
    shap_top = sorted(zip(cols, sv.values[0]), key=lambda z: -abs(z[1]))[:4]

    exp = lime_expl.explain_instance(np.array(vals, float), lime_fn, num_features=4, num_samples=3000)
    lime_top = exp.as_list()

    print(f"\n=== {name.upper()}  (calibrated prob = {prob:.3f}) ===")
    print("  SHAP top:", ", ".join(f"{SHORT[c]} {v:+.3f}" for c, v in shap_top))
    print("  LIME top:", ", ".join(f"{feat.split(' ')[0][:14]} {w:+.3f}" for feat, w in lime_top))

    shap.plots.waterfall(sv[0], show=False)
    plt.title(f"{name} — calibrated prob {prob:.2f}", fontsize=9)
    plt.tight_layout(); plt.savefig(f"explain/shap_{name}.png", dpi=105, bbox_inches="tight"); plt.close()

print("\nsaved explain/shap_risky.png, shap_safe.png, shap_borderline.png")
