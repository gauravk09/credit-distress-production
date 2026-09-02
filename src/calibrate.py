"""Calibrate the champion: fit an isotonic corrector on validation (data the model
didn't train on), then measure honesty on the untouched TEST set.
Calibration changes the probabilities, NOT the ranking (PR-AUC is unchanged)."""
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, average_precision_score

from data import load_splits

model = joblib.load("src/model_xgb.joblib")
X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()

# fit the corrector on validation (prefit = don't retrain the model, only the calibrator)
calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
calibrated.fit(X_val, y_val)

# honesty check on the untouched TEST set
p_raw = model.predict_proba(X_te)[:, 1]
p_cal = calibrated.predict_proba(X_te)[:, 1]

print(f"actual default rate (test) : {y_te.mean():.3f}")
print(f"avg predicted  BEFORE      : {p_raw.mean():.3f}   Brier {brier_score_loss(y_te, p_raw):.4f}")
print(f"avg predicted  AFTER       : {p_cal.mean():.3f}   Brier {brier_score_loss(y_te, p_cal):.4f}")
print(f"PR-AUC before/after        : {average_precision_score(y_te, p_raw):.3f} / "
      f"{average_precision_score(y_te, p_cal):.3f}   (ranking unchanged)")

# before vs after reliability curves
fig, ax = plt.subplots(figsize=(5.4, 5))
ax.plot([0, 1], [0, 1], "--", c="grey", label="perfect")
for p, name, c in [(p_raw, "before", "#d62728"), (p_cal, "after (isotonic)", "#2ca02c")]:
    fp, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", c=c, label=name)
ax.set(xlabel="predicted probability", ylabel="actual default rate",
       title="Reliability — before vs after calibration (test)", xlim=(0, 1), ylim=(0, 1))
ax.legend(); fig.tight_layout(); fig.savefig("explain/reliability_after.png", dpi=110)

joblib.dump(calibrated, "src/model_calibrated.joblib")
print("saved src/model_calibrated.joblib and explain/reliability_after.png")
