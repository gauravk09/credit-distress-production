"""Head-to-head: isotonic vs sigmoid (Platt) calibration.
Both fit on validation (held-out from training), both judged on the untouched TEST set.
The ONLY code difference is method='isotonic' vs method='sigmoid'."""
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, average_precision_score

from data import load_splits

model = joblib.load("src/model_xgb.joblib")
X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()

# fit BOTH correctors on validation (prefit = keep the model, learn only the mapping)
iso = CalibratedClassifierCV(model, cv="prefit", method="isotonic").fit(X_val, y_val)
sig = CalibratedClassifierCV(model, cv="prefit", method="sigmoid").fit(X_val, y_val)   # Platt

scores = {
    "raw (before)":      model.predict_proba(X_te)[:, 1],
    "isotonic":          iso.predict_proba(X_te)[:, 1],
    "sigmoid (Platt)":   sig.predict_proba(X_te)[:, 1],
}

print(f"actual default rate (test): {y_te.mean():.3f}\n")
print(f"{'method':<18}{'avg pred':>10}{'Brier':>9}{'PR-AUC':>9}")
for name, p in scores.items():
    print(f"{name:<18}{p.mean():>10.3f}{brier_score_loss(y_te, p):>9.4f}{average_precision_score(y_te, p):>9.3f}")

fig, ax = plt.subplots(figsize=(5.6, 5))
ax.plot([0, 1], [0, 1], "--", c="grey", label="perfect")
for (name, p), c in zip(scores.items(), ["#d62728", "#2ca02c", "#1f77b4"]):
    fp, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", c=c, label=name)
ax.set(xlabel="predicted probability", ylabel="actual default rate",
       title="Isotonic vs sigmoid (Platt) — test", xlim=(0, 1), ylim=(0, 1))
ax.legend(); fig.tight_layout(); fig.savefig("explain/calibration_compare.png", dpi=110)
print("\nsaved explain/calibration_compare.png")
