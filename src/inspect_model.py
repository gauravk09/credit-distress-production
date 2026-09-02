"""Look inside the served (@champion) model: pipeline steps, clip bounds, and the
logistic-regression coefficients. Because inputs are standardized, the coefficient
magnitudes ARE the global feature importances (which features the model leans on overall)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn
import clipper  # noqa: F401

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/credit-distress@champion")

cols = list(model.feature_names_in_)
clip = model.named_steps["clip"]
coef = model.named_steps["clf"].coef_[0]      # standardized space -> importance

print("pipeline steps:", " -> ".join(model.named_steps.keys()))
print(f"\n{'feature':<40} {'coef(importance)':>16} {'clip_low':>10} {'clip_high':>12}")
for i in np.argsort(-np.abs(coef)):
    print(f"{cols[i]:<40} {coef[i]:>16.3f} {clip.lo_[i]:>10.3f} {clip.hi_[i]:>12.3f}")

# global importance bar chart
order = np.argsort(np.abs(coef))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.barh([cols[i] for i in order], [coef[i] for i in order],
        color=["#d62728" if coef[i] >= 0 else "#1f77b4" for i in order])
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("logistic-regression coefficient (standardized)")
ax.set_title("Served model — global feature importance\n(red = raises risk, blue = lowers)")
fig.tight_layout(); os.makedirs("explain", exist_ok=True)
fig.savefig("explain/global_importance.png", dpi=110)
print("\nsaved explain/global_importance.png")
