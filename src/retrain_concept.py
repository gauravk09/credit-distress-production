"""Retrain-on-drift under CONCEPT drift (the rule changed), with promotion.

SIMULATION NOTE: to demonstrate concept drift we FABRICATE new labels whose driver
is different from what the champion learned. The champion keyed on late-payments/
utilization; in the 'new world' default is driven by DEBT RATIO and LOW INCOME
instead. So the old model goes blind and a retrained challenger should win.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss
from xgboost import XGBClassifier
import mlflow, mlflow.sklearn
from mlflow import MlflowClient

TARGET, NAME, MARGIN = "FinancialDistressNextTwoYears", "credit-distress", 0.02
rng = np.random.RandomState(7)

df = pd.read_csv("data/credit.csv")
X = df.drop(columns=[TARGET]).copy()

# --- fabricate concept-drifted labels: default now depends on debt ratio & low income ---
debt = X["DebtRatio"].clip(upper=X["DebtRatio"].quantile(0.99)).fillna(0)
inc = X["MonthlyIncome"].fillna(X["MonthlyIncome"].median())
z_debt = (debt - debt.mean()) / debt.std()
z_inc = (inc - inc.mean()) / inc.std()
logit = -3.1 + 2.0 * z_debt - 1.0 * z_inc            # NEW rule (old model doesn't know it)
p_new = 1 / (1 + np.exp(-logit))
y = pd.Series((rng.rand(len(X)) < p_new).astype(int), index=X.index)
print(f"new-world default rate: {y.mean():.3f}")

X_fit, X_tmp, y_fit, y_tmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=1)
X_cal, X_judge, y_cal, y_judge = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=1)

champion = joblib.load("src/model_calibrated.joblib")
spw = float((y_fit == 0).sum() / (y_fit == 1).sum())
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                    random_state=42).fit(X_fit, y_fit)
challenger = CalibratedClassifierCV(xgb, cv="prefit", method="sigmoid").fit(X_cal, y_cal)

def score(m):
    p = m.predict_proba(X_judge)[:, 1]
    return average_precision_score(y_judge, p), brier_score_loss(y_judge, p)

ca, cb = score(champion); ha, hb = score(challenger)
print(f"champion   : PR-AUC {ca:.3f} | Brier {cb:.4f}")
print(f"challenger : PR-AUC {ha:.3f} | Brier {hb:.4f}")

if ha >= ca + MARGIN:
    print(f"\nDECISION: PROMOTE (PR-AUC +{ha - ca:.3f})")
    joblib.dump(challenger, "src/model_retrained.joblib")
    mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment("credit-distress")
    client = MlflowClient()
    with mlflow.start_run(run_name="07_retrained_concept") as run:
        mlflow.log_params({"model": "xgboost+platt", "trigger": "concept_drift_sim", "threshold": 0.091})
        mlflow.log_metrics({"pr_auc": ha, "brier": hb})
        mlflow.sklearn.log_model(challenger, name="model", serialization_format="cloudpickle")
        rid = run.info.run_id
    mv = mlflow.register_model(f"runs:/{rid}/model", NAME)
    client.set_model_version_tag(NAME, mv.version, "threshold", "0.091")
    client.set_registered_model_alias(NAME, "champion", mv.version)
    print(f"promoted challenger as {NAME} v{mv.version}, @champion")
else:
    print(f"\nDECISION: KEEP champion (gain {ha - ca:+.3f} < {MARGIN})")
