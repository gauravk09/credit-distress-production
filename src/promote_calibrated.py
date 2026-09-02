"""Wrap the XGBoost champion in a Platt (sigmoid) calibrator, register as v3,
and move @champion to it. Because probabilities are now honest, the day-one
cost-optimal threshold formula 1/(1+K) applies again -- we verify it empirically."""
import joblib
import numpy as np
import mlflow, mlflow.sklearn
from mlflow import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, recall_score, precision_score,
                             brier_score_loss, confusion_matrix)

from data import load_splits

NAME, K = "credit-distress", 10
base = joblib.load("src/model_xgb.joblib")
X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()

calibrated = CalibratedClassifierCV(base, cv="prefit", method="sigmoid").fit(X_val, y_val)
joblib.dump(calibrated, "src/model_calibrated.joblib")

# threshold: theory says 1/(1+K) on calibrated probs; confirm with empirical cost-min on val
p_val = calibrated.predict_proba(X_val)[:, 1]; yv = y_val.values
formula_thr = round(1 / (1 + K), 3)
emp_thr = float(min(
    ((t, K * int(((p_val < t) & (yv == 1)).sum()) + int(((p_val >= t) & (yv == 0)).sum()))
     for t in np.round(np.arange(0.02, 0.60, 0.01), 2)), key=lambda z: z[1])[0])
print(f"threshold: formula 1/(1+K) = {formula_thr}   empirical cost-min = {emp_thr}")
thr = formula_thr

# final honest metrics on TEST
p_te = calibrated.predict_proba(X_te)[:, 1]; yt = y_te.values
pred = (p_te >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
print(f"TEST: PR-AUC {average_precision_score(yt, p_te):.3f} | Brier {brier_score_loss(yt, p_te):.4f} "
      f"| recall {recall_score(yt, pred):.3f} precision {precision_score(yt, pred):.3f} @thr {thr}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-distress")
client = MlflowClient()
with mlflow.start_run(run_name="06_calibrated") as run:
    mlflow.log_params({"model": "xgboost+platt", "calibration": "sigmoid", "threshold": thr, "K": K})
    mlflow.log_metrics({"pr_auc": float(average_precision_score(yt, p_te)),
                        "brier": float(brier_score_loss(yt, p_te)),
                        "recall": float(recall_score(yt, pred)),
                        "precision": float(precision_score(yt, pred)),
                        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
    mlflow.sklearn.log_model(calibrated, name="model", serialization_format="cloudpickle")
    run_id = run.info.run_id

mv = mlflow.register_model(f"runs:/{run_id}/model", NAME)
client.set_model_version_tag(NAME, mv.version, "threshold", str(thr))
client.set_registered_model_alias(NAME, "champion", mv.version)
print(f"promoted calibrated model as {NAME} v{mv.version}, @champion, threshold={thr}")
