"""Log the XGBoost model to MLflow, register it as credit-distress v2, and move
the @champion alias to it. This is the real promotion the registry was built for."""
import joblib
import numpy as np
import mlflow, mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score, recall_score, precision_score, confusion_matrix)

from data import load_splits

NAME = "credit-distress"
model = joblib.load("src/model_xgb.joblib")
X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()
proba = model.predict_proba(X_val)[:, 1]

# cost-optimal threshold (K=10) for THIS model on validation
K, y = 10, y_val.values
thr = float(min(
    ((t, K * int(((proba < t) & (y == 1)).sum()) + int(((proba >= t) & (y == 0)).sum()))
     for t in np.round(np.arange(0.05, 0.96, 0.01), 2)), key=lambda z: z[1])[0])
pred = (proba >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-distress")
client = MlflowClient()

with mlflow.start_run(run_name="05_xgboost") as run:
    mlflow.log_params({"model": "xgboost", "scale_pos_weight": round(float((y_tr == 0).sum() / (y_tr == 1).sum()), 2),
                       "n_estimators": 300, "max_depth": 4, "lr": 0.05, "threshold": thr})
    mlflow.log_metrics({
        "pr_auc": float(average_precision_score(y, proba)),
        "recall": float(recall_score(y, pred)), "precision": float(precision_score(y, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    run_id = run.info.run_id

mv = mlflow.register_model(f"runs:/{run_id}/model", NAME)          # -> version 2
client.set_model_version_tag(NAME, mv.version, "threshold", str(thr))
client.set_registered_model_alias(NAME, "champion", mv.version)     # move the crown

print(f"promoted xgboost as {NAME} v{mv.version}, @champion, threshold={thr}")
