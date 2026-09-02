"""Champion vs FE-challenger on the SEALED test set (a genuine improvement, not drift).
Challenger = xgboost_fe + Platt calibration (same recipe family as the champion).
Promote to v_next only if it beats the champion's PR-AUC by a margin."""
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss
import mlflow, mlflow.sklearn
from mlflow import MlflowClient

from data import load_splits
from train import build_pipeline, RECIPES

NAME, MARGIN = "credit-distress", 0.002
X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()

champion = joblib.load("src/model_calibrated.joblib")

spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
fe = build_pipeline(RECIPES["xgboost_fe"], spw).fit(X_tr, y_tr)
challenger = CalibratedClassifierCV(fe, cv="prefit", method="sigmoid").fit(X_val, y_val)

def scores(m):
    p = m.predict_proba(X_te)[:, 1]
    return average_precision_score(y_te, p), brier_score_loss(y_te, p)

ca, cb = scores(champion)
ha, hb = scores(challenger)
print(f"champion (v3, no FE) : PR-AUC {ca:.4f} | Brier {cb:.4f}")
print(f"challenger (FE)      : PR-AUC {ha:.4f} | Brier {hb:.4f}")
print(f"delta PR-AUC         : {ha - ca:+.4f}  (margin {MARGIN})")

if ha >= ca + MARGIN:
    joblib.dump(challenger, "src/model_calibrated.joblib")   # new champion snapshot for export
    mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment("credit-distress")
    client = MlflowClient()
    with mlflow.start_run(run_name="08_xgboost_fe_calibrated") as run:
        mlflow.log_params({"model": "xgboost_fe+platt", "threshold": 0.091, "features": "FE"})
        mlflow.log_metrics({"pr_auc": ha, "brier": hb})
        mlflow.sklearn.log_model(challenger, name="model", serialization_format="cloudpickle")
        rid = run.info.run_id
    mv = mlflow.register_model(f"runs:/{rid}/model", NAME)
    client.set_model_version_tag(NAME, mv.version, "threshold", "0.091")
    client.set_registered_model_alias(NAME, "champion", mv.version)
    print(f"\nDECISION: PROMOTE -> {NAME} v{mv.version} @champion")
else:
    print("\nDECISION: KEEP champion (FE gain below margin) -> no redeploy")
