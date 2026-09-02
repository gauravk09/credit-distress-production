"""Export the @champion snapshot from the MLflow registry into the self-contained
serving/ folder, so the deployed container needs NO MLflow server.

Registry = source of truth (dev); the deploy image bundles a frozen snapshot.
Produces: serving/model.joblib, serving/meta.json, serving/background.csv
"""
import os, json
import joblib
import pandas as pd
import mlflow, mlflow.sklearn
from mlflow import MlflowClient

FEATURES = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]

os.makedirs("serving", exist_ok=True)
# Read tracking URI from env so CD can point at a hosted registry; default to local sqlite.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mv = MlflowClient().get_model_version_by_alias("credit-distress", "champion")
model = mlflow.sklearn.load_model("models:/credit-distress@champion")

joblib.dump(model, "serving/model.joblib")
json.dump({"model": "credit-distress", "version": int(mv.version),
           "threshold": float(mv.tags["threshold"])}, open("serving/meta.json", "w"), indent=2)
# small SHAP background so the container needn't ship the full dataset
pd.read_csv("data/credit.csv")[FEATURES].sample(100, random_state=0).to_csv("serving/background.csv", index=False)

print(f"exported champion v{mv.version} (threshold {mv.tags['threshold']}) -> serving/")
