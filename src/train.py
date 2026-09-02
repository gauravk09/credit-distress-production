"""One parametrized trainer for every model iteration.

Replaces the old per-iteration scripts (train_weighted / train_clipped / train_xgb /
train_mlflow / train_all_mlflow). Each RECIPE names its estimator, whether to clip
outliers, and how to weight the rare class. Trains on train, scores on validation,
records to the lean scoreboard, and saves the model artifact.

Usage:
    python3 src/train.py <recipe>        # one recipe
    python3 src/train.py all             # every recipe
    python3 src/train.py all --mlflow    # also log each run to MLflow
"""
import sys
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, recall_score, precision_score,
                             confusion_matrix)
from xgboost import XGBClassifier

from data import load_splits
from evaluate import evaluate_and_record
from clipper import PercentileClipper
from features import FeatureEngineer

# name -> how to build it. `est` is 'logreg' or 'xgboost'; `clip`/`weight` are knobs.
# `file` keeps the historic artifact names so downstream scripts keep loading them.
RECIPES = {
    "baseline":    {"record": "01_baseline",    "est": "logreg",  "clip": False, "weight": None,       "file": "model.joblib"},
    "weighted":    {"record": "03_classweight", "est": "logreg",  "clip": False, "weight": "balanced", "file": "model_weighted.joblib"},
    "clipped":     {"record": "04_clipped",     "est": "logreg",  "clip": True,  "weight": "balanced", "file": "model_clipped.joblib"},
    "xgboost":     {"record": "05_xgboost",     "est": "xgboost", "clip": False, "weight": "balanced", "file": "model_xgb.joblib"},
    "xgboost_fe":  {"record": "06_xgboost_fe",  "est": "xgboost", "clip": False, "weight": "balanced", "fe": True, "file": "model_xgb_fe.joblib"},
}


def build_pipeline(recipe, spw):
    """Assemble the sklearn Pipeline for a recipe. Trees skip clip/impute/scale
    (scale-invariant, handle NaN); logreg needs all three."""
    if recipe["est"] == "xgboost":
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=(spw if recipe["weight"] else 1),
                            eval_metric="aucpr", random_state=42)
        steps = [("fe", FeatureEngineer())] if recipe.get("fe") else []
        return Pipeline(steps + [("clf", clf)])

    steps = []
    if recipe["clip"]:
        steps.append(("clip", PercentileClipper(1, 99)))
    steps += [("impute", SimpleImputer(strategy="median")),
              ("scale", StandardScaler()),
              ("clf", LogisticRegression(max_iter=1000, class_weight=recipe["weight"]))]
    return Pipeline(steps)


def run(name, recipe, X_tr, X_val, y_tr, y_val, use_mlflow):
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    model = build_pipeline(recipe, spw)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_val)[:, 1]

    params = {"est": recipe["est"], "clip": recipe["clip"], "weight": recipe["weight"]}
    evaluate_and_record(name=recipe["record"], params=params,
                        y_true=y_val, proba=proba, threshold=0.5)

    joblib.dump(model, f"src/{recipe['file']}")
    if use_mlflow:
        _log_mlflow(recipe, params, y_val, proba, model)
    return average_precision_score(y_val, proba)


def _log_mlflow(recipe, params, y_val, proba, model):
    import mlflow, mlflow.sklearn
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-distress")
    with mlflow.start_run(run_name=recipe["record"]):
        mlflow.log_params(params)
        mlflow.log_metrics({"pr_auc": float(average_precision_score(y_val, proba)),
                            "recall": float(recall_score(y_val, pred, zero_division=0)),
                            "precision": float(precision_score(y_val, pred, zero_division=0)),
                            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_mlflow = "--mlflow" in sys.argv
    target = args[0] if args else "baseline"
    names = list(RECIPES) if target == "all" else [target]
    if target != "all" and target not in RECIPES:
        sys.exit(f"unknown recipe '{target}'. choose from: {', '.join(RECIPES)} | all")

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_splits()
    for name in names:
        auc = run(name, RECIPES[name], X_tr, X_val, y_tr, y_val, use_mlflow)
        print(f"  {name:<10} PR-AUC={auc:.3f}  -> src/{RECIPES[name]['file']}")


if __name__ == "__main__":
    main()
