"""Retrain-on-drift with a champion/challenger guardrail.

Steps:
  1. Simulate the drifted, matured, LABELED new world (recession: income x0.6, real labels).
  2. Champion = the currently served model (loaded, NOT retrained).
  3. Challenger = the winning recipe (XGBoost + Platt) retrained on the NEW world.
  4. Judge BOTH on a fresh held-out slice of the NEW world.
  5. Promote the challenger ONLY if it beats the champion; else keep champion + alert.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss
from xgboost import XGBClassifier

TARGET = "FinancialDistressNextTwoYears"
MARGIN = 0.005   # challenger must beat champion PR-AUC by at least this


def evaluate_challenger():
    """Retrain a challenger on the drifted world and judge it against the champion.
    Returns the decision dict. The GUARDRAIL lives here: promote only if it wins."""
    df = pd.read_csv("data/credit.csv")
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET]).copy()
    X["MonthlyIncome"] = X["MonthlyIncome"] * 0.6      # recession covariate shift, labels kept

    X_fit, X_tmp, y_fit, y_tmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=1)
    X_cal, X_judge, y_cal, y_judge = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=1)

    champion = joblib.load("src/model_calibrated.joblib")
    spw = float((y_fit == 0).sum() / (y_fit == 1).sum())
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                        eval_metric="aucpr", random_state=42).fit(X_fit, y_fit)
    challenger = CalibratedClassifierCV(xgb, cv="prefit", method="sigmoid").fit(X_cal, y_cal)

    def auc(m):
        return average_precision_score(y_judge, m.predict_proba(X_judge)[:, 1])

    champ, chall = auc(champion), auc(challenger)
    promote = chall >= champ + MARGIN
    if promote:
        joblib.dump(challenger, "src/model_retrained.joblib")
    return {"champion_auc": champ, "challenger_auc": chall, "promote": promote}


def main():
    r = evaluate_challenger()
    print(f"champion   : PR-AUC {r['champion_auc']:.3f}")
    print(f"challenger : PR-AUC {r['challenger_auc']:.3f}")
    if r["promote"]:
        print(f"\nDECISION: PROMOTE (PR-AUC +{r['challenger_auc'] - r['champion_auc']:.3f}) "
              "-> saved src/model_retrained.joblib")
    else:
        print(f"\nDECISION: KEEP champion (gain {r['challenger_auc'] - r['champion_auc']:+.3f} "
              f"< margin {MARGIN}) -> alert a human")


if __name__ == "__main__":
    main()
