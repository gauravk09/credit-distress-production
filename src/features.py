"""Feature engineering (ideas borrowed from top Give-Me-Some-Credit solutions,
re-implemented). Adds NEW information a tree model can't derive on its own:
  - sentinel flag: past-due columns use 96/98 as data-entry sentinels, strongly
    linked to default;
  - missingness flags: missing income / dependents is itself predictive;
  - total_past_due aggregate;
  - income_per_dependent ratio.
Runs as the FIRST pipeline step, so it sees RAW inputs (incl. NaN) before anything
else, and the same transform runs at serving time (10-field API contract unchanged).
Monotone transforms (log/scale) are deliberately omitted — they don't help trees."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LATE_COLS = ["NumberOfTime30-59DaysPastDueNotWorse",
             "NumberOfTimes90DaysLate",
             "NumberOfTime60-89DaysPastDueNotWorse"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = X.copy()
        out["income_missing"] = X["MonthlyIncome"].isna().astype(int)
        out["dependents_missing"] = X["NumberOfDependents"].isna().astype(int)
        # 96/98 are sentinels; real delinquency counts are small. Flag any row with one.
        out["sentinel_delinquency"] = (X[LATE_COLS] >= 90).any(axis=1).astype(int)
        out["total_past_due"] = X[LATE_COLS].clip(upper=20).sum(axis=1)   # cap sentinels before summing
        out["income_per_dependent"] = X["MonthlyIncome"] / (X["NumberOfDependents"].fillna(0) + 1)
        return out
