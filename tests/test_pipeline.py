"""Smoke tests for the parametrized trainer: every recipe fits and returns
valid probabilities, including with a missing value present."""
import numpy as np
import pandas as pd
import pytest
from train import build_pipeline, RECIPES

COLS = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]


def _toy():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(300, 10), columns=COLS)
    X.loc[0, "MonthlyIncome"] = np.nan            # a missing value the pipeline must handle
    y = (X["RevolvingUtilizationOfUnsecuredLines"] + rng.rand(300) * 0.3 > 0.8).astype(int)
    return X, y


@pytest.mark.parametrize("name", list(RECIPES))
def test_recipe_fits_and_predicts(name):
    X, y = _toy()
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    model = build_pipeline(RECIPES[name], spw)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()     # valid probabilities
