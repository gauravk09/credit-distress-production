"""Unit tests for PercentileClipper — guards the None-vs-NaN serving bug."""
import numpy as np
import pandas as pd
from clipper import PercentileClipper


def test_clips_extreme_outliers():
    train = pd.DataFrame({"a": list(range(100)) + [10_000], "b": range(101)})
    clip = PercentileClipper(1, 99).fit(train)
    out = clip.transform(pd.DataFrame({"a": [10_000], "b": [50]}))
    assert out["a"].iloc[0] <= train["a"].quantile(0.99) + 1   # monster pulled down to the 99th pct


def test_none_becomes_nan_not_error():
    """The exact regression: a missing value arriving as Python None must not crash."""
    train = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [1.0, 2, 3, 4]})
    clip = PercentileClipper(1, 99).fit(train)
    out = clip.transform(pd.DataFrame({"a": [None], "b": [2.0]}))   # None, not NaN
    assert np.isnan(out["a"].iloc[0])                               # coerced, passed through
    assert out["b"].iloc[0] == 2.0
