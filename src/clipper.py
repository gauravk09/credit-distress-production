"""A pipeline step that caps extreme values.

Learns a per-column floor and ceiling from the TRAINING data (1st and 99th
percentiles), then clamps every value into that range. This stops a few corrupt
outliers (e.g. RevolvingUtilization = 50,708) from inflating the scale and
blinding the model to the normal 0-1 range.

Fit on train only; the learned bounds are baked into the saved model, so the
live server clips incoming requests the exact same way.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns)
        self.lo_ = np.nanpercentile(X.values, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X.values, self.upper, axis=0)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        # Coerce to float so a missing value arriving as Python None becomes NaN
        # (np.clip errors on None, but passes NaN through untouched).
        vals = X.apply(pd.to_numeric, errors="coerce").values
        clipped = np.clip(vals, self.lo_, self.hi_)
        return pd.DataFrame(clipped, columns=X.columns, index=X.index)
