# Bug Journal

Grouped by lesson, not date. Each entry: what happened, why, the fix, what caught it.

## Lesson: outliers can silently kill a feature under StandardScaler

**Symptom:** three live requests with very different `RevolvingUtilizationOfUnsecuredLines`
(0.95 / 0.20 / 0.60) all returned the identical probability 0.6787.

**Diagnosis:** the feature's training data has median 0.154 but max 50,708 and std 249.8
(values are supposed to be a 0–1 fraction; the huge ones are corrupt). StandardScaler divides
by that inflated std, so a realistic 0.05→0.95 swing maps to a ~0.0036 change in scaled space —
invisible to the model. A strong predictor was effectively inert.

**Fix (DONE, iteration 04):** added `PercentileClipper` (clip every feature to its train 1st/99th
percentile) as the first pipeline step, before scaling. RevolvingUtilization sweep now moves the
prediction (0.05→0.35, 0.95→0.78, was flat 0.68). Bonus: PR-AUC 0.294→0.369. Verified live on the
server after promoting model_clipped.joblib.

**What caught it:** eyeballing the *request log* — noticing identical outputs for different inputs.
The scoreboard (PR-AUC) did NOT catch it, because the model was still "OK on average"; only per-request
inspection exposed the dead feature. Verified by sweeping the feature through predict_proba directly.

## Lesson: a thin sample makes the drift monitor cry wolf

**Symptom:** monitoring 400 live requests (all sampled from the SAME training data, no real drift)
flagged `NumberRealEstateLoansOrLines` at 0.126, just over the 0.1 threshold. Predicted "all clear",
got a false alarm.

**Diagnosis:** 400 current rows vs 5,000 reference rows. With few rows, a feature's empirical shape
wobbles from luck alone, and a low-variance count feature can drift past a tight threshold by chance.

**Fix / rule:** don't trust drift on tiny batches. Re-running at n=1000 and n=3000 → 0 drifted.
Wait for enough traffic (≥ ~1k rows here) before computing drift, and/or don't alert on a single
borderline feature — alert when drift is large or persists across batches.

**What caught it:** the monitor disagreeing with a known-true expectation (same data → no drift),
then verified by sweeping sample size (400 → 1000 → 3000) and watching the false alarm disappear.

## Lesson: a failing install can be the environment's fault, not the package's

**Symptom:** `pip install lime` failed with "setup.py egg_info did not run: setuptools is not available",
even with `--no-build-isolation`.

**Diagnosis:** setuptools itself was broken — `import setuptools` raised
`cannot import name 'tarfile' from 'backports'`. A stale `backports` namespace was missing
`backports.tarfile`, which modern setuptools vendors. So EVERY source build would have failed, not just LIME.

**Fix:** `pip install backports.tarfile` (pure wheel, no build) → setuptools imports → LIME installs.

**What caught it:** not stopping at "LIME is broken" — probing `import setuptools` directly surfaced the
real culprit one layer down.

## Lesson: train-time and serve-time can feed a component different TYPES

**Symptom:** live `/predict` on a borrower with missing income/dependents returned HTTP 500
(`TypeError: '>=' not supported between NoneType and float` in PercentileClipper).

**Diagnosis:** at TRAIN time missing values were `np.nan` (float), and `np.clip` passes NaN through.
At SERVE time a missing field arrives from the API as Python `None`, making the column object-dtype;
`np.clip` then tries `None >= 0.5` and throws. The clipper was tested only against training-shaped data.

**Fix:** coerce to numeric first (`X.apply(pd.to_numeric, errors='coerce')`) so `None` -> `NaN`.
No retraining needed: the pickled model stores only learned bounds; methods come from the class,
so editing clipper.py + restarting the server fixed the live model.

**What caught it:** a deliberate corrupt-input test suite (`data/corrupt_sample.csv` +
`src/test_corrupt.py`) hitting the real endpoint — NOT the training tests, which never send `None`.
Also surfaced a design smell: the pipeline silently CLAMPS impossible values (−5 age, −$9k income)
instead of flagging them (robust but silent — noted for a future validation step).

**RESOLVED (validation):** added `Field(ge=0, ...)` constraints (age also `le=120`) to `BorrowerFeatures`.
Impossible inputs now return a 422 naming every bad field (e.g. "age: Input should be >= 0"); unusual-
but-real values (huge debt ratio, utilization 50708) still flow through. Reject the impossible, not the
merely weird. Validation lives at the request boundary, so it runs before the model.

**Follow-up (the fix was in the wrong place):** the None coercion above was put INSIDE the logreg's
PercentileClipper. When we swapped @champion to XGBoost (no clipper), missing income 500'd again
(`DataFrame.dtypes must be int/float...: MonthlyIncome: object`). Correct fix: coerce at the SERVING
BOUNDARY in `serve.py` (`X.apply(pd.to_numeric, errors="coerce")`), so it protects ANY champion.
Lesson: put a cross-cutting guard where every model passes through, not inside one model's pipeline.

## Tally — what caught each bug
- 1 bug — reading the live request log (identical outputs, varying inputs)
- 1 bug — monitor verdict contradicting a known-true case (no-drift traffic), confirmed by sample-size sweep
- 1 env bug — probing the dependency (setuptools) instead of blaming the package (LIME)
- 1 serve bug — corrupt-input test against the REAL endpoint (None vs NaN type mismatch)
- 1 swap bug — champion swap to a clipper-less model re-exposed None; fix belonged at the boundary
