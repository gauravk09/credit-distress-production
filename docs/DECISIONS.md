# Decisions

Finalised choices only. Each: what was chosen, what was rejected, why.

## D1 — Dataset: "Give Me Some Credit" — LOCKED
- **Chosen:** Give Me Some Credit (150k borrowers, ~7% default, target = `FinancialDistressNextTwoYears`).
- **Rejected:** Credit Card Fraud (features anonymized PCA — drift hard to narrate); Telco Churn (imbalance too gentle at 27%).
- **Why:** strong-but-not-extreme imbalance for real lessons, AND human-readable features (age, income, debt ratio) so drift stories are vivid.

## D2 — Data source: OpenML, not Kaggle — LOCKED
- **Chosen:** `sklearn.datasets.fetch_openml(name='GiveMeSomeCredit', version=1)`.
- **Rejected:** Kaggle CLI (needs API token + competition-rules acceptance = friction).
- **Why:** identical data, zero account setup, trusted research host. Verified: 150000×10 loads clean.

## D3 — Primary metric: PR-AUC — AGREED
- **Chosen:** PR-AUC (area under precision-recall curve) as the headline score; report recall & precision at a chosen threshold too.
- **Rejected:** accuracy (lazy always-"No" scores 93.32%), ROC-AUC (FP rate denominator = 139,974 negatives hides false alarms on imbalanced data).
- **Why:** rare positive class (6.68%) → precision's small denominator (TP+FP) stays honest where ROC flatters.

## D4 — Experiment tracking: lean scoreboard now, MLflow later — AGREED
- **Chosen:** `runs/scoreboard.csv` (one row per experiment) + `runs/<name>/` PNGs (PR curve, confusion, score histogram), written by `src/evaluate.py`. Every dev iteration scored on VALIDATION; test sealed until the end.
- **Rejected (for now):** MLflow — parked to a later step as a "here's the real tool" lesson once the concept is understood from first principles.
- **Why:** understand what tracking IS before adopting a tool that hides it; matches lean-code rule.

## D5 — Split: 60/20/20 train/val/test, stratified — AGREED
- **Chosen:** tune threshold/model on validation, report final once on test.
- **Rejected:** 2-way train/test (would force tuning on test = leakage).
- **Why:** keep test an honest stand-in for future borrowers.

## D6 — Serve the weighted model at threshold 0.56 — AGREED
- **Chosen:** serve `model_weighted.joblib` (PR-AUC 0.294 > baseline 0.241). Threshold picked EMPIRICALLY as the cost-minimizer (10·FN + FP) on validation = 0.56, not via 1/(1+K), because weighting broke calibration.
- **Rejected:** baseline model (weaker); 1/(1+K) formula on weighted probabilities (invalid — uncalibrated).
- **Why:** best ranking + an operating point that reflects real miss/false-alarm costs without needing calibration.

## D7 — Drift: understand PSI by hand, then use Evidently — AGREED
- **Chosen:** taught PSI from first principles (buckets → shares → weighted sum → threshold), then ran Evidently `DataDriftPreset` for the automated per-feature report.
- **Note:** Evidently auto-selected Wasserstein distance (not PSI) for numeric income; same concept, different ruler.
- **Why:** lean-first — know what the tool computes before trusting its verdict.

## D8 — Clip outliers at 1st/99th percentile (learned on train) — LOCKED
- **Chosen:** `PercentileClipper` as first pipeline step; per-feature floor/ceiling from train percentiles, baked into the saved model so serving clips identically.
- **Rejected:** hand-picked cap (e.g. 1.0 — magic number, feature-specific); RobustScaler/QuantileTransformer (bigger change, less transparent for teaching).
- **Why:** generic (no per-feature tuning), corrupt-row-proof, revived RevolvingUtilization and lifted PR-AUC 0.294→0.369. serve.py now loads model path from config, so promotion = edit JSON.

## D9 — Adopt MLflow (SQLite backend) for tracking — AGREED
- **Chosen:** `src/train_all_mlflow.py` logs all 4 iterations (full metrics + plots + model) to MLflow, tracking_uri `sqlite:///mlflow.db`. UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`.
- **Gotchas hit:** (1) MLflow 3 skops saver rejects custom `PercentileClipper` → use `serialization_format="cloudpickle"`. (2) MLflow 3 deprecated the file store (`./mlruns`) for the UI → must use a DB backend.
- **Why:** MLflow snapshots the model + its exact environment per run (CSV scoreboard couldn't); real UI to sort/compare. Lean scoreboard (D4) stays for quick in-loop plots.

## D10 — Serve via MLflow Model Registry alias, not a file path — AGREED
- **Chosen:** register best run as `credit-distress` v1, alias `@champion`, threshold stored as a version TAG. serve.py loads `models:/credit-distress@champion` at startup. Promotion = move the alias (no code change).
- **Rejected:** hardcoded joblib path + serving_threshold.json (works, zero deps, but manual file juggling and no history).
- **Tradeoff (honest):** server now depends on the MLflow DB at startup; a file path had no such dependency. Bought governance/history for a moving part.

## D11 — Serve explanations with SHAP, gated & model-agnostic — AGREED
- **Chosen:** `/predict?explain=true` returns a SHAP receipt (base + per-feature contributions). Permutation explainer built + warmed once at startup. Score-only stays ~0.03s; explain ~0.04s steady (first call ~5s without warmup). Client renders waterfall from JSON.
- **Rejected:** SHAP on every call (score-only calls shouldn't pay for it); LIME for serving (weights don't sum to the prediction — not auditable); linear-SHAP shortcut (wouldn't survive a GBM champion swap).
- **Why:** explanations must be auditable (SHAP sums exactly) and opt-in; model-agnostic method survives future champion changes. LIME kept as a dev-time sanity check only.

## D12 — Champion = XGBoost v2, coercion at the serving boundary — AGREED
- **Chosen:** XGBoost (PR-AUC 0.381 > logreg 0.369) registered as v2, `@champion` moved to it (thr 0.6). No clip/impute/scale needed (trees). `None`→`NaN` coercion moved to serve.py boundary so any champion is safe.
- **Rejected:** HistGradientBoosting (user preferred XGBoost); keeping coercion inside the clipper (broke on a clipper-less champion).
- **Why:** small gain confirms data is mostly additive (honest finding); boundary guard is model-agnostic; alias swap = zero endpoint code change. SHAP (model-agnostic) survived the swap unchanged.

## D13 — Calibrate with Platt (sigmoid), promote as champion v3 — LOCKED
- **Chosen:** wrap XGBoost in `CalibratedClassifierCV(cv='prefit', method='sigmoid')` fit on validation; champion v3 @ threshold 0.091 (= 1/(1+K), K=10 — valid now that probs are honest).
- **Rejected:** isotonic (equal Brier 0.0486 vs 0.0490 but dinged PR-AUC 0.407→0.388 via step-function ties); leaving raw XGBoost (avg pred 0.31 vs actual 0.067 = overconfident, misleads downstream decisions).
- **Why:** miscalibration was a smooth systematic shift → Platt's 2-param sigmoid nails it while preserving ranking exactly. Honest probs also fixed the SHAP base (0.32→0.049) and revived the day-one cost-threshold formula. Calibrated probs cap ~0.5 because even the riskiest 1% only default ~66% (credit risk is inherently uncertain).

## D14 — Retrain-on-drift: champion/challenger guardrail, never auto-promote blind — LOCKED
- **Chosen:** on drift, retrain a challenger on fresh labeled data, judge champion vs challenger on a held-out slice of the NEW world, promote only if challenger beats champion by a margin (0.02 PR-AUC). Rollback = move the alias back.
- **Rejected:** auto-promote on retrain (deploys worse models — proven: covariate-drift challenger scored 0.409 < champion 0.425, guardrail correctly kept champion).
- **Why (key lessons):** (1) input drift ≠ performance drop — scaling income didn't hurt the scale-invariant tree champion, so retraining was needless. (2) Only CONCEPT drift (rule change) genuinely degrades the model (champion 0.425→0.09) and justifies a promotion (challenger 0.79). (3) Label lag means retraining runs on a rolling window of matured labels, slower than detection. (4) Alias-based promotion makes rollback a one-liner — the safety net that makes automation acceptable.
