# Roadmap — Learning ML in Production

Goal: Gaurav can defend every production-ML decision in an interview. Build train → serve → drift for real, on an imbalanced credit dataset.

## The journey (train → serve → watch it rot)
- [x] **0. Pick data** — Give Me Some Credit (see D1/D2). Saved to `data/credit.csv`.
- [ ] **1. Understand the data + imbalance** — why accuracy lies here. ← current
- [x] **2. Train a baseline model** — logistic regression. PR-AUC 0.230 (lazy 0.067) but recall@0.5 only 3.8% (77/2005). Model ranks OK, acts timid.
- [x] **3a. Threshold tuning** — cost ratio K=10 → thr=0.091. Recall 0.044→0.472, caught 88→947. PR-AUC unchanged (moved ALONG curve). Saved src/threshold.json.
- [x] **3b. Class weights** — class_weight='balanced' (defaulter ~14x). PR-AUC 0.241→0.294, the whole curve lifted. Caveat: probabilities now uncalibrated. Saved src/model_weighted.joblib.
- [x] **4. Serve it live** — FastAPI `src/serve.py`, POST /predict. Weighted model, threshold 0.56 (cost-optimal, K=10). Live on :8137. Tested risky→flag, safe→clear.
- [x] **5. Log every request** — JSONL to logs/requests.jsonl (features+prediction+ts). Immediately caught a dead-feature bug (see BUGS.md).
- [x] **6. Measure drift** — by hand (PSI=0.537 on income) then Evidently (`src/drift_report.py`, flagged 1/10 cols, income Wasserstein 0.49 > 0.1). PSI/drift concept understood from first principles first.
- [x] **6b. Real monitoring loop** — `src/send_traffic.py` (feeds live server) + `src/monitor.py` (reads logs/requests.jsonl, drift vs training). Learned: thin samples (400) false-alarm; needs ≥~1k rows (see BUGS.md).
- [x] **7. Simulate rot** — recession (income_scale=0.6), 2000 rows. Monitor flipped to 1 drifted: MonthlyIncome 0.486. True positive vs the healthy all-clear at same n. FULL LOOP DONE.
- [x] **8. Fix dead feature** — PercentileClipper (1/99 pct) before scaling. RU revived; PR-AUC 0.294→0.369. Promoted model_clipped.joblib (thr 0.52) to live server, verified end-to-end.
- [x] **9. MLflow** — all 4 runs logged to sqlite:///mlflow.db with full metrics+plots+model. UI live at :5001. Hit cloudpickle + file-store-deprecated gotchas (see D9).

## Next up (queued for 2026-08-28+)
- [x] **10. Explainability — SHAP & LIME** — SHAP taught + served (`/predict?explain=true`, gated + warmed, ~0.04s steady). Waterfall via `src/render_waterfall.py`. LIME (`src/explain_lime.py`) compared: agrees on top drivers, diverges on weak-feature signs & scale. Takeaway: SHAP = auditable accounting, LIME = fast sanity check.
- [x] **11. Version 2 model — XGBoost** — PR-AUC 0.369→0.381 (modest: data is mostly additive). Logged to MLflow, registered v2, moved `@champion` (thr 0.6). Live swap verified. SHAP survived (model-agnostic). Boundary None-coercion added to serve.py (per-model clipper fix didn't cover a clipper-less champion).
- [x] **12. Calibration** — reliability curve (binned + bin-free kernel) showed overconfidence (avg pred 0.31 vs actual 0.067). Compared isotonic vs Platt on test: Platt wins (Brier 0.049, PR-AUC 0.407 preserved; isotonic dinged AUC to 0.388 via step-ties). Explained corrector shapes (2-knob sigmoid vs PAV staircase). Why calibrated probs cap ~0.5: riskiest 1% only default 66%. Promoted Platt-calibrated as v3 @champion, thr 0.091 (1/(1+K) works now). SHAP base fixed 0.32→0.049.
- [x] **13. Retrain-on-drift** — champion/challenger guardrail (`src/retrain_on_drift.py`, `src/retrain_concept.py`). KEY LESSON: input drift ≠ performance drop. Covariate drift (income×0.6) didn't hurt the tree champion (0.425), guardrail KEPT it. Concept drift (fabricated new rule) crashed champion to 0.09, challenger 0.79 → auto-promoted v4 → rolled back to v3 (one alias move). Label lag + rolling window discussed. LOOP CLOSED.

## Learned / pushbacks that changed the design
- **LIVE (2026-09-02):** deployed free at https://huggingface.co/spaces/gauravk26/credit_distress (Gradio, `space/`). HF now charges PRO for free-CPU Gradio/Docker → used free **ZeroGPU**. Gotcha: ZeroGPU needs a `@spaces.GPU` function to EXIST at startup; wrapping the real CPU-fn crashed on figure serialization, so a dummy `@spaces.GPU` stub satisfies it while `score()` runs on CPU. Verified end-to-end (typical borrower 2.2% CLEAR + SHAP waterfall render).
- **Deploy + CD (2026-09-02):** `serving/` = self-contained Docker app (app.py + bundled champion snapshot + HTML frontend + Dockerfile + HF README metadata), decoupled from MLflow. `src/export_champion.py` freezes @champion → serving/ (reads MLFLOW_TRACKING_URI from env for CD). Verified locally on :7860 (form → prob + SHAP waterfall; validation 422). `.github/workflows/cd.yml` deploys to HF Space on promotion (repository_dispatch champion-promoted) — gated by the guardrail, needs HF_TOKEN/SPACE_ID secrets. Top-level README.md (architecture + mermaid diagram) written. Last mile: user creates HF Space + pushes serving/.
- **CI/CD (2026-08-28):** Pipeline 1 = `.github/workflows/ci.yml` runs `tests/` (test_clipper: guards the None bug; test_pipeline: every recipe fits+predicts) on push; `requirements.txt` added. Pipeline 2 = `src/auto_retrain.py` wires `monitor.check_drift()` → `retrain_on_drift.evaluate_challenger()` (guardrail decides). Verified: drift on income fired the check, guardrail KEPT champion (0.425 vs 0.409). Refactor pattern: split compute (returns data) from print, so functions are reusable. Last mile = a cron/schedule trigger.
- **Input validation (2026-08-28):** added `Field(ge=0)`/`age<=120` constraints to serve.py `BorrowerFeatures`. Impossible values → 422 with field-level reason; unusual-but-real values still predict. Resolves the silent-clamping smell in BUGS.md.
- **Refactor (2026-08-28):** collapsed 6 training scripts (train + weighted/clipped/xgb/mlflow/all_mlflow) into one parametrized `src/train.py` with a RECIPES registry + `build_pipeline`. CLI: `train.py <recipe|all> [--mlflow]`. Verified by reproducing PR-AUC 0.241/0.294/0.369/0.381 exactly. Threshold-derivation left OUT (a serving concern, not training).

## Open questions / parked
- Which drift metric (PSI vs KS vs KL) — decide at step 6.
