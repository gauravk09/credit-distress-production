---
title: Credit Distress Model
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Credit Distress Model — live demo

A calibrated XGBoost classifier that estimates a borrower's risk of financial distress,
served behind FastAPI with per-prediction SHAP explanations.

- **Model:** XGBoost + Platt (sigmoid) calibration, chosen via a champion/challenger process.
- **Serving:** self-contained snapshot of the MLflow-registry champion — no tracking server needed at runtime.
- **Endpoints:** `GET /` (demo UI), `GET /health`, `POST /predict?explain=true`.
- **Guardrails:** input validation rejects impossible values (422); explanations are SHAP (auditable).

Enter a borrower on the home page and score them; the waterfall shows why.

Trained on the "Give Me Some Credit" dataset. Full training, drift-monitoring, and
retrain-on-drift pipeline live in the parent project repository.
