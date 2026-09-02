---
title: Credit Distress Model
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# Credit Distress Model — live demo

A calibrated XGBoost classifier estimating a borrower's risk of financial distress,
with per-prediction **SHAP** explanations. Enter a borrower and score them; the chart
shows which facts pushed the risk up (red) or down (green).

- **Model:** XGBoost + Platt calibration (chosen via a champion/challenger process).
- **Snapshot:** frozen from an MLflow registry — no tracking server needed at runtime.
- **Guardrail:** impossible inputs (negatives, age > 120) are rejected, not silently clamped.

Trained on the "Give Me Some Credit" dataset. Full training, drift-monitoring, and
retrain-on-drift pipeline live in the project repository.
