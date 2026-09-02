"""Lean experiment scoreboard.

evaluate_and_record(...) is called once per experiment. It:
  1. computes the metrics we care about (PR-AUC + recall/precision at a threshold),
  2. appends one row to runs/scoreboard.csv,
  3. saves 3 plots under runs/<name>/ so each iteration is visible, not eyeballed.

We always pass VALIDATION-set predictions here during development.
Test set is scored once, at the very end.
"""
import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # no display needed, just write PNGs
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve,
)

RUNS_DIR = "runs"
SCOREBOARD = os.path.join(RUNS_DIR, "scoreboard.csv")


def evaluate_and_record(name, params, y_true, proba, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, proba)
    prevalence = y_true.mean()
    rec = recall_score(y_true, pred, zero_division=0)
    prec = precision_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": name, "threshold": round(float(threshold), 4),
        "pr_auc": round(float(pr_auc), 4),
        "recall": round(float(rec), 4), "precision": round(float(prec), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(len(y_true)), "positives": int(y_true.sum()),
        "params": json.dumps(params),
    }
    _append_scoreboard(row)
    _save_plots(name, y_true, proba, threshold, pr_auc, prevalence)

    print(f"[{name}] PR-AUC={pr_auc:.3f} (base {prevalence:.3f}) | "
          f"recall={rec:.3f} precision={prec:.3f} @thr={threshold} | "
          f"caught {tp}/{tp+fn}, flagged {tp+fp}")
    return row


def _append_scoreboard(row):
    os.makedirs(RUNS_DIR, exist_ok=True)
    df = pd.DataFrame([row])
    header = not os.path.exists(SCOREBOARD)
    df.to_csv(SCOREBOARD, mode="a", header=header, index=False)


def _save_plots(name, y_true, proba, threshold, pr_auc, prevalence):
    d = os.path.join(RUNS_DIR, name)
    os.makedirs(d, exist_ok=True)

    # 1. Precision-Recall curve (the whole ranking, all thresholds at once)
    prec_c, rec_c, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec_c, prec_c, lw=2)
    ax.axhline(prevalence, ls="--", c="grey", lw=1, label=f"lazy baseline {prevalence:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"{name}\nPR curve (PR-AUC={pr_auc:.3f})", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(d, "pr_curve.png"), dpi=110); plt.close(fig)

    # 2. Confusion matrix at this threshold (the four boxes).
    #    Layout: columns = ACTUAL, rows = PREDICTED, Positive before Negative.
    #        | Actual P | Actual N
    #    Pred P |   TP    |   FP
    #    Pred N |   FN    |   TN
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    M = np.array([[tp, fp],
                  [fn, tn]])
    tags = [["TP", "FP"],
            ["FN", "TN"]]
    fig, ax = plt.subplots(figsize=(4.4, 4))
    im = ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tags[i][j]}\n{M[i,j]:,}", ha="center", va="center",
                    color="black", fontsize=11)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Actual P", "Actual N"], yticklabels=["Pred P", "Pred N"],
           title=f"{name}\nconfusion @thr={threshold}")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_xlabel("ACTUAL"); ax.set_ylabel("PREDICTED")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(d, "confusion.png"), dpi=110); plt.close(fig)

    # 3. Probability histogram by true class (shows WHERE the threshold sits)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(proba[y_true == 0], bins=50, alpha=0.6, label="true No", density=True)
    ax.hist(proba[y_true == 1], bins=50, alpha=0.6, label="true Yes", density=True)
    ax.axvline(threshold, c="red", lw=1.5, label=f"threshold {threshold}")
    ax.set(xlabel="predicted probability of distress", ylabel="density",
           title=f"{name}\nscore distribution by true class")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(d, "score_hist.png"), dpi=110); plt.close(fig)
