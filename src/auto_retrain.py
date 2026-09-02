"""Automated pipeline 2 (Continuous Training): the monitor triggers the retrain
evaluation, but the GUARDRAIL still decides whether to promote.

    monitor.check_drift() --[drift?]--> retrain_on_drift.evaluate_challenger() --[wins?]--> promote

KEY: input drift only triggers the CHECK. It never auto-promotes — a challenger
that doesn't beat the champion is rejected, even here in the automated path.
"""
from monitor import check_drift
from retrain_on_drift import evaluate_challenger

DRIFT_TRIGGER = 1   # retrain-eval fires when at least this many columns drift


def main():
    n_live, drifted = check_drift()
    print(f"[monitor] {n_live} live rows, {len(drifted)} drifted column(s): "
          f"{[c for c, _ in drifted] or 'none'}")

    if len(drifted) < DRIFT_TRIGGER:
        print("[pipeline] no significant drift -> nothing to do")
        return

    print("[pipeline] drift detected -> evaluating a challenger (guardrail decides)")
    r = evaluate_challenger()
    print(f"[guardrail] champion PR-AUC {r['champion_auc']:.3f}  vs  "
          f"challenger {r['challenger_auc']:.3f}")
    if r["promote"]:
        print("[pipeline] challenger WINS -> promote (register v_next + move @champion)")
    else:
        print("[pipeline] challenger did NOT beat champion -> KEEP champion, alert a human")
        print("           (input drift != model degradation — the guardrail protected production)")


if __name__ == "__main__":
    main()
