"""Client demo: call the LIVE /predict endpoint and draw a waterfall purely from
the JSON it returns. Proves the served explanation is enough to render for a user.
"""
import sys, json, urllib.request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BORROWER = {
    "revolving_utilization": 0.95, "age": 24, "times_30_59_days_late": 3,
    "debt_ratio": 0.8, "monthly_income": 2000, "open_credit_lines": 3,
    "times_90_days_late": 1, "real_estate_loans": 0, "times_60_89_days_late": 2,
    "dependents": 2,
}

req = urllib.request.Request("http://localhost:8137/predict?explain=true",
                             data=json.dumps(BORROWER).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read())

base = resp["explanation"]["base_value"]
contribs = resp["explanation"]["contributions"]      # already sorted by impact
prob, decision = resp["probability"], resp["decision"]

# Build the waterfall: start at base, add each contribution in order.
labels, running, cum = [], base, []
bars = []
for c in contribs:
    labels.append(f'{c["feature"]}={c["value"]:g}')
    bars.append(c["shap"]); cum.append(running); running += c["shap"]

fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (start, delta) in enumerate(zip(cum, bars)):
    ax.barh(i, delta, left=start, color=("#d62728" if delta >= 0 else "#1f77b4"))
    ax.text(start + delta + (0.005 if delta >= 0 else -0.005), i,
            f"{delta:+.2f}", va="center", ha="left" if delta >= 0 else "right", fontsize=8)
ax.axvline(base, color="grey", ls="--", lw=1)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("probability of distress")
ax.set_title(f"Why: {decision.upper()} (prob {prob:.2f})\n"
             f"base {base:.2f} → each fact pushes the score", fontsize=10)
fig.tight_layout(); fig.savefig("explain/served_waterfall.png", dpi=110)
print("saved explain/served_waterfall.png from the LIVE API response")
