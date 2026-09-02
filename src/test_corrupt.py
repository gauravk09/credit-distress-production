"""Send each corrupt/edge-case row to the LIVE server and show what happens.
Reveals how the pipeline (clip -> impute -> scale) tames bad inputs, and where
validation (pydantic) rejects them outright."""
import json, urllib.request, urllib.error
import pandas as pd

FIELDS = ["revolving_utilization", "age", "times_30_59_days_late", "debt_ratio",
          "monthly_income", "open_credit_lines", "times_90_days_late",
          "real_estate_loans", "times_60_89_days_late", "dependents"]

df = pd.read_csv("data/corrupt_sample.csv")
print(f"{'case':<20} {'result'}")
for _, r in df.iterrows():
    payload = {f: (None if pd.isna(r[f]) else float(r[f])) for f in FIELDS}
    req = urllib.request.Request("http://localhost:8137/predict",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
        print(f"{r['case']:<20} prob={resp['probability']:.3f}  {resp['decision']}")
    except urllib.error.HTTPError as e:
        print(f"{r['case']:<20} REJECTED  HTTP {e.code}")
