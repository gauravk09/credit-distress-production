"""Send pretend live traffic to the running server so the request log fills up.
Samples real borrowers from the dataset. Optional income_scale simulates a
recession (e.g. 0.6 = everyone earns 40% less) to inject drift later.

Usage: python3 src/send_traffic.py [n] [income_scale]
"""
import sys, json, urllib.request
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
INCOME_SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
URL = "http://localhost:8137/predict"

COL_TO_FIELD = {
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age", "NumberOfTime30-59DaysPastDueNotWorse": "times_30_59_days_late",
    "DebtRatio": "debt_ratio", "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "times_90_days_late",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "times_60_89_days_late",
    "NumberOfDependents": "dependents",
}

df = pd.read_csv("data/credit.csv").drop(columns=["FinancialDistressNextTwoYears"])
sample = df.sample(N, random_state=7)

sent = 0
for _, r in sample.iterrows():
    payload = {}
    for col, field in COL_TO_FIELD.items():
        v = r[col]
        if col == "MonthlyIncome" and pd.notna(v):
            v = v * INCOME_SCALE
        payload[field] = None if pd.isna(v) else float(v)
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req).read()
    sent += 1

print(f"sent {sent} requests (income_scale={INCOME_SCALE})")
