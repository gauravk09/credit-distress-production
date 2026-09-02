"""Monitoring job: read the live request log and check it for drift against
the training data. This is the real loop — drift measured on actual traffic,
not a synthetic table.

Reference = a sample of the training data (the world the model learned).
Current   = the features from logs/requests.jsonl (the world it's seeing now).
"""
import json
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
]


def load_live_features(path="logs/requests.jsonl"):
    rows = [json.loads(l)["features"] for l in open(path) if l.strip()]
    return pd.DataFrame(rows)[FEATURES]


def check_drift():
    """Return (n_live, drifted) where drifted is a list of (column, score).
    Reusable so an orchestrator can act on the result, not just print it."""
    reference = pd.read_csv("data/credit.csv")[FEATURES].sample(5000, random_state=1)
    current = load_live_features()

    dd = DataDefinition(numerical_columns=FEATURES)
    ref_ds = Dataset.from_pandas(reference, data_definition=dd)
    cur_ds = Dataset.from_pandas(current, data_definition=dd)

    result = Report([DataDriftPreset()]).run(cur_ds, ref_ds)
    result.save_html("drift/monitor_report.html")

    drifted = []
    for m in result.dict()["metrics"]:
        name = m["metric_name"]
        if name.startswith("ValueDrift") and m["value"] > _threshold(name):
            drifted.append((name.split("column=")[1].split(",")[0], round(m["value"], 3)))
    return len(current), drifted


def main():
    n_live, drifted = check_drift()
    print(f"live rows checked : {n_live}")
    print(f"drifted columns   : {len(drifted)} of {len(FEATURES)}")
    for col, val in drifted:
        print(f"   DRIFT -> {col}: {val}")
    if not drifted:
        print("   all clear — live traffic looks like training data")
    print("saved drift/monitor_report.html")


def _threshold(name):
    return float(name.split("threshold=")[1].rstrip(")"))


if __name__ == "__main__":
    main()
