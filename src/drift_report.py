"""Hand the drift job to Evidently.

Reference = the old crowd (what we trained on).
Current   = a pretend new crowd whose income dropped 40%.
Evidently runs a PSI-style test on EVERY feature and builds an HTML report.
"""
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
]

df = pd.read_csv("data/credit.csv")[FEATURES]
reference = df.sample(5000, random_state=1).reset_index(drop=True)   # old crowd
current = reference.copy()
current["MonthlyIncome"] = current["MonthlyIncome"] * 0.6            # new crowd earns 40% less

data_def = DataDefinition(numerical_columns=FEATURES)
ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
cur_ds = Dataset.from_pandas(current, data_definition=data_def)

report = Report([DataDriftPreset()])
result = report.run(cur_ds, ref_ds)         # (current, reference)
result.save_html("drift/evidently_report.html")
print("saved drift/evidently_report.html")

# also print the machine-readable summary so we can see the verdict in the terminal
import json
d = result.dict()
print(json.dumps(d, indent=2)[:1500])
