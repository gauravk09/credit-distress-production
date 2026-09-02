"""Single source of truth for the data split.
60% train / 20% validation / 20% test, stratified to keep the 6.68% positive rate
in every part. Fixed seed so runs are comparable across experiments."""
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "FinancialDistressNextTwoYears"
SEED = 42


def load_splits(path="data/credit.csv"):
    df = pd.read_csv(path)
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])

    # First carve off test (20%). Then split the rest into train (60) / val (20).
    X_rest, X_te, y_rest, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=SEED)  # 0.25*0.8 = 0.20

    return X_tr, X_val, X_te, y_tr, y_val, y_te
