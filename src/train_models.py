from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

PROC_DIR = Path("data/processed")


def load_data():
    df = pd.read_csv(PROC_DIR / "features_tsla.csv")

    feature_cols = [
        "return_1d",
        "return_5d",
        "sentiment_lag1",
        "sentiment_roll3",
    ]

    X = df[feature_cols]
    y = df["target_up"]

    # Time-series style split: no shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()

    print("=== Logistic Regression ===")
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("\n=== XGBoost ===")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    y_pred2 = xgb.predict(X_test)
    print(classification_report(y_test, y_pred2))


if __name__ == "__main__":
    train_and_evaluate()
