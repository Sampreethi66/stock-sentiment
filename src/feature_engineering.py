from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")


def build_features():
    """Merge prices + sentiment and create ML features + target."""
    # ---- 1. Load data ----
    prices = pd.read_csv(RAW_DIR / "prices_TSLA.csv")
    sent = pd.read_csv(PROC_DIR / "daily_sentiment_tsla.csv")

    # Ensure proper date types
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.date
    sent["date"] = pd.to_datetime(sent["date"]).dt.date

    # 🔍 Make sure Close is numeric (this fixes your error)
    # If there are any bad values like text, they become NaN
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")

    # Optionally you can also check dtypes while debugging:
    # print(prices.dtypes)

    # Drop rows where Close is missing after conversion
    prices = prices.dropna(subset=["Close"])

    # ---- 2. Merge with sentiment ----
    df = prices.merge(sent, left_on="Date", right_on="date", how="left")
    df["sentiment"] = df["sentiment"].fillna(0.0)

    # ---- 3. Create features ----
    # Returns
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)

    # Target: 1 if next-day return > 0, else 0
    df["target_up"] = (df["return_1d"].shift(-1) > 0).astype(int)

    # Sentiment lags / rolling
    df["sentiment_lag1"] = df["sentiment"].shift(1)
    df["sentiment_roll3"] = df["sentiment"].rolling(3).mean()

    # ---- 4. Drop rows with NA from lag/roll/returns ----
    df = df.dropna().reset_index(drop=True)

    # ---- 5. Save ----
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "features_tsla.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved features to {out_path} with {len(df)} rows.")
    return df


if __name__ == "__main__":
    build_features()
