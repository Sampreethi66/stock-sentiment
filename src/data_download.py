from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data/raw")


def download_prices(ticker="TSLA", start="2018-01-01", end="2024-01-01"):
    """Download daily price data for a ticker and save to CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()  # Make Date a normal column
    out_path = DATA_DIR / f"prices_{ticker}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    download_prices()
