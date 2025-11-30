from pathlib import Path
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")


def add_sentiment():
    """
    Read news_tsla.csv (Tesla tweets/news), ensure we have a sentiment score,
    aggregate to daily average sentiment, and save to data/processed/daily_sentiment_tsla.csv.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    news_path = RAW_DIR / "news_tsla.csv"
    news = pd.read_csv(news_path)

    # --- 1. Detect date column ---
    date_col = None
    for col in ["date", "created_at", "tweet_date", "Date", "datetime", "time"]:
        if col in news.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            f"Could not find a date-like column in {news_path}. "
            f"Columns found: {list(news.columns)}"
        )

    # --- 2. Detect if dataset already has sentiment ---
    sentiment_col = None
    for col in ["sentiment", "Sentiment", "label", "polarity", "SentimentScore"]:
        if col in news.columns:
            sentiment_col = col
            break

    if sentiment_col is not None:
        print("Dataset already has sentiment — using existing values.")
        news["sentiment"] = news[sentiment_col]

        # If sentiment is text like Positive/Negative/Neutral, map to numbers
        if news["sentiment"].dtype == object:
            news["sentiment"] = news["sentiment"].replace(
                {
                    "Positive": 1,
                    "Negative": -1,
                    "Neutral": 0,
                    "positive": 1,
                    "negative": -1,
                    "neutral": 0,
                }
            )
    else:
        # --- 3. No sentiment column → compute using VADER ---
        # Detect text column
        text_col = None
        for col in ["headline", "text", "clean_text", "tweet", "Tweet", "content"]:
            if col in news.columns:
                text_col = col
                break

        if text_col is None:
            raise ValueError(
                f"Could not find a text/headline column in {news_path}. "
                f"Columns found: {list(news.columns)}"
            )

        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

        news[text_col] = news[text_col].astype(str)
        news["sentiment"] = news[text_col].apply(
            lambda t: sia.polarity_scores(t)["compound"]
        )

    # --- 4. Normalize date ---
    news["date"] = pd.to_datetime(news[date_col]).dt.date

    # --- 5. Aggregate to daily sentiment ---
    daily_sent = (
        news.groupby("date")["sentiment"]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    out_path = PROC_DIR / "daily_sentiment_tsla.csv"
    daily_sent.to_csv(out_path, index=False)
    print(f"Saved daily sentiment to {out_path}")
    return daily_sent


if __name__ == "__main__":
    add_sentiment()
