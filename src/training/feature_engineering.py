"""
Step 2a: Feature engineering for LUCK stock direction prediction.
Converts raw OHLCV into ML-ready features + binary target (next-day up/down).
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "reference" / "luck_historical.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"


def load_data():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Return-based features ---
    df["daily_return"] = df["close"].pct_change()
    df["return_lag1"] = df["daily_return"].shift(1)
    df["return_lag2"] = df["daily_return"].shift(2)
    df["return_lag3"] = df["daily_return"].shift(3)

    # --- Moving averages ---
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()

    # Price relative to moving average (normalized signal, model ke liye better than raw price)
    df["price_vs_ma5"] = (df["close"] - df["ma_5"]) / df["ma_5"]
    df["price_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]

    # --- Volatility ---
    df["volatility_5"] = df["daily_return"].rolling(window=5).std()
    df["volatility_10"] = df["daily_return"].rolling(window=10).std()

    # --- Volume features ---
    df["volume_change"] = df["volume"].pct_change()
    df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
    df["volume_vs_ma5"] = (df["volume"] - df["volume_ma_5"]) / df["volume_ma_5"]

    # --- High-Low range (intraday volatility proxy) ---
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # --- TARGET: next day close higher than today? (1 = up, 0 = down/flat) ---
    df["next_close"] = df["close"].shift(-1)
    df["target"] = (df["next_close"] > df["close"]).astype(int)

    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    # Rolling windows aur shift() se shuru/end mein NaN aate hain — drop karna zaroori
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN (rolling window warmup + last row with no next-day target)")
    return df


def validate_features(df: pd.DataFrame):
    print("\n--- Feature Validation ---")
    print(f"Final rows: {len(df)}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
    print(f"\nAny NaN remaining: {df.isnull().sum().sum()}")
    print(f"\nFeature columns: {[c for c in df.columns if c not in ['date','next_close']]}")


if __name__ == "__main__":
    raw = load_data()
    features = engineer_features(raw)
    features = clean_features(features)
    validate_features(features)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")