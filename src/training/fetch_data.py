import psxdata
import pandas as pd
from datetime import date
from pathlib import Path

TICKER = "LUCK"
START_DATE = "2018-01-01"
END_DATE = date.today().isoformat()

# Path resolved relative to THIS FILE's location, not the current working directory.
# Isliye ab yeh chale ga chahe aap root se run karo ya src/training se.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "reference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # folder na ho to khud bana lega
OUTPUT_PATH = OUTPUT_DIR / "luck_historical.csv"


def fetch_historical_data():
    print(f"Fetching {TICKER} data from {START_DATE} to {END_DATE}...")
    df = psxdata.stocks(TICKER, start=START_DATE, end=END_DATE)

    if df is None or df.empty:
        raise ValueError("No data returned. Check ticker symbol or date range.")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def validate_data(df: pd.DataFrame):
    print("\n--- Basic Data Validation ---")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {list(df.columns)}")

    null_counts = df.isnull().sum()
    print(f"\nNull values per column:\n{null_counts}")

    duplicate_dates = df['date'].duplicated().sum()
    print(f"\nDuplicate dates: {duplicate_dates}")

    print(f"\nClose price range: {df['close'].min()} - {df['close'].max()}")
    print(f"Any zero/negative prices: {(df['close'] <= 0).sum()}")


def save_data(df: pd.DataFrame):
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    data = fetch_historical_data()
    validate_data(data)
    save_data(data)
    print("\nStep 1 complete. Preview:")
    print(data.head())
    print(data.tail())