"""
Failure type 1: Data Quality issues.
Adds nulls, duplicates, and invalid values into the processed dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"


def inject_bad_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # 1. Add nulls into a few random rows/columns
    null_indices = df.sample(5, random_state=1).index
    df.loc[null_indices, "close"] = np.nan

    # 2. Add duplicate rows
    duplicate_rows = df.tail(3)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # 3. Add invalid values (negative price, impossible volume)
    df.loc[df.index[-1], "close"] = -999
    df.loc[df.index[-2], "volume"] = -500

    df.to_csv(DATA_PATH, index=False)
    print(f"Injected: 5 nulls in 'close', 3 duplicate rows, 2 invalid values.")
    print(f"File updated: {DATA_PATH}")


if __name__ == "__main__":
    inject_bad_data()