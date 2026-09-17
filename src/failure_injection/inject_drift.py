"""
Failure type 2: Data Drift.
Shifts feature distributions in the most recent rows (simulating regime change).
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"

FEATURES_TO_SHIFT = [
    "volatility_5", "volatility_10", "volume_change", "volume_vs_ma5",
]


def inject_drift(n_rows=60, shift_multiplier=4.0):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    recent_idx = df.tail(n_rows).index

    for col in FEATURES_TO_SHIFT:
        # Artificially inflate volatility/volume features — simulates a sudden
        # regime change (e.g. market shock) that alters distributions sharply.
        df.loc[recent_idx, col] = df.loc[recent_idx, col] * shift_multiplier

    df.to_csv(DATA_PATH, index=False)
    print(f"Injected drift: scaled {FEATURES_TO_SHIFT} by {shift_multiplier}x in last {n_rows} rows.")


if __name__ == "__main__":
    inject_drift()