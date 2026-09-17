"""
Failure type 3: Concept Drift / Model Degradation.
Flips the target label in recent rows — simulates the input-output
relationship changing (what the model learned no longer holds).
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"


def inject_concept_drift(n_rows=40):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    recent_idx = df.tail(n_rows).index
    df.loc[recent_idx, "target"] = 1 - df.loc[recent_idx, "target"]

    df.to_csv(DATA_PATH, index=False)
    print(f"Injected concept drift: flipped 'target' label in last {n_rows} rows.")
    print("Model's learned patterns no longer match reality for this window.")


if __name__ == "__main__":
    inject_concept_drift()