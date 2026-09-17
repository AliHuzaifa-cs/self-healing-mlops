"""
Step 6c: Model performance monitoring + a simple concept-drift signal.
Checks rolling accuracy over recent data vs the original test-set baseline.
"""

import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.pkl"

FEATURE_COLUMNS = [
    "return_lag1", "return_lag2", "return_lag3",
    "price_vs_ma5", "price_vs_ma20",
    "volatility_5", "volatility_10",
    "volume_change", "volume_vs_ma5",
    "hl_range",
]

# Baseline accuracy from Step 2/3's original test split
BASELINE_ACCURACY = 0.4953
DEGRADATION_THRESHOLD = 0.05  # if accuracy drops more than this vs baseline -> flag


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_recent_data(n=60):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df.tail(n)


def check_performance():
    model = load_model()
    recent = load_recent_data()

    X_recent = recent[FEATURE_COLUMNS]
    y_recent = recent["target"]

    preds = model.predict(X_recent)
    recent_accuracy = accuracy_score(y_recent, preds)

    drop = BASELINE_ACCURACY - recent_accuracy
    concept_drift_flag = drop > DEGRADATION_THRESHOLD

    print("--- Performance / Concept Drift Check ---")
    print(f"Baseline accuracy: {BASELINE_ACCURACY:.4f}")
    print(f"Recent accuracy (last {len(recent)} rows): {recent_accuracy:.4f}")
    print(f"Accuracy drop: {drop:.4f}")
    print(f"Concept drift flag: {concept_drift_flag}")

    return {
        "baseline_accuracy": BASELINE_ACCURACY,
        "recent_accuracy": recent_accuracy,
        "drop": drop,
        "concept_drift_flag": concept_drift_flag,
    }


if __name__ == "__main__":
    check_performance()