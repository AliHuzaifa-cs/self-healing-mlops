"""
Step 2b: Train baseline classification model (next-day direction: up/down).
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def split_data(df: pd.DataFrame):
    # Time-series data — RANDOM split use nahi karenge (future data leak ho jayega).
    # Chronological split: purana data train, recent data test.
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    print(f"Train: {len(X_train)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test:  {len(X_test)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, preds)}")

    return metrics


def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)