"""
Step 3: Train baseline model with MLflow experiment tracking.
Same model as Step 2, but now every run's params/metrics/model are logged.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

FEATURE_COLUMNS = [
    "return_lag1", "return_lag2", "return_lag3",
    "price_vs_ma5", "price_vs_ma20",
    "volatility_5", "volatility_10",
    "volume_change", "volume_vs_ma5",
    "hl_range",
]

EXPERIMENT_NAME = "luck_direction_prediction"


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def split_data(df: pd.DataFrame):
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]
    return X_train, X_test, y_train, y_test


def run_experiment(n_estimators=100, max_depth=5):
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    with mlflow.start_run():
        # --- Log parameters ---
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": "balanced",
            "random_state": 42,
        }
        mlflow.log_params(params)

        # --- Train ---
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # --- Evaluate ---
        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }
        mlflow.log_metrics(metrics)

        # --- Log model + register it ---
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="luck_direction_model",
        )

        print("--- Run Complete ---")
        print(f"Params: {params}")
        print(f"Metrics: {metrics}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

        return mlflow.active_run().info.run_id, metrics


if __name__ == "__main__":
    run_experiment()