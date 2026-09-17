"""
Step 6b: Data drift detection using Evidently.
Compares REFERENCE (older/training-time data) vs CURRENT (recent data)
distributions for our engineered features.
"""

import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_drift_report.html"

FEATURE_COLUMNS = [
    "return_lag1", "return_lag2", "return_lag3",
    "price_vs_ma5", "price_vs_ma20",
    "volatility_5", "volatility_10",
    "volume_change", "volume_vs_ma5",
    "hl_range",
]


def load_split_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # Reference = everything except last 60 rows ("training-time" distribution)
    # Current   = last 60 rows (simulated "production" data)
    reference = df.iloc[:-60][FEATURE_COLUMNS]
    current = df.iloc[-60:][FEATURE_COLUMNS]
    return reference, current


def run_drift_check():
    reference, current = load_split_data()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=None)

    report.save_html(str(REPORT_PATH))

    # Extract summary as JSON to check programmatically (for later self-healing thresholds)
    result_json = report.as_dict()
    drift_summary = result_json["metrics"][0]["result"]

    print("--- Data Drift Report ---")
    print(f"Dataset drift detected: {drift_summary['dataset_drift']}")
    print(f"Number of drifted columns: {drift_summary['number_of_drifted_columns']} / {drift_summary['number_of_columns']}")
    print(f"\nFull HTML report saved to: {REPORT_PATH}")

    return drift_summary


if __name__ == "__main__":
    run_drift_check()