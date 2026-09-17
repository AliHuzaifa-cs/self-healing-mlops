"""
Step 6a: Custom data quality checks (no external dependency — reliable & simple).
Checks the CURRENT (simulated production) data for common issues.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"

EXPECTED_COLUMNS = [
    "date", "open", "high", "low", "close", "volume",
    "return_lag1", "return_lag2", "return_lag3",
    "price_vs_ma5", "price_vs_ma20",
    "volatility_5", "volatility_10",
    "volume_change", "volume_vs_ma5",
    "hl_range", "target",
]


def load_current_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # "Current" = most recent 60 rows, simulating live production data
    return df.tail(60).reset_index(drop=True)


def check_data_quality(df: pd.DataFrame) -> dict:
    issues = []

    # 1. Schema check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # 2. Nulls
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Null values found: {cols_with_nulls.to_dict()}")

    # 3. Duplicates
    dup_count = df.duplicated(subset=["date"]).sum()
    if dup_count > 0:
        issues.append(f"Duplicate dates: {dup_count}")

    # 4. Out-of-range values (prices should never be <= 0)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns and (df[col] <= 0).any():
            issues.append(f"Non-positive values in {col}")

    # 5. Volume should never be negative
    if "volume" in df.columns and (df["volume"] < 0).any():
        issues.append("Negative volume values found")

    # 6. Wrong data types (dates should actually be datetime)
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        issues.append("'date' column is not datetime type")

    result = {
        "rows_checked": len(df),
        "issues_found": len(issues),
        "issues": issues,
        "passed": len(issues) == 0,
    }
    return result


if __name__ == "__main__":
    current = load_current_data()
    result = check_data_quality(current)

    print("--- Data Quality Report ---")
    print(f"Rows checked: {result['rows_checked']}")
    print(f"Passed: {result['passed']}")
    if result["issues"]:
        print("Issues found:")
        for issue in result["issues"]:
            print(f"  - {issue}")
    else:
        print("No issues found.")