"""
Basic smoke tests for the core ML pipeline.
Run in CI to catch obvious breakages (import errors, schema drift, etc.)
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))


def test_feature_engineering_imports():
    """Confirm the feature engineering module loads without error."""
    import feature_engineering
    assert hasattr(feature_engineering, "engineer_features")


def test_feature_engineering_logic():
    """Run feature engineering on synthetic OHLCV data and check output shape/columns."""
    import feature_engineering

    dummy_data = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": [100 + i for i in range(30)],
        "high": [105 + i for i in range(30)],
        "low": [95 + i for i in range(30)],
        "close": [102 + i for i in range(30)],
        "volume": [1000 + i * 10 for i in range(30)],
    })

    result = feature_engineering.engineer_features(dummy_data)
    result = feature_engineering.clean_features(result)

    assert "target" in result.columns
    assert result["target"].isin([0, 1]).all()
    assert len(result) > 0


def test_train_baseline_imports():
    import train_baseline
    assert hasattr(train_baseline, "train_model")