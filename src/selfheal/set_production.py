"""
One-time utility: set the current latest model version as 'production'.
Run this once before using orchestrator.py for the first time.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "luck_direction_model"

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")

client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = sorted(versions, key=lambda v: int(v.version))[-1]
client.set_registered_model_alias(MODEL_NAME, "production", latest.version)
print(f"Set production alias to version {latest.version}")