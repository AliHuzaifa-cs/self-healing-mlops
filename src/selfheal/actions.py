"""
THE ONLY functions the self-healing system (and later, the AI agent) is
allowed to call. No arbitrary shell commands — everything goes through here.
"""

import subprocess
import mlflow
from mlflow.tracking import MlflowClient
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

MODEL_NAME = "luck_direction_model"
PRODUCTION_ALIAS = "production"
CONTAINER_NAME = "luck-prediction-api"

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")


def restart_service(container_name: str = CONTAINER_NAME) -> dict:
    """Bring a container back up, regardless of how it went down."""
    print(f"[ACTION] restart_service({container_name})")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--no-deps", container_name.replace("luck-prediction-", "")],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    success = result.returncode == 0
    return {"action": "restart_service", "success": success, "output": result.stdout or result.stderr}


def retrain_model() -> dict:
    """Train a new model version and register it in MLflow (NOT promoted yet)."""
    print("[ACTION] retrain_model()")
    import train_mlflow
    run_id, metrics = train_mlflow.run_experiment()
    return {"action": "retrain_model", "run_id": run_id, "metrics": metrics}


def validate_model(new_run_id: str, min_improvement: float = 0.01) -> dict:
    """Compare the new run's F1 against the current production version's F1."""
    print(f"[ACTION] validate_model({new_run_id})")
    client = MlflowClient()
    new_metrics = mlflow.get_run(new_run_id).data.metrics

    try:
        prod_version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        prod_metrics = mlflow.get_run(prod_version.run_id).data.metrics
    except Exception:
        # No production model yet -> anything passes
        prod_metrics = {"f1": 0.0}

    improved = new_metrics.get("f1", 0.0) >= prod_metrics.get("f1", 0.0) + min_improvement
    return {
        "action": "validate_model",
        "new_f1": new_metrics.get("f1"),
        "production_f1": prod_metrics.get("f1"),
        "improved": improved,
    }


def promote_model(run_id: str) -> dict:
    """Promote a validated model version to 'production' alias."""
    print(f"[ACTION] promote_model({run_id})")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
    if not versions:
        return {"action": "promote_model", "success": False, "reason": "version not found"}
    version_number = versions[0].version
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version_number)
    return {"action": "promote_model", "success": True, "version": version_number}


def rollback_model() -> dict:
    """Point 'production' alias back to the previous version."""
    print("[ACTION] rollback_model()")
    client = MlflowClient()
    all_versions = sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        current_idx = next(i for i, v in enumerate(all_versions) if v.version == current.version)
        if current_idx == 0:
            return {"action": "rollback_model", "success": False, "reason": "no earlier version exists"}
        previous = all_versions[current_idx - 1]
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, previous.version)
        return {"action": "rollback_model", "success": True, "reverted_to_version": previous.version}
    except Exception as e:
        return {"action": "rollback_model", "success": False, "reason": str(e)}


def mark_incident_resolved(incident_id: int, result: str) -> dict:
    from incident_store import update_incident
    update_incident(incident_id, status="resolved", result=result)
    return {"action": "mark_incident_resolved", "incident_id": incident_id}