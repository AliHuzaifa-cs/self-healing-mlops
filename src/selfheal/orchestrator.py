"""
Ties detection -> incident -> remediation -> validation together.
Rule-based for now (Stage 9 replaces the decision step with an LLM agent).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "monitoring"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "selfheal"))

import thresholds
from incident_store import create_incident, update_incident
import actions

import data_quality_check
import data_drift_check
import performance_check


def check_data_quality():
    current = data_quality_check.load_current_data()
    result = data_quality_check.check_data_quality(current)
    if result["issues_found"] > thresholds.DATA_QUALITY_MAX_ISSUES:
        incident_id = create_incident("data_quality", f"{result['issues_found']} issues found", result)
        update_incident(incident_id, root_cause="Bad data detected in incoming batch",
                         action_taken="Flagged for review (no auto-fix yet)", status="open")
        return incident_id
    return None


def check_data_drift():
    drift_summary = data_drift_check.run_drift_check()
    if drift_summary["dataset_drift"]:
        incident_id = create_incident("data_drift", "Dataset drift detected", drift_summary)
        update_incident(incident_id, root_cause="Feature distributions shifted vs reference",
                         action_taken="Triggering retrain", status="in_progress")
        _handle_retrain_flow(incident_id)
        return incident_id
    return None


def check_model_performance():
    perf = performance_check.check_performance()
    if perf["concept_drift_flag"]:
        incident_id = create_incident("concept_drift", f"Accuracy dropped by {perf['drop']:.4f}", perf)
        update_incident(incident_id, root_cause="Prediction relationship may have changed (concept drift)",
                         action_taken="Triggering retrain", status="in_progress")
        _handle_retrain_flow(incident_id)
        return incident_id
    return None


def _handle_retrain_flow(incident_id: int):
    """Shared retrain -> validate -> promote/reject flow."""
    retrain_result = actions.retrain_model()
    validation = actions.validate_model(retrain_result["run_id"])

    if validation["improved"]:
        promote_result = actions.promote_model(retrain_result["run_id"])
        result_msg = f"Retrained model improved (F1 {validation['production_f1']:.4f} -> {validation['new_f1']:.4f}), promoted."
        update_incident(incident_id, action_taken="retrain_model + promote_model",
                         result=result_msg, status="resolved")
    else:
        result_msg = f"Retrained model did NOT improve (F1 {validation['new_f1']:.4f} vs production {validation['production_f1']:.4f}). Rejected, keeping current production model."
        update_incident(incident_id, action_taken="retrain_model (rejected)",
                         result=result_msg, status="resolved")

    print(f"\n{result_msg}")


def check_container_health():
    import subprocess
    result = subprocess.run(
        ["docker", "inspect", "--format={{.State.Status}}", actions.CONTAINER_NAME],
        capture_output=True, text=True,
    )
    status = result.stdout.strip()
    if status != "running":
        incident_id = create_incident("container_down", f"Container status: {status}", {"status": status})
        update_incident(incident_id, root_cause="Container crashed or was killed",
                         action_taken="Triggering restart_service", status="in_progress")
        restart_result = actions.restart_service()
        update_incident(incident_id, result=str(restart_result), status="resolved" if restart_result["success"] else "failed")
        return incident_id
    return None


def run_all_checks():
    print("=== Running self-healing checks ===\n")
    check_container_health()
    check_data_quality()
    check_data_drift()
    check_model_performance()
    print("\n=== Checks complete ===")


if __name__ == "__main__":
    run_all_checks()