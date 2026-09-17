"""
Presentation-friendly viewer for the incident history.
Run this anytime to see a clean table of everything the self-healing
system has detected and done so far.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "selfheal"))

from incident_store import get_recent_incidents


def print_incident_table(limit=20):
    incidents = get_recent_incidents(limit=limit)

    if not incidents:
        print("No incidents recorded yet.")
        return

    print(f"\n{'='*100}")
    print(f"{'ID':<4} {'Type':<16} {'Status':<12} {'Action Taken':<25} {'Result (truncated)':<40}")
    print(f"{'='*100}")

    for inc in reversed(incidents):  # oldest first, easier to follow the story
        result = (inc["result"] or "")[:38]
        action = (inc["action_taken"] or "")[:23]
        print(f"{inc['id']:<4} {inc['incident_type']:<16} {inc['status']:<12} {action:<25} {result:<40}")

    print(f"{'='*100}\n")


def print_incident_detail(incident_id: int):
    from incident_store import get_incident
    inc = get_incident(incident_id)
    if inc is None:
        print(f"Incident #{incident_id} not found.")
        return

    print(f"\n--- Incident #{inc['id']} ---")
    print(f"Created:      {inc['created_at']}")
    print(f"Type:         {inc['incident_type']}")
    print(f"Description:  {inc['description']}")
    print(f"Root Cause:   {inc['root_cause']}")
    print(f"Action Taken: {inc['action_taken']}")
    print(f"Result:       {inc['result']}")
    print(f"Status:       {inc['status']}")
    if inc["metrics_json"]:
        print(f"Metrics:      {json.dumps(json.loads(inc['metrics_json']), indent=2)}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_incident_detail(int(sys.argv[1]))
    else:
        print_incident_table()