"""
Persistent incident history (SQLite). This is what Stage 10 will feed
to the AI Ops Agent as historical context.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "incidents" / "incidents.db"


def _get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            incident_type TEXT,
            description TEXT,
            metrics_json TEXT,
            root_cause TEXT,
            action_taken TEXT,
            result TEXT,
            status TEXT
        )
    """)
    return conn


def create_incident(incident_type: str, description: str, metrics: dict) -> int:
    conn = _get_connection()
    cur = conn.execute(
        "INSERT INTO incidents (created_at, incident_type, description, metrics_json, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), incident_type, description, json.dumps(metrics), "open"),
    )
    conn.commit()
    incident_id = cur.lastrowid
    conn.close()
    print(f"[Incident #{incident_id}] Created: {incident_type} — {description}")
    return incident_id


def update_incident(incident_id: int, root_cause: str = None, action_taken: str = None,
                     result: str = None, status: str = None):
    conn = _get_connection()
    fields, values = [], []
    for col, val in [("root_cause", root_cause), ("action_taken", action_taken),
                      ("result", result), ("status", status)]:
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)
    if fields:
        values.append(incident_id)
        conn.execute(f"UPDATE incidents SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
    conn.close()
    print(f"[Incident #{incident_id}] Updated: status={status}, result={result}")


def get_recent_incidents(limit: int = 5) -> list:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT id, created_at, incident_type, description, root_cause, action_taken, result, status "
        "FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    columns = ["id", "created_at", "incident_type", "description", "root_cause", "action_taken", "result", "status"]
    return [dict(zip(columns, row)) for row in rows]

def get_incident(incident_id: int) -> dict:
    conn = _get_connection()
    row = conn.execute(
        "SELECT id, created_at, incident_type, description, metrics_json, root_cause, action_taken, result, status "
        "FROM incidents WHERE id = ?", (incident_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    columns = ["id", "created_at", "incident_type", "description", "metrics_json", "root_cause", "action_taken", "result", "status"]
    return dict(zip(columns, row))


if __name__ == "__main__":
    # Quick self-test
    iid = create_incident("test", "This is a test incident", {"dummy_metric": 1.0})
    update_incident(iid, root_cause="testing", action_taken="none", result="ok", status="resolved")
    print("\nRecent incidents:")
    for inc in get_recent_incidents():
        print(inc)