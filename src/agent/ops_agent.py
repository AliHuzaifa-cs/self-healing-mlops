"""
The AI Ops Agent: reads a structured incident + history, asks a local LLM
(via Ollama) to diagnose it and recommend ONE allowed action, then executes
that action through the tools.py whitelist. Never lets the LLM run free text
as a command -- its output is parsed as JSON and matched against known actions.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "selfheal"))

import ollama
from incident_store import get_incident, get_recent_incidents, update_incident
import tools

MODEL_NAME = "llama3.2:3b"

ALLOWED_ACTIONS_DESC = """
- restart_service: use when a container/service is down or crashed
- retrain_model: use when data drift or concept drift was detected
- rollback_model: use when the current production model itself seems to be the problem
- no_action: use when the incident doesn't require automated remediation (e.g. informational only)
"""

SYSTEM_PROMPT = f"""You are an AI Ops Agent for a self-healing MLOps system monitoring a stock \
price direction prediction model (LUCK / Lucky Cement, PSX).

You will be given a JSON description of a current incident, plus recent incident history for context.

You must respond with ONLY a JSON object (no other text) in this exact format:
{{
  "root_cause": "<your short root-cause analysis, 1-2 sentences>",
  "recommended_action": "<one of: restart_service, retrain_model, rollback_model, no_action>",
  "reasoning": "<why this action, 1-2 sentences>"
}}

Allowed actions and when to use them:
{ALLOWED_ACTIONS_DESC}

Do not recommend any action other than the ones listed. Do not include markdown formatting or explanation outside the JSON.
"""


def build_context(incident_id: int) -> dict:
    incident = get_incident(incident_id)
    if incident is None:
        raise ValueError(f"Incident #{incident_id} not found")

    history = [i for i in get_recent_incidents(limit=5) if i["id"] != incident_id]

    return {
        "current_incident": {
            "type": incident["incident_type"],
            "description": incident["description"],
            "metrics": json.loads(incident["metrics_json"]) if incident["metrics_json"] else {},
        },
        "recent_history": [
            {"type": h["incident_type"], "root_cause": h["root_cause"],
             "action_taken": h["action_taken"], "result": h["result"], "status": h["status"]}
            for h in history
        ],
    }


def diagnose(context: dict) -> dict:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, indent=2)},
        ],
        format="json",
        options={"temperature": 0.2},
    )
    content = response["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"root_cause": "Could not parse LLM response", "recommended_action": "no_action",
                   "reasoning": content[:200]}
    return parsed


def diagnose_and_act(incident_id: int) -> dict:
    print(f"\n[AI AGENT] Analyzing incident #{incident_id}...")
    context = build_context(incident_id)

    diagnosis = diagnose(context)
    action_name = diagnosis.get("recommended_action", "no_action")

    print(f"[AI AGENT] Root cause: {diagnosis.get('root_cause')}")
    print(f"[AI AGENT] Recommended action: {action_name}")
    print(f"[AI AGENT] Reasoning: {diagnosis.get('reasoning')}")

    result = tools.execute_tool(action_name)

    # If retrain was chosen, follow through with validate (agent doesn't need to
    # know HOW validation works -- that's still handled by controlled code, not the LLM)
    if action_name == "retrain_model" and "run_id" in result:
        validation = tools.execute_tool("validate_model", new_run_id=result["run_id"])
        if validation.get("improved"):
            promote_result = tools.execute_tool("promote_model", run_id=result["run_id"]) \
                if "promote_model" in tools.ALLOWED_ACTIONS else actions_promote_fallback(result["run_id"])
            result["validation"] = validation
            result["promotion"] = promote_result
        else:
            result["validation"] = validation

    update_incident(
        incident_id,
        root_cause=diagnosis.get("root_cause"),
        action_taken=f"AI Agent -> {action_name}",
        result=json.dumps(result, default=str)[:500],
        status="resolved",
    )

    print(f"[AI AGENT] Incident #{incident_id} updated.\n")
    return {"diagnosis": diagnosis, "execution_result": result}


def actions_promote_fallback(run_id):
    import actions
    return actions.promote_model(run_id)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python ops_agent.py <incident_id>")
    else:
        diagnose_and_act(int(_sys.argv[1]))