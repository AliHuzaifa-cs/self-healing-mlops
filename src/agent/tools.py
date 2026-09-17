"""
THE WHITELIST. The LLM can only trigger these 5 named actions — nothing else.
No eval(), no exec(), no raw shell access from the agent's output.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "selfheal"))

import actions

ALLOWED_ACTIONS = {
    "restart_service": actions.restart_service,
    "retrain_model": actions.retrain_model,
    "validate_model": actions.validate_model,
    "rollback_model": actions.rollback_model,
    "no_action": lambda **kwargs: {"action": "no_action", "note": "Agent decided no action was needed"},
}


def execute_tool(action_name: str, **kwargs) -> dict:
    if action_name not in ALLOWED_ACTIONS:
        return {"action": action_name, "success": False, "error": f"'{action_name}' is not an allowed action"}
    fn = ALLOWED_ACTIONS[action_name]
    try:
        # Only pass kwargs the function actually accepts (avoids crashing on extra LLM-provided keys)
        import inspect
        sig = inspect.signature(fn)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**filtered_kwargs)
    except Exception as e:
        return {"action": action_name, "success": False, "error": str(e)}