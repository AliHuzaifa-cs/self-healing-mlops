"""
Central place for ALL self-healing decision thresholds.
Change values here, not scattered across scripts.
"""

# Data quality: any issue at all triggers an incident
DATA_QUALITY_MAX_ISSUES = 0

# Data drift: Evidently's own "dataset_drift" boolean is used directly
# (no separate threshold needed here — it's a pass/fail from the report)

# Concept drift / performance degradation
PERFORMANCE_DEGRADATION_THRESHOLD = 0.05  # accuracy drop vs baseline

# Model validation: how much better must a retrained model be to get promoted?
MIN_IMPROVEMENT_TO_PROMOTE = 0.01  # at least +1% F1 over current production

# Container health: consecutive failed health checks before declaring "down"
CONTAINER_UNHEALTHY_RETRIES = 3