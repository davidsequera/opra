import numpy as np

from .activity import (
    ActivitySelector,
    DRLActivitySelector,
    EmpiricalDMActivitySelector,
    GreedyProbActivitySelector,
    RandomActivitySelector,
)
from .resource import (
    DRLResourceSelector,
    GreedyProcessingTimeResourceSelector,
    RandomResourceSelector,
    ResourceSelector,
)


SUPPORTED_POLICIES = ("RA-RR", "DM-RR", "DM-GR", "DM-DRL", "DRL-DRL")


def build_policy(
    name: str,
    *,
    agent=None,
    processing_time_policy=None,
    rng: np.random.Generator | None = None,
) -> tuple[ActivitySelector, ResourceSelector]:
    """
    Build the (activity_selector, resource_selector) pair for a named policy.
    Naming convention: {activity}-{resource}.

    - "DM-DRL", "DRL-DRL" require `agent` (resource-only for DM-DRL, full for DRL-DRL).
    - "DM-GR" requires `processing_time_policy`: stochastic activity sampling
      (to avoid loops) paired with a greedy-by-expected-processing-time
      resource selector.
    """
    rng = rng or np.random.default_rng()

    if name == "RA-RR":
        return RandomActivitySelector(rng), RandomResourceSelector(rng)
    if name == "DM-RR":
        return EmpiricalDMActivitySelector(rng), RandomResourceSelector(rng)
    if name == "DM-GR":
        if processing_time_policy is None:
            raise ValueError("DM-GR requires processing_time_policy")
        return EmpiricalDMActivitySelector(rng), GreedyProcessingTimeResourceSelector(processing_time_policy)
    if name == "DM-DRL":
        if agent is None:
            raise ValueError("DM-DRL requires a resource-only agent (none provided)")
        return EmpiricalDMActivitySelector(rng), DRLResourceSelector(agent)
    if name == "DRL-DRL":
        if agent is None:
            raise ValueError("DRL-DRL requires the full DRL agent (none provided)")
        return DRLActivitySelector(agent), DRLResourceSelector(agent)

    raise ValueError(f"Unknown policy '{name}'. Supported: {SUPPORTED_POLICIES}")
