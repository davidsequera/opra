from typing import List, Optional

import numpy as np

from ..entities.episode_metrics import EpisodeMetrics


def compute_episode_metrics(
    episode: int,
    total_reward: float,
    num_steps: int,
    cycle_times: List[float],
    sla_threshold: float,
    episode_duration_sec: float,
    resource_utilizations: Optional[List[float]] = None,
) -> EpisodeMetrics:
    """
    Build an EpisodeMetrics from raw simulation outputs.

    Args:
        cycle_times: list of cycle times (in seconds) for each completed case.
        sla_threshold: the T used for compliance.
        resource_utilizations: optional list of per-resource utilization ratios.
    """
    ct = np.array(cycle_times) if cycle_times else np.array([0.0])
    num_cases = len(ct)
    num_compliant = int(np.sum(ct < sla_threshold))

    util_cv = None
    if resource_utilizations and len(resource_utilizations) > 1:
        mean_u = np.mean(resource_utilizations)
        if mean_u > 0:
            util_cv = float(np.std(resource_utilizations) / mean_u)

    return EpisodeMetrics(
        episode=episode,
        total_reward=total_reward,
        num_steps=num_steps,
        num_cases=num_cases,
        num_compliant=num_compliant,
        sla_compliance_rate=num_compliant / max(num_cases, 1),
        avg_cycle_time=float(np.mean(ct)),
        median_cycle_time=float(np.median(ct)),
        std_cycle_time=float(np.std(ct)),
        min_cycle_time=float(np.min(ct)),
        max_cycle_time=float(np.max(ct)),
        p75_cycle_time=float(np.percentile(ct, 75)),
        p90_cycle_time=float(np.percentile(ct, 90)),
        p95_cycle_time=float(np.percentile(ct, 95)),
        episode_duration_sec=episode_duration_sec,
        resource_utilization_cv=util_cv,
    )
