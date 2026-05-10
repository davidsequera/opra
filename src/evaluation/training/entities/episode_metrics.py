from dataclasses import dataclass
from typing import Optional


@dataclass
class EpisodeMetrics:
    """Metrics collected at the end of each training episode (one full simulation)."""

    episode: int
    total_reward: float
    num_steps: int  # decision points in this episode

    # --- SLA Compliance (primary objective) ---
    num_cases: int
    num_compliant: int
    sla_compliance_rate: float  # CR = num_compliant / num_cases

    # --- Cycle Time Stats ---
    avg_cycle_time: float
    median_cycle_time: float
    std_cycle_time: float
    min_cycle_time: float
    max_cycle_time: float
    p75_cycle_time: float
    p90_cycle_time: float
    p95_cycle_time: float

    # --- Wall-clock ---
    episode_duration_sec: float  # how long the episode took (seconds)

    # --- Optional: resource utilization ---
    resource_utilization_cv: Optional[float] = None  # CV of utilization across resources
