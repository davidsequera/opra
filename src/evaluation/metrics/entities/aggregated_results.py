from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AggregatedResults:
    """Mean ± 95% CI across K simulation runs for one policy-log combination."""
    policy_name: str
    log_name: str
    num_runs: int

    # Performance: mean ± ci
    compliance_rates_mean: Dict[str, float] = field(default_factory=dict)
    compliance_rates_ci: Dict[str, float] = field(default_factory=dict)
    cir_mean: Dict[str, float] = field(default_factory=dict)
    cir_ci: Dict[str, float] = field(default_factory=dict)

    avg_cycle_time_mean: float = 0.0
    avg_cycle_time_ci: float = 0.0
    median_cycle_time_mean: float = 0.0
    median_cycle_time_ci: float = 0.0
    std_cycle_time_mean: float = 0.0
    resource_utilization_cv_mean: Optional[float] = None

    # Similarity: mean ± ci
    similarity_mean: Dict[str, float] = field(default_factory=dict)
    similarity_ci: Dict[str, float] = field(default_factory=dict)
