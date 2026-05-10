from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PerformanceResult:
    """Performance metrics for a single simulated log against reference thresholds."""
    # Per-threshold compliance
    compliance_rates: Dict[str, float]  # e.g. {"T95": 0.98, "T90": 0.95, ...}
    compliance_improvement_ratios: Dict[str, float]  # CIR relative to original

    # Cycle time statistics
    avg_cycle_time: float
    median_cycle_time: float
    std_cycle_time: float
    min_cycle_time: float
    max_cycle_time: float
    p75_cycle_time: float
    p90_cycle_time: float
    p95_cycle_time: float

    # Resource utilization balance
    resource_utilization_cv: Optional[float] = None

    # Metadata
    num_cases: int = 0
    log_path: str = ""
