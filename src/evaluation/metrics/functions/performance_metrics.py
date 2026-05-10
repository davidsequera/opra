from typing import Dict

import numpy as np
import pandas as pd

from ..entities.performance_result import PerformanceResult
from .cycle_time import compute_cycle_times
from .compliance import compute_compliance_rate, compute_cir
from .resource_utilization import compute_resource_utilization_cv


def compute_performance_metrics(
    sim_log_df: pd.DataFrame,
    ref_cycle_times: np.ndarray,
    sla_thresholds: Dict[str, float],
    ref_compliance_rates: Dict[str, float],
    case_col: str = "case_id",
    start_col: str = "start_time",
    end_col: str = "end_time",
    resource_col: str = "resource",
) -> PerformanceResult:
    """Compute all performance metrics for one simulated log."""
    sim_ct = compute_cycle_times(sim_log_df, case_col, start_col, end_col)

    compliance_rates = {}
    cir_values = {}
    for label, threshold in sla_thresholds.items():
        cr = compute_compliance_rate(sim_ct, threshold)
        compliance_rates[label] = cr
        cir_values[label] = compute_cir(cr, ref_compliance_rates.get(label, 0.0))

    util_cv = compute_resource_utilization_cv(sim_log_df, resource_col, start_col, end_col)

    return PerformanceResult(
        compliance_rates=compliance_rates,
        compliance_improvement_ratios=cir_values,
        avg_cycle_time=float(np.mean(sim_ct)),
        median_cycle_time=float(np.median(sim_ct)),
        std_cycle_time=float(np.std(sim_ct)),
        min_cycle_time=float(np.min(sim_ct)) if len(sim_ct) > 0 else 0.0,
        max_cycle_time=float(np.max(sim_ct)) if len(sim_ct) > 0 else 0.0,
        p75_cycle_time=float(np.percentile(sim_ct, 75)) if len(sim_ct) > 0 else 0.0,
        p90_cycle_time=float(np.percentile(sim_ct, 90)) if len(sim_ct) > 0 else 0.0,
        p95_cycle_time=float(np.percentile(sim_ct, 95)) if len(sim_ct) > 0 else 0.0,
        resource_utilization_cv=util_cv,
        num_cases=len(sim_ct),
    )
