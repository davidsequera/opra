from typing import List, Tuple

import numpy as np

from ..entities.performance_result import PerformanceResult
from ..entities.similarity_result import SimilarityResult
from ..entities.aggregated_results import AggregatedResults


def mean_and_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute mean and 95% CI using t-distribution."""
    from scipy import stats
    n = len(values)
    if n < 2:
        return (np.mean(values), 0.0)
    mean = np.mean(values)
    se = stats.sem(values)
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_val * se
    return float(mean), float(ci)


def aggregate_results(
    performance_results: List[PerformanceResult],
    similarity_results: List[SimilarityResult],
    policy_name: str,
    log_name: str,
) -> AggregatedResults:
    """Aggregate K runs into mean ± 95% CI."""
    agg = AggregatedResults(
        policy_name=policy_name,
        log_name=log_name,
        num_runs=len(performance_results),
    )

    if not performance_results:
        return agg

    threshold_labels = list(performance_results[0].compliance_rates.keys())
    for label in threshold_labels:
        crs = [r.compliance_rates[label] for r in performance_results]
        mean, ci = mean_and_ci(crs)
        agg.compliance_rates_mean[label] = mean
        agg.compliance_rates_ci[label] = ci

        cirs = [r.compliance_improvement_ratios[label] for r in performance_results]
        mean, ci = mean_and_ci(cirs)
        agg.cir_mean[label] = mean
        agg.cir_ci[label] = ci

    avg_cts = [r.avg_cycle_time for r in performance_results]
    agg.avg_cycle_time_mean, agg.avg_cycle_time_ci = mean_and_ci(avg_cts)

    med_cts = [r.median_cycle_time for r in performance_results]
    agg.median_cycle_time_mean, agg.median_cycle_time_ci = mean_and_ci(med_cts)

    std_cts = [r.std_cycle_time for r in performance_results]
    agg.std_cycle_time_mean = float(np.mean(std_cts))

    util_cvs = [r.resource_utilization_cv for r in performance_results if r.resource_utilization_cv is not None]
    if util_cvs:
        agg.resource_utilization_cv_mean = float(np.mean(util_cvs))

    sim_keys = ["ngd", "aed", "ced", "red", "cwd", "car", "ctd"]
    for key in sim_keys:
        vals = [getattr(r, key) for r in similarity_results if getattr(r, key) is not None]
        if vals:
            mean, ci = mean_and_ci(vals)
            agg.similarity_mean[key] = mean
            agg.similarity_ci[key] = ci

    return agg
