import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .entities.aggregated_results import AggregatedResults
from .entities.performance_result import PerformanceResult
from .entities.similarity_result import SimilarityResult
from .functions.aggregation import aggregate_results
from .functions.compliance import compute_compliance_rate
from .functions.cycle_time import compute_cycle_times
from .functions.performance_metrics import compute_performance_metrics
from .functions.similarity_metrics import compute_similarity_metrics


class PolicyEvaluator:
    """
    Evaluates a trained policy by running K simulations and computing
    all performance + similarity metrics from the thesis.
    """

    def __init__(
        self,
        original_log_path: str,
        original_log_names: Dict,  # {"case": ..., "activity": ..., ...}
        sla_percentiles: List[int] = [95, 90, 75, 50],
    ):
        self.original_log_path = original_log_path
        self.original_log_names = original_log_names

        self.original_df = pd.read_csv(original_log_path)
        self.ref_cycle_times = compute_cycle_times(
            self.original_df,
            original_log_names["case"],
            original_log_names["start"],
            original_log_names["end"],
        )

        self.sla_thresholds: Dict[str, float] = {}
        self.ref_compliance_rates: Dict[str, float] = {}
        for p in sla_percentiles:
            label = f"T{p}"
            threshold = float(np.percentile(self.ref_cycle_times, p))
            self.sla_thresholds[label] = threshold
            self.ref_compliance_rates[label] = compute_compliance_rate(self.ref_cycle_times, threshold)

        print(f"Original log: {len(self.ref_cycle_times)} cases")
        print(f"SLA thresholds: {self.sla_thresholds}")
        print(f"Reference CRs: {self.ref_compliance_rates}")

    def evaluate_single_log(
        self,
        sim_log_path: str,
        sim_case_col: str = "case_id",
        sim_start_col: str = "start_time",
        sim_end_col: str = "end_time",
        sim_resource_col: str = "resource",
    ) -> Tuple[PerformanceResult, SimilarityResult]:
        """Evaluate one simulated log file."""
        sim_df = pd.read_csv(sim_log_path)

        perf = compute_performance_metrics(
            sim_df,
            self.ref_cycle_times,
            self.sla_thresholds,
            self.ref_compliance_rates,
            case_col=sim_case_col,
            start_col=sim_start_col,
            end_col=sim_end_col,
            resource_col=sim_resource_col,
        )
        perf.log_path = sim_log_path

        sim = compute_similarity_metrics(self.original_log_path, sim_log_path, self.original_log_names)

        return perf, sim

    def evaluate_policy(
        self,
        simulated_log_paths: List[str],
        policy_name: str,
        log_name: str,
    ) -> AggregatedResults:
        """Evaluate K simulated logs for one policy and aggregate."""
        perf_results = []
        sim_results = []

        for path in simulated_log_paths:
            print(f"  Evaluating: {path}")
            perf, sim = self.evaluate_single_log(path)
            perf_results.append(perf)
            sim_results.append(sim)

        return aggregate_results(perf_results, sim_results, policy_name, log_name)

    def save_results(self, results: AggregatedResults, output_dir: str):
        """Save aggregated results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "results.json")
        with open(path, "w") as f:
            json.dump(asdict(results), f, indent=2, default=str)
        print(f"  Results saved: {path}")

    def print_results(self, results: AggregatedResults):
        """Pretty-print aggregated results."""
        print(f"\n{'='*60}")
        print(f"Policy: {results.policy_name} | Log: {results.log_name} | Runs: {results.num_runs}")
        print(f"{'='*60}")

        print("\nSLA Compliance Rates (mean ± 95% CI):")
        for label in results.compliance_rates_mean:
            cr = results.compliance_rates_mean[label]
            ci = results.compliance_rates_ci.get(label, 0)
            cir = results.cir_mean.get(label, 0)
            cir_ci = results.cir_ci.get(label, 0)
            print(f"  {label}: CR = {cr:.2%} ± {ci:.2%}  |  CIR = {cir:+.2%} ± {cir_ci:.2%}")

        print(f"\nCycle Time:")
        print(f"  Mean:   {results.avg_cycle_time_mean:.1f} ± {results.avg_cycle_time_ci:.1f}")
        print(f"  Median: {results.median_cycle_time_mean:.1f} ± {results.median_cycle_time_ci:.1f}")
        print(f"  Std:    {results.std_cycle_time_mean:.1f}")

        if results.resource_utilization_cv_mean is not None:
            print(f"  Resource Util CV: {results.resource_utilization_cv_mean:.4f}")

        if results.similarity_mean:
            print(f"\nSimilarity Metrics (mean ± 95% CI):")
            for key, val in results.similarity_mean.items():
                ci = results.similarity_ci.get(key, 0)
                print(f"  {key.upper()}: {val:.4f} ± {ci:.4f}")
