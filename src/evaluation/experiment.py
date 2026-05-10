"""High-level helper that runs K simulations of a single (log, policy) combination."""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch

from environment.core.env import BusinessProcessEnvironment
from environment.core.mask import NucleusMaskFunction
from environment.simulator.adapters.event_log_to_csv import export_event_log_to_csv
from environment.simulator.core.engine import SimulatorEngine
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.setup import SimulationSetup
from initializer.implementations.DDPSInitializer import DDPSInitializer

from .metrics.entities import AggregatedResults, PerformanceResult, SimilarityResult
from .metrics.functions.aggregation import aggregate_results
from .metrics.policy_evaluator import PolicyEvaluator
from .runner import run_episode
from .selectors import build_policy


DEFAULT_LOG_COLUMNS = dict(
    case_id="case_id",
    activity="activity",
    resource="resource",
    start_timestamp="start_time",
    end_timestamp="end_time",
)

DEFAULT_MASKS = dict(top_k=3, top_p=0.95, p_min_end=0.1)

POLICIES_NEEDING_AGENT = {"DM-DRL", "DRL-DRL"}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _log_column_names() -> LogColumnNames:
    return LogColumnNames(**DEFAULT_LOG_COLUMNS)


def _compute_sla_threshold(log_df: pd.DataFrame, log_names: LogColumnNames, percentile: int) -> float:
    cycle_times = []
    for _, group in log_df.groupby(log_names.case_id):
        st = pd.to_datetime(group[log_names.start_timestamp], format="mixed").min()
        et = pd.to_datetime(group[log_names.end_timestamp], format="mixed").max()
        cycle_times.append((et - st).total_seconds())
    return float(np.percentile(cycle_times, percentile))


def _load_full_agent(checkpoint_path: str, env: BusinessProcessEnvironment, simulator: SimulatorEngine):
    from agent.JointAgent.agent import PPOAgent
    from train import load_checkpoint

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        num_activities=simulator.num_activities,
        num_resources=simulator.num_resources,
    )
    load_checkpoint(agent, checkpoint_path)
    return agent


def _load_resource_only_agent(checkpoint_path: str, env: BusinessProcessEnvironment, simulator: SimulatorEngine):
    from agent.ResourcesOnlyAgent.resource_only_agent import PPOResourceOnlyAgent

    agent = PPOResourceOnlyAgent(
        state_dim=env.observation_space.shape[0],
        num_activities=simulator.num_activities,
        num_resources=simulator.num_resources,
    )
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  Resource-only checkpoint loaded from: {checkpoint_path} (episode {checkpoint.get('episode', '?')})")
    return agent


def _load_agent_for_policy(policy_name: str, checkpoint_path: str, env: BusinessProcessEnvironment, simulator: SimulatorEngine):
    if policy_name == "DRL-DRL":
        return _load_full_agent(checkpoint_path, env, simulator)
    if policy_name == "DM-DRL":
        return _load_resource_only_agent(checkpoint_path, env, simulator)
    raise ValueError(f"Policy '{policy_name}' does not require an agent checkpoint")


def _make_env(setup: SimulationSetup, sla_threshold: float, max_cases: int, masks_cfg: dict):
    simulator = SimulatorEngine(setup)
    env = BusinessProcessEnvironment(
        simulator,
        sla_threshold=sla_threshold,
        max_cases=max_cases,
        activity_mask_function=NucleusMaskFunction(
            k=masks_cfg["top_k"],
            p=masks_cfg["top_p"],
            p_min_end=masks_cfg["p_min_end"],
        ),
    )
    return simulator, env


def _flatten_run_record(
    log_name: str,
    policy_name: str,
    run_id: int,
    perf: PerformanceResult,
    sim: SimilarityResult,
) -> dict:
    """Flatten a single run's metrics into a CSV-friendly dict."""
    rec: dict = {
        "log": log_name,
        "policy": policy_name,
        "run_id": run_id,
        "num_cases": perf.num_cases,
        "avg_ct": perf.avg_cycle_time,
        "median_ct": perf.median_cycle_time,
        "std_ct": perf.std_cycle_time,
        "min_ct": perf.min_cycle_time,
        "max_ct": perf.max_cycle_time,
        "p75_ct": perf.p75_cycle_time,
        "p90_ct": perf.p90_cycle_time,
        "p95_ct": perf.p95_cycle_time,
        "util_cv": perf.resource_utilization_cv,
    }
    for label, value in perf.compliance_rates.items():
        rec[f"cr_{label}"] = value
    for label, value in perf.compliance_improvement_ratios.items():
        rec[f"cir_{label}"] = value
    for key in ("ngd", "aed", "ced", "red", "cwd", "car", "ctd"):
        rec[key] = getattr(sim, key)
    return rec


def evaluate_policy_on_log(
    log_path: str,
    policy_name: str,
    *,
    K: int = 10,
    sla_percentiles: tuple[int, ...] = (95, 90, 75, 50),
    train_percentile: int = 95,
    checkpoint: Optional[str] = None,
    max_cases: Optional[int] = None,
    masks_cfg: Optional[dict] = None,
    output_dir: str = "data/evaluation_results",
    log_name: Optional[str] = None,
    seed: int = 0,
    evaluator: Optional[PolicyEvaluator] = None,
    verbose: bool = True,
) -> tuple[AggregatedResults, list[dict]]:
    """
    Run K independent simulations for one (log, policy) and return aggregated + per-run metrics.

    Side effects: writes per-run simulated CSV logs and a results.json under
        {output_dir}/{log_name}/{policy_name}/

    Parameters
    ----------
    checkpoint
        Required for "DM-DRL" and "DRL-DRL". Loaded into a PPOAgent. (When the
        resource-only agent class lands, DM-DRL will load that variant instead.)
    train_percentile
        SLA percentile used for the *training* threshold passed into the env's
        reward function. The evaluation thresholds (`sla_percentiles`) are
        independent of this.
    evaluator
        If provided, reuse it (so reference compliance data is shared across
        policies on the same log). If None, a fresh PolicyEvaluator is built
        from `log_path` and `sla_percentiles`.
    """
    masks_cfg = masks_cfg or dict(DEFAULT_MASKS)
    log_names = _log_column_names()
    log_df = pd.read_csv(log_path)

    if log_name is None:
        log_name = os.path.splitext(os.path.basename(log_path))[0]

    initializer = DDPSInitializer()
    start_timestamp = log_df[log_names.start_timestamp].min()
    setup: SimulationSetup = initializer.build(log_df, log_names, start_timestamp, "seconds")

    num_original_cases = log_df[log_names.case_id].nunique()
    max_cases = max_cases or num_original_cases

    sla_threshold = _compute_sla_threshold(log_df, log_names, train_percentile)

    if evaluator is None:
        evaluator = PolicyEvaluator(
            original_log_path=log_path,
            original_log_names={
                "case": log_names.case_id,
                "start": log_names.start_timestamp,
                "end": log_names.end_timestamp,
            },
            sla_percentiles=list(sla_percentiles),
        )

    # ---- agent (only for DRL policies) ----
    agent = None
    if policy_name in POLICIES_NEEDING_AGENT:
        if checkpoint is None:
            raise ValueError(f"Policy '{policy_name}' requires a checkpoint")
        # build a temporary env to discover state_dim / num_activities / num_resources
        sim_tmp, env_tmp = _make_env(setup, sla_threshold, max_cases, masks_cfg)
        agent = _load_agent_for_policy(policy_name, checkpoint, env_tmp, sim_tmp)
        del sim_tmp, env_tmp

    # ---- output dir ----
    policy_dir = os.path.join(output_dir, log_name, policy_name)
    sim_log_dir = os.path.join(policy_dir, "simulated_logs")
    os.makedirs(sim_log_dir, exist_ok=True)

    # ---- K-run loop ----
    rng = np.random.default_rng(seed)
    activity_selector, resource_selector = build_policy(
        policy_name,
        agent=agent,
        processing_time_policy=setup.processing_time_policy,
        rng=rng,
    )

    simulated_log_paths = []
    if verbose:
        print(f"\nRunning {K} simulations: log='{log_name}' policy='{policy_name}'")

    for k in range(1, K + 1):
        _seed_all(seed + k)

        simulator_k, env_k = _make_env(setup, sla_threshold, max_cases, masks_cfg)

        t0 = time.time()
        result = run_episode(env_k, simulator_k, activity_selector, resource_selector)
        duration = time.time() - t0

        simulator_k._convert_event_log_to_absolute_time()
        sim_path = os.path.join(sim_log_dir, f"sim_run_{k:02d}.csv")
        export_event_log_to_csv(simulator_k.event_log, sim_path)
        simulated_log_paths.append(sim_path)

        if verbose:
            ct = np.array(result.cycle_times)
            cr = float(np.mean(ct < sla_threshold)) if ct.size > 0 else 0.0
            avg_ct = float(np.mean(ct)) if ct.size > 0 else 0.0
            print(
                f"  Run {k:>2d}/{K}: cases={ct.size}, steps={result.num_steps}, "
                f"reward={result.total_reward:.2f}, "
                f"CR(p{train_percentile})={cr:.2%}, avg_ct={avg_ct:.1f}, "
                f"time={duration:.1f}s"
            )

    # ---- aggregate metrics + collect per-run records ----
    if verbose:
        print(f"Computing metrics for {policy_name}...")

    perf_results: list[PerformanceResult] = []
    sim_results: list[SimilarityResult] = []
    per_run_records: list[dict] = []
    for run_id, path in enumerate(simulated_log_paths, start=1):
        perf, sim = evaluator.evaluate_single_log(path)
        perf_results.append(perf)
        sim_results.append(sim)
        per_run_records.append(_flatten_run_record(log_name, policy_name, run_id, perf, sim))

    aggregated = aggregate_results(perf_results, sim_results, policy_name, log_name)

    # ---- persist ----
    evaluator.save_results(aggregated, policy_dir)
    if verbose:
        evaluator.print_results(aggregated)

    return aggregated, per_run_records


def load_per_run_records_from_disk(
    output_dir: str, log_name: str, policy_name: str, evaluator: PolicyEvaluator
) -> tuple[Optional[AggregatedResults], list[dict]]:
    """
    Resume helper: re-derive per-run records from previously-exported simulated logs.
    Returns (aggregated_results, per_run_records). Returns (None, []) if no logs found.
    """
    sim_log_dir = os.path.join(output_dir, log_name, policy_name, "simulated_logs")
    if not os.path.isdir(sim_log_dir):
        return None, []

    paths = sorted(os.path.join(sim_log_dir, f) for f in os.listdir(sim_log_dir) if f.endswith(".csv"))
    if not paths:
        return None, []

    perf_results: list[PerformanceResult] = []
    sim_results: list[SimilarityResult] = []
    per_run_records: list[dict] = []
    for run_id, path in enumerate(paths, start=1):
        perf, sim = evaluator.evaluate_single_log(path)
        perf_results.append(perf)
        sim_results.append(sim)
        per_run_records.append(_flatten_run_record(log_name, policy_name, run_id, perf, sim))

    aggregated = aggregate_results(perf_results, sim_results, policy_name, log_name)
    return aggregated, per_run_records
