"""
OPRA training script for the resource-only PPO agent (DM-DRL backbone).

The activity at each decision is sampled from the empirical routing policy
(the same DM used at evaluation), so the agent only learns to optimize
resource allocation under a fixed control-flow distribution.

Usage:
    python src/train_resource_only.py --log_path data/logs/AcademicCredentials/AcademicCredentials_train.csv --episodes 300 --max_cases 400
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from agent.ResourcesOnlyAgent.resource_only_agent import PPOResourceOnlyAgent
from environment.core.env import BusinessProcessEnvironment
from environment.core.mask import NucleusMaskFunction
from environment.core.reward import SLARewardFunction
from environment.simulator.core.engine import SimulatorEngine
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.setup import SimulationSetup
from evaluation.selectors.activity import EmpiricalDMActivitySelector
from initializer.implementations.DDPSInitializer import DDPSInitializer
from evaluation.training.functions import compute_episode_metrics
from evaluation.training.training_metrics_tracker import TrainingMetricsTracker, UpdateMetrics
from train import compute_cycle_times_from_log


def parse_args():
    parser = argparse.ArgumentParser(description="OPRA resource-only RL training")
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max_cases", type=int, default=None, help="Cases per episode (default: same as original log)")
    parser.add_argument("--percentile", type=int, default=75)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--p_min_end", type=float, default=0.1)
    return parser.parse_args()


def save_checkpoint(agent: PPOResourceOnlyAgent, path: str, episode: int, metrics_summary: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "policy_state_dict": agent.policy.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "metrics_summary": metrics_summary,
            "agent_kind": "resource_only",
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def run_episode(env, simulator, agent, dm_selector, deterministic: bool = False):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    num_steps = 0

    while not (terminated or truncated):
        case = simulator.get_case_needing_decision()
        if case is None:
            break

        act_mask = env.get_activity_mask(case)
        act_idx = dm_selector.select(env, case, obs, act_mask, deterministic)

        act_name = simulator.all_activities[act_idx]
        res_mask = env.get_resource_mask(act_name, case)

        res_idx = agent.select_action(state=obs, activity_idx=act_idx, resource_mask=res_mask, deterministic=deterministic)

        obs, reward, terminated, truncated, _ = env.step(np.array([act_idx, res_idx]))

        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(terminated or truncated)

        total_reward += float(reward)
        num_steps += 1

    cycle_times = compute_cycle_times_from_log(simulator.event_log)
    return total_reward, num_steps, cycle_times


def train_resource_only_agent(
    *,
    log_path: str,
    episodes: int,
    max_cases: int | None = None,
    percentile: int = 75,
    top_k: int = 3,
    top_p: float = 0.9,
    p_min_end: float = 0.1,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 42,
    save_every: int = 10,
    update_every: int = 1,
    run_name: str | None = None,
) -> str:
    """Train the resource-only PPO agent (DM-DRL backbone). Returns the run directory."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if run_name is None:
        run_name = f"resource_only_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join("data/training_models", run_name)

    log = pd.read_csv(log_path)
    log_names = LogColumnNames(
        case_id="case_id", activity="activity", resource="resource",
        start_timestamp="start_time", end_timestamp="end_time",
    )

    initializer = DDPSInitializer()
    start_timestamp = log[log_names.start_timestamp].min()
    setup: SimulationSetup = initializer.build(log, log_names, start_timestamp, "seconds")
    simulator = SimulatorEngine(setup)

    num_original_cases = log[log_names.case_id].nunique()
    max_cases = max_cases or num_original_cases

    cycle_times_orig = []
    for _, group in log.groupby(log_names.case_id):
        st = pd.to_datetime(group[log_names.start_timestamp], format="mixed").min()
        et = pd.to_datetime(group[log_names.end_timestamp], format="mixed").max()
        cycle_times_orig.append((et - st).total_seconds())
    sla_threshold = float(np.percentile(cycle_times_orig, percentile))
    baseline_cr = float(np.mean(np.array(cycle_times_orig) < sla_threshold))

    print(f"SLA threshold (p{percentile}): {sla_threshold:.2f}s")
    print(f"Baseline CR: {baseline_cr:.2%}")
    print(
        f"Parameters: episodes={episodes}, max_cases={max_cases}, lr={lr}, "
        f"gamma={gamma}, top_p={top_p}, top_k={top_k}, p_min_end={p_min_end}"
    )

    env = BusinessProcessEnvironment(
        simulator,
        sla_threshold=sla_threshold,
        max_cases=max_cases,
        activity_mask_function=NucleusMaskFunction(k=top_k, p=top_p, p_min_end=p_min_end),
        reward_function=SLARewardFunction(),  # no regularization
    )

    agent = PPOResourceOnlyAgent(
        state_dim=env.observation_space.shape[0],
        num_activities=simulator.num_activities,
        num_resources=simulator.num_resources,
        lr=lr,
        gamma=gamma,
    )

    dm_selector = EmpiricalDMActivitySelector(rng=np.random.default_rng(seed))

    hyperparams = {
        "log_path": log_path, "agent_kind": "resource_only",
        "episodes": episodes, "max_cases": max_cases,
        "sla_percentile": percentile, "sla_threshold": sla_threshold, "baseline_cr": baseline_cr,
        "lr": lr, "gamma": gamma, "seed": seed,
        "top_p": top_p, "top_k": top_k, "p_min_end": p_min_end,
    }
    tracker = TrainingMetricsTracker(log_dir=run_dir, hyperparams=hyperparams)

    print(f"\nStarting training: {episodes} episodes, {max_cases} cases each")
    print(f"Run directory: {run_dir}\n")

    best_cr = -1.0
    update_count = 0

    for ep in range(1, episodes + 1):
        ep_start = time.time()
        total_reward, num_steps, cycle_times = run_episode(env, simulator, agent, dm_selector)
        ep_duration = time.time() - ep_start

        ep_metrics = compute_episode_metrics(
            episode=ep, total_reward=total_reward, num_steps=num_steps,
            cycle_times=cycle_times, sla_threshold=sla_threshold,
            episode_duration_sec=ep_duration,
        )
        tracker.log_episode(ep_metrics)
        tracker.print_episode_summary(ep_metrics, baseline_cr=baseline_cr)

        if ep % update_every == 0:
            update_count += 1
            loss_info = agent.update()
            if loss_info is not None:
                upd = UpdateMetrics(
                    update=update_count, episode=ep,
                    policy_loss=loss_info.get("policy_loss", 0.0),
                    value_loss=loss_info.get("value_loss", 0.0),
                    entropy=loss_info.get("entropy", 0.0),
                    total_loss=loss_info.get("total_loss", 0.0),
                )
                tracker.log_update(upd)
                tracker.print_update_summary(upd)

        is_best = ep_metrics.sla_compliance_rate > best_cr
        if is_best:
            best_cr = ep_metrics.sla_compliance_rate

        if ep % save_every == 0 or is_best:
            summary = {
                "sla_compliance_rate": ep_metrics.sla_compliance_rate,
                "avg_cycle_time": ep_metrics.avg_cycle_time,
                "total_reward": ep_metrics.total_reward,
            }
            ckpt = os.path.join(run_dir, "checkpoints", f"checkpoint_ep{ep:04d}.pt")
            save_checkpoint(agent, ckpt, ep, summary)
            if is_best:
                save_checkpoint(agent, os.path.join(run_dir, "checkpoints", "best_model.pt"), ep, summary)

        if ep % save_every == 0:
            tracker.save()

    tracker.save()
    save_checkpoint(
        agent,
        os.path.join(run_dir, "checkpoints", "final_model.pt"),
        episodes,
        {"sla_compliance_rate": tracker.episode_history[-1].sla_compliance_rate},
    )
    print(f"\nTraining complete. Best CR: {best_cr:.2%}. Run dir: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    train_resource_only_agent(
        log_path=args.log_path,
        episodes=args.episodes,
        max_cases=args.max_cases,
        percentile=args.percentile,
        top_k=args.top_k,
        top_p=args.top_p,
        p_min_end=args.p_min_end,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        save_every=args.save_every,
        update_every=args.update_every,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
