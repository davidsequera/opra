"""
OPRA Training Script.

Runs multiple episodes of the RL simulation, tracks metrics per episode,
saves model checkpoints, and exports training curves.

Usage:
    python src/train.py
    python src/train.py --episodes 200 --max_cases 50 --percentile 90
"""

import argparse
import os
import time
import random

import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext

from initializer.implementations.DDPSInitializer import DDPSInitializer
from environment.simulator.core.setup import SimulationSetup
from environment.core.env import BusinessProcessEnvironment
from environment.core.reward import RewardFunction,RegularizedSLARewardFunction, SLARewardFunction, CombinedRegularizedSLARewardFunction, KLOnlyRewardFunction
from environment.core.mask import NucleusMaskFunction
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.engine import SimulatorEngine
from agent.JointAgent.agent import PPOAgent

from evaluation.training.functions import (
    compute_episode_metrics,
)

from evaluation.training.training_metrics_tracker import (
    TrainingMetricsTracker,
    UpdateMetrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OPRA RL Training")
    parser.add_argument("--log_path", type=str, default="data/logs/LoanApp/LoanApp.csv")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max_cases", type=int, default=20, help="Cases per episode")
    parser.add_argument("--percentile", type=int, default=95, help="SLA percentile threshold")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N episodes")
    parser.add_argument("--update_every", type=int, default=1, help="PPO update every N episodes")
    parser.add_argument("--run_name", type=str, default=None, help="Name for this run")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering for activity mask")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k filtering for activity mask")
    parser.add_argument("--p_min_end", type=float, default=0.1, help="Minimum end probability for activity mask")
    parser.add_argument("--beta", type=float, default=0.0, help="Distributional regularization strength (0=off, 1=full)")
    parser.add_argument("--warmup_episodes", type=int, default=0, help="Episodes using KLOnlyRewardFunction before switching to main reward (0=off)")
    return parser.parse_args()


def compute_cycle_times_from_log(event_log: list, time_unit: str = "seconds") -> list:
    """
    Compute cycle times from the simulator's event_log (list of dicts).
    Each dict has keys: case_id, activity, resource, start_time, end_time (numeric SimPy times).
    """
    cases = {}
    for event in event_log:
        cid = event["case_id"]
        start = event["start_time"]
        end = event["end_time"]
        if cid not in cases:
            cases[cid] = {"min_start": start, "max_end": end}
        else:
            cases[cid]["min_start"] = min(cases[cid]["min_start"], start)
            cases[cid]["max_end"] = max(cases[cid]["max_end"], end)

    cycle_times = []
    for cid, times in cases.items():
        ct = times["max_end"] - times["min_start"]
        cycle_times.append(ct)
    return cycle_times


def save_checkpoint(agent: PPOAgent, path: str, episode: int, metrics_summary: dict):
    """Save model weights + training metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "episode": episode,
        "policy_state_dict": agent.policy.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "metrics_summary": metrics_summary,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(agent: PPOAgent, path: str) -> int:
    """Load model weights. Returns the episode number."""
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  Checkpoint loaded from: {path} (episode {checkpoint['episode']})")
    return checkpoint["episode"]


def run_single_episode(
    env: BusinessProcessEnvironment,
    simulator: SimulatorEngine,
    agent: PPOAgent,
    deterministic: bool = False,
    eval_mode: bool = False,
) -> tuple:
    """
    Run one full simulation episode.
    Returns (total_reward, num_steps, cycle_times).
    """
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    num_steps = 0
    context = torch.no_grad() if eval_mode else nullcontext()
    with context:
        while not (terminated or truncated):
            case = simulator.get_case_needing_decision()
            if case is None:
                break

            activity_mask = env.get_activity_mask(case)

            def res_mask_cb(act_idx):
                act_name = simulator.all_activities[act_idx]
                return env.get_resource_mask(act_name, case)

            act_idx, res_idx, act_probs = agent.select_action(
                state=obs,
                activity_mask=activity_mask,
                resource_mask_callback=res_mask_cb,
                deterministic=deterministic,
            )

            action = np.array([act_idx, res_idx])

            activity_type = simulator.all_activities[act_idx]
            env.set_agent_activity_probs(act_probs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store transition in agent buffer (only during training)
            if not eval_mode:
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(terminated or truncated)

            obs = next_obs
            total_reward += reward
            num_steps += 1

    cycle_times = compute_cycle_times_from_log(simulator.event_log)
    return total_reward, num_steps, cycle_times


def train_full_agent(
    *,
    log_path: str,
    episodes: int,
    max_cases: int,
    percentile: int = 75,
    top_k: int = 3,
    top_p: float = 0.9,
    p_min_end: float = 0.1,
    beta: float = 0.0,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 42,
    save_every: int = 10,
    update_every: int = 1,
    run_name: str | None = None,
    resume: str | None = None,
    warmup_episodes: int = 0,
    warmup_reward_function: "RewardFunction | None" = None,
) -> str:
    """Train the full DRL-DRL PPO agent. Returns the run directory."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if run_name is None:
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
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

    cycle_times_orig = []
    for _, group in log.groupby(log_names.case_id):
        st = pd.to_datetime(group[log_names.start_timestamp], format="mixed").min()
        et = pd.to_datetime(group[log_names.end_timestamp], format="mixed").max()
        cycle_times_orig.append((et - st).total_seconds())
    sla_threshold = float(np.percentile(cycle_times_orig, percentile))
    baseline_cr = float(np.mean(np.array(cycle_times_orig) < sla_threshold))
    print(f"SLA threshold (p{percentile}): {sla_threshold:.2f}s")
    print(f"Baseline CR (original log): {baseline_cr:.2%}")
    print(
        f"Parameters: episodes={episodes}, max_cases={max_cases}, lr={lr}, gamma={gamma}, "
        f"top_p={top_p}, top_k={top_k}, p_min_end={p_min_end}, beta={beta}"
    )

    reward_function = SLARewardFunction() if beta == 0.0 else CombinedRegularizedSLARewardFunction(beta=beta)
    env = BusinessProcessEnvironment(
        simulator,
        sla_threshold=sla_threshold,
        max_cases=max_cases,
        activity_mask_function=NucleusMaskFunction(k=top_k, p=top_p, p_min_end=p_min_end),
        reward_function=reward_function,
    )

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        num_activities=simulator.num_activities,
        num_resources=simulator.num_resources,
        lr=lr,
        gamma=gamma,
    )

    start_episode = 1
    if resume is not None:
        start_episode = load_checkpoint(agent, resume) + 1
        print(f"  Resuming training from episode {start_episode}")

    hyperparams = {
        "log_path": log_path, "agent_kind": "full",
        "episodes": episodes, "max_cases": max_cases,
        "sla_percentile": percentile, "sla_threshold": sla_threshold, "baseline_cr": baseline_cr,
        "lr": lr, "gamma": gamma, "seed": seed,
        "top_p": top_p, "top_k": top_k, "p_min_end": p_min_end, "beta": beta,
    }
    tracker = TrainingMetricsTracker(log_dir=run_dir, hyperparams=hyperparams)

    print(f"\nStarting training: {episodes} episodes, {max_cases} cases each")
    print(f"Run directory: {run_dir}\n")

    best_cr = -1.0
    update_count = 0

    for ep in range(start_episode, episodes + 1):
        if warmup_reward_function is not None and ep <= warmup_episodes:
            if ep == start_episode or ep == 1:
                print(f"  [Warmup] Using {warmup_reward_function.__class__.__name__} for episodes 1–{warmup_episodes}")
            env.reward_function = warmup_reward_function
        elif warmup_reward_function is not None and ep == warmup_episodes + 1:
            print(f"  [Warmup done] Switching to {reward_function.__class__.__name__} from episode {ep}")
            env.reward_function = reward_function

        ep_start = time.time()
        total_reward, num_steps, cycle_times = run_single_episode(
            env=env, simulator=simulator, agent=agent, deterministic=False,
        )
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
                    approx_kl=loss_info.get("approx_kl"),
                    clip_fraction=loss_info.get("clip_fraction"),
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
    print(f"\nTraining complete. Best CR: {best_cr:.2%} at episode {tracker._best_episode}")
    print(f"Metrics saved to: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    train_full_agent(
        log_path=args.log_path,
        episodes=args.episodes,
        max_cases=args.max_cases,
        percentile=args.percentile,
        top_k=args.top_k,
        top_p=args.top_p,
        p_min_end=args.p_min_end,
        beta=args.beta,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        save_every=args.save_every,
        update_every=args.update_every,
        run_name=args.run_name,
        resume=args.resume,
        warmup_episodes=args.warmup_episodes,
        warmup_reward_function=KLOnlyRewardFunction() if args.warmup_episodes > 0 else None,
    )


if __name__ == "__main__":
    main()
