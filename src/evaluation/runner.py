"""Generic episode runner that drives BusinessProcessEnvironment via pluggable selectors."""

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeResult:
    total_reward: float
    num_steps: int
    cycle_times: list[float]


def compute_cycle_times_from_log(event_log: list[dict]) -> list[float]:
    """Compute per-case cycle time from the simulator's numeric event log."""
    cases: dict = {}
    for ev in event_log:
        cid, s, e = ev["case_id"], ev["start_time"], ev["end_time"]
        if cid not in cases:
            cases[cid] = [s, e]
        else:
            cases[cid][0] = min(cases[cid][0], s)
            cases[cid][1] = max(cases[cid][1], e)
    return [end - start for start, end in cases.values()]


def run_episode(
    env,
    simulator,
    activity_selector,
    resource_selector,
    deterministic: bool = False,
) -> EpisodeResult:
    """
    Run one full simulation episode using the given (activity, resource) selectors.

    Mirrors the loop in train.run_single_episode but is decoupled from PPOAgent —
    every policy (random, greedy, DM, DRL, mixed) goes through this same path.
    """
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
        act_idx = activity_selector.select(env, case, obs, act_mask, deterministic)

        act_name = simulator.all_activities[act_idx]
        res_mask = env.get_resource_mask(act_name, case)
        res_idx = resource_selector.select(env, case, obs, act_idx, res_mask, deterministic)

        obs, reward, terminated, truncated, _ = env.step(np.array([act_idx, res_idx]))
        total_reward += float(reward)
        num_steps += 1

    cycle_times = compute_cycle_times_from_log(simulator.event_log)
    return EpisodeResult(total_reward=total_reward, num_steps=num_steps, cycle_times=cycle_times)
