from abc import ABC, abstractmethod

import numpy as np


class ActivitySelector(ABC):
    """Picks the next activity index given the current case, state, and feasibility mask."""

    @abstractmethod
    def select(self, env, case, state, activity_mask, deterministic: bool = False) -> int:
        ...


def _masked_routing_probs(env, case, activity_mask) -> np.ndarray:
    probs_dict = env.simulator.setup.routing_policy.get_activity_probabilities(case)
    all_acts = env.simulator.all_activities
    probs = np.array([float(probs_dict.get(a, 0.0)) for a in all_acts])
    return probs * np.asarray(activity_mask, dtype=float)


def _fallback_valid_index(activity_mask, rng) -> int:
    valid = np.where(np.asarray(activity_mask) == 1)[0]
    if len(valid) == 0:
        return 0
    return int(rng.choice(valid))


class RandomActivitySelector(ActivitySelector):
    """Uniform sample over feasible activities (RA)."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def select(self, env, case, state, activity_mask, deterministic: bool = False) -> int:
        return _fallback_valid_index(activity_mask, self.rng)


class GreedyProbActivitySelector(ActivitySelector):
    """Argmax of routing-policy probabilities, restricted to the feasibility mask (GP)."""

    def select(self, env, case, state, activity_mask, deterministic: bool = False) -> int:
        masked = _masked_routing_probs(env, case, activity_mask)
        if masked.sum() == 0:
            return _fallback_valid_index(activity_mask, np.random.default_rng())
        return int(np.argmax(masked))


class EmpiricalDMActivitySelector(ActivitySelector):
    """Sample from routing-policy probabilities, restricted to the feasibility mask (DM)."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def select(self, env, case, state, activity_mask, deterministic: bool = False) -> int:
        masked = _masked_routing_probs(env, case, activity_mask)
        total = masked.sum()
        if total == 0:
            return _fallback_valid_index(activity_mask, self.rng)
        probs = masked / total
        return int(self.rng.choice(len(probs), p=probs))


class DRLActivitySelector(ActivitySelector):
    """Use the agent's activity head."""

    def __init__(self, agent):
        self.agent = agent

    def select(self, env, case, state, activity_mask, deterministic: bool = False) -> int:
        import torch
        from torch.distributions import Categorical

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            mask_t = torch.FloatTensor(np.asarray(activity_mask, dtype=np.float32)).unsqueeze(0).to(self.agent.device)
            logits = self.agent.policy_old.get_activity_logits(state_t)
            logits = logits.masked_fill(mask_t == 0, -1e9)
            if deterministic:
                idx = torch.argmax(logits, dim=-1)
            else:
                idx = Categorical(logits=logits).sample()
            return int(idx.item())
