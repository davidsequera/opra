from abc import ABC, abstractmethod

import numpy as np


class ResourceSelector(ABC):
    """Picks a resource index given the chosen activity and feasibility mask."""

    @abstractmethod
    def select(self, env, case, state, activity_idx, resource_mask, deterministic: bool = False) -> int:
        ...


class RandomResourceSelector(ResourceSelector):
    """Uniform sample over feasible resources (RR)."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def select(self, env, case, state, activity_idx, resource_mask, deterministic: bool = False) -> int:
        valid = np.where(np.asarray(resource_mask) == 1)[0]
        if len(valid) == 0:
            return 0
        return int(self.rng.choice(valid))


class GreedyProcessingTimeResourceSelector(ResourceSelector):
    """
    Picks the feasible resource with the lowest estimated processing time for
    the chosen activity. Holds a reference to the processing-time policy at
    construction so the heuristic is explicit about what it relies on.

    The estimate is computed externally via `n_samples` draws of the policy's
    public `get_activity_duration(...)` interface — the policy itself stays
    unaware of this consumer (hexagonal boundary). Whatever fallback the
    policy implements for unseen (activity, resource) pairs is honored
    transparently.
    """

    def __init__(self, processing_time_policy, n_samples: int = 8):
        self.policy = processing_time_policy
        self.n_samples = max(1, n_samples)

    def _estimate_duration(self, activity_name, resource) -> float:
        total = 0.0
        for _ in range(self.n_samples):
            total += float(self.policy.get_activity_duration(activity_name, resource))
        return total / self.n_samples

    def select(self, env, case, state, activity_idx, resource_mask, deterministic: bool = False) -> int:
        activity_name = env.simulator.all_activities[activity_idx]
        all_resources = env.simulator.all_resources
        mask = np.asarray(resource_mask)

        best_idx = -1
        best_duration = float("inf")
        for i, allowed in enumerate(mask):
            if not allowed:
                continue
            duration = self._estimate_duration(activity_name, all_resources[i])
            if duration < best_duration:
                best_duration = duration
                best_idx = i

        if best_idx == -1:
            return 0
        return best_idx


class DRLResourceSelector(ResourceSelector):
    """
    Use the agent's resource head conditioned on the chosen activity.
    Works with any agent whose policy_old exposes get_resource_logits(state, activity)
    — currently PPOAgent (full) and (eventually) PPOResourceOnlyAgent.
    """

    def __init__(self, agent):
        self.agent = agent

    def select(self, env, case, state, activity_idx, resource_mask, deterministic: bool = False) -> int:
        import torch
        from torch.distributions import Categorical

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            mask_t = torch.FloatTensor(np.asarray(resource_mask, dtype=np.float32)).unsqueeze(0).to(self.agent.device)
            act_t = torch.tensor([activity_idx], dtype=torch.long, device=self.agent.device)
            logits = self.agent.policy_old.get_resource_logits(state_t, act_t)
            logits = logits.masked_fill(mask_t == 0, -1e9)
            if deterministic:
                idx = torch.argmax(logits, dim=-1)
            else:
                idx = Categorical(logits=logits).sample()
            return int(idx.item())
