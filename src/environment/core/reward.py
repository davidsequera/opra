from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.entities.Case import Case


@dataclass
class CaseRewardContext:
    """
    All the information a reward function might need to score a case.

    This context is built by the environment at each decision point and at
    case-completion time and passed to the reward function.  Adding new fields
    here (rather than changing the RewardFunction signature) keeps every
    implementation forward-compatible.
    """
    cycle_time: float       # elapsed time (final if completed, current if not)
    sla_threshold: float
    num_events: int
    start_time: float
    end_time: float
    is_completed: bool = True
    chosen_activity_prob: float = 1.0  # as-is routing probability of the chosen activity


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    A reward function receives a CaseRewardContext for each completed case
    and returns a scalar reward signal that the RL agent will optimize.
    """

    @abstractmethod
    def compute(self, ctx: CaseRewardContext) -> float:
        """
        Computes the reward for a single completed case.

        :param ctx: Context with all case-level metrics.
        :return: A scalar reward value.
        """
        pass


class SLARewardFunction(RewardFunction):
    """
    SLA-based reward encouraging completion before the deadline.

    The reward is based on the relation between the current cycle time (ct)
    and the SLA threshold (T).

    Intermediate states (case not yet completed):
        If ct < T:
            R(σ) = K/1000 * (1 - ct / T)

            The reward decreases linearly from +K (at ct → 0)
            to 0 when the cycle time approaches the SLA threshold.

        If ct ≥ T:
            R(σ) = -K/1000 * (ct / T)

            The penalty grows linearly as the cycle time exceeds
            the SLA threshold.

    Terminal states (case completed):
        If ct < T:
            R(σ) += +K

            The case finished within the SLA.

        If ct ≥ T:
            R(σ) += -K

            The case violated the SLA.

    Parameters
    ----------
    K : float
        Scaling constant controlling the magnitude of rewards
        and penalties. Defaults to 1.0.
    """

    def __init__(self, K: float = 1.0):
        self.K = K

    def compute(self, ctx: CaseRewardContext) -> float:
        ct = max(ctx.cycle_time, 1e-6)
        T = max(ctx.sla_threshold, 1e-6)
        K = self.K
        # Intermediate: directional signal at each step
        if ct < T:
            reward = (K/1000) * (1.0 - ct / T)  # Linear: 1→0 as you approach T
        else:
            reward = -(K/10) * (ct / T)        # Increasingly negative past T

        # Terminal: case completed
        if ctx.is_completed:
            if ct < T:
                reward = +K
            else:
                reward = -K

        return reward

class RegularizedSLARewardFunction(SLARewardFunction):
    """
    SLA reward with distributional regularization on positive signals.

    Positive rewards are scaled by (alpha * (Prob(A) - 1) + 1), where
    Prob(A) is the as-is routing probability of the chosen activity.
    Negative rewards are unchanged so SLA violations are not softened.

    alpha=0  →  identical to SLARewardFunction (no regularization).
    alpha=1, Prob(A)=1  →  multiplier=1 (action aligns with original dist).
    alpha=1, Prob(A)=0  →  multiplier=0 (action impossible in original dist).

    Parameters
    ----------
    K : float
        Passed through to SLARewardFunction.
    alpha : float
        Regularization strength. 0 disables regularization; 1 fully scales
        rewards by the as-is routing probability.
    """

    def __init__(self, K: float = 1.0, alpha: float = 0.0):
        super().__init__(K=K)
        self.alpha = alpha

    def compute(self, ctx: CaseRewardContext) -> float:
        base = super().compute(ctx)
        if base > 0:
            multiplier = self.alpha * (ctx.chosen_activity_prob - 1.0) + 1.0
            print(f"{base:.4f}", f"{multiplier:.4f}", f"{(base*multiplier):.4f}")
            return base * multiplier
        return base


class BinaryRewardFunction(RewardFunction):
    """
    Simple binary reward (the previous default).

    +1 if the case meets the SLA threshold, 0 otherwise.
    """

    def compute(self, ctx: CaseRewardContext) -> float:
        return 1.0 if ctx.cycle_time <= ctx.sla_threshold else 0.0
