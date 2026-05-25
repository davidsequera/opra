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
            R(σ) = H/1000 * (1 - ct / T)

            The reward decreases linearly from +H (at ct → 0)
            to 0 when the cycle time approaches the SLA threshold.

        If ct ≥ T:
            R(σ) = -H/100 * (ct / T)

            The penalty grows linearly as the cycle time exceeds
            the SLA threshold.

    Terminal states (case completed):
        If ct < T:
            R(σ) += +H

            The case finished within the SLA.

        If ct ≥ T:
            R(σ) += -H

            The case violated the SLA.

    Parameters
    ----------
    H : float
        Scaling constant controlling the magnitude of rewards
        and penalties. Defaults to 1.0.
    """

    def __init__(self, H: float = 1.0):
        self.H = H

    def compute(self, ctx: CaseRewardContext) -> float:
        ct = max(ctx.cycle_time, 1e-6)
        T = max(ctx.sla_threshold, 1e-6)
        H = self.H
        # Intermediate: directional signal at each step
        if ct < T:
            reward = (H/1000) * (1.0 - ct / T)  # Linear: 1→0 as you approach T
        else:
            reward = -(H/100) * (ct / T)        # Increasingly negative past T

        # Terminal: case completed
        if ctx.is_completed:
            if ct < T:
                reward = +H
            else:
                reward = -H

        return reward


class BinaryRewardFunction(RewardFunction):
    """
    Simple binary reward (the previous default).

    +1 if the case meets the SLA threshold, 0 otherwise.
    """

    def compute(self, ctx: CaseRewardContext) -> float:
        return 1.0 if ctx.cycle_time <= ctx.sla_threshold else 0.0
