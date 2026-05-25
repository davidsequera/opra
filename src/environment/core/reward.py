from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

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
    activity_probs_agent: Optional[list] = None    # π(a|s) — agent's full activity distribution
    activity_probs_routing: Optional[list] = None  # P(a|a_last) — empirical routing distribution
    mean_routing_prob: float = 1.0  # geometric mean of chosen-activity probs across all case decisions


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

class RegularizedSLARewardFunction(SLARewardFunction):
    """
    SLA reward regularized by the KL divergence between the agent's policy and
    the empirical routing distribution.

    R(s, a) = R_base(s, a) − β · D_KL(π ∥ P)

    where

        D_KL(π ∥ P) = Σ_{a∈A} π(a|s) · log( π(a|s) / P(a|a_last) )

    π(a|s) is the agent's masked activity distribution and P(a|a_last) is the
    empirical (Markov) routing distribution from the as-is log.  The KL term
    penalises divergence from the original process without touching the sign
    of the base SLA signal.

    beta=0  →  identical to SLARewardFunction (no regularization).
    beta>0  →  increasingly penalises distributional drift from the as-is log.

    The two distribution vectors are supplied via CaseRewardContext fields
    `activity_probs_agent` and `activity_probs_routing`.  If either is absent
    the method falls back to the un-regularized base reward.

    Parameters
    ----------
    H : float
        Passed through to SLARewardFunction.
    beta : float
        Regularization strength (≥ 0).  0 disables regularization.
    """

    def __init__(self, H: float = 1.0, beta: float = 0.0):
        super().__init__(H=H)
        self.beta = beta

    def compute(self, ctx: CaseRewardContext) -> float:
        import math
        base = super().compute(ctx)
        if (
            self.beta == 0.0
            or ctx.activity_probs_agent is None
            or ctx.activity_probs_routing is None
        ):
            return base

        # D_KL(π ∥ P) = Σ_a π(a) · log(π(a) / P(a))
        eps = 1e-8
        kl = 0.0
        for pi_a, p_a in zip(ctx.activity_probs_agent, ctx.activity_probs_routing):
            if pi_a > eps:
                kl += pi_a * math.log((pi_a + eps) / (p_a + eps))

        return base - self.beta * kl


class CombinedRegularizedSLARewardFunction(SLARewardFunction):
    """
    SLA reward combining distributional regularization at terminal time with
    per-step KL penalization.

    Intermediate (case not yet completed):
        R = R_base(s, a) − β · D_KL(π ∥ P)

    Terminal (case completed):
        R = mean_routing_prob · R_base(s, a) − β · D_KL(π ∥ P)

    where `mean_routing_prob` is the geometric mean of the as-is routing
    probability of the chosen activity across all decisions in the case.
    It scales the terminal signal down when the agent repeatedly chose
    low-probability activities, combining outcome credit with process fidelity.

    D_KL(π ∥ P) = Σ_{a∈A} π(a|s) · log( π(a|s) / P(a|a_last) )

    beta=0  →  identical to SLARewardFunction.
    """

    def __init__(self, H: float = 1.0, beta: float = 0.0):
        super().__init__(H=H)
        self.beta = beta

    def _kl(self, ctx: CaseRewardContext) -> float:
        import math
        eps = 1e-8
        kl = 0.0
        for pi_a, p_a in zip(ctx.activity_probs_agent, ctx.activity_probs_routing):
            if pi_a > eps:
                kl += pi_a * math.log((pi_a + eps) / (p_a + eps))
        return kl

    def compute(self, ctx: CaseRewardContext) -> float:
        base = super().compute(ctx)
        if (
            self.beta == 0.0
            or ctx.activity_probs_agent is None
            or ctx.activity_probs_routing is None
        ):
            return base

        kl = self._kl(ctx)
        if ctx.is_completed:
            return ctx.mean_routing_prob * base - self.beta * kl
        return base - self.beta * kl


class KLOnlyRewardFunction(RewardFunction):
    """
    Warm-up reward that teaches the agent to imitate the empirical routing
    distribution before SLA optimization begins.

    R = −β · D_KL(π ∥ P)

    The base SLA signal is ignored entirely.  Use this for an initial set of
    episodes so the agent learns to stay close to the as-is process, then
    switch to a regularized SLA reward function for the main training phase.

    Returns 0.0 if either distribution is absent from the context.

    Parameters
    ----------
    beta : float
        Scaling factor for the KL penalty. Defaults to 1.0.
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def compute(self, ctx: CaseRewardContext) -> float:
        import math
        if ctx.activity_probs_agent is None or ctx.activity_probs_routing is None:
            return 0.0
        eps = 1e-8
        kl = 0.0
        for pi_a, p_a in zip(ctx.activity_probs_agent, ctx.activity_probs_routing):
            if pi_a > eps:
                kl += pi_a * math.log((pi_a + eps) / (p_a + eps))
        return -self.beta * kl


class BinaryRewardFunction(RewardFunction):
    """
    Simple binary reward (the previous default).

    +1 if the case meets the SLA threshold, 0 otherwise.
    """

    def compute(self, ctx: CaseRewardContext) -> float:
        return 1.0 if ctx.cycle_time <= ctx.sla_threshold else 0.0
