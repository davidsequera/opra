from dataclasses import dataclass
from typing import Optional


@dataclass
class UpdateMetrics:
    """Metrics collected after each PPO policy update."""

    update: int
    episode: int  # which episode triggered this update
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: Optional[float] = None
    clip_fraction: Optional[float] = None
