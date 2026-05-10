from dataclasses import dataclass
from typing import Optional


@dataclass
class SimilarityResult:
    """Similarity metrics comparing a simulated log to the original (Section 8.2.2)."""
    ngd: Optional[float] = None   # N-gram distance (control-flow)
    aed: Optional[float] = None   # Absolute event distribution (temporal)
    ced: Optional[float] = None   # Circadian event distribution (temporal)
    red: Optional[float] = None   # Relative event distribution (temporal)
    cwd: Optional[float] = None   # Circadian workforce distribution (resource)
    car: Optional[float] = None   # Case arrival rate (congestion)
    ctd: Optional[float] = None   # Cycle time distribution (congestion)
