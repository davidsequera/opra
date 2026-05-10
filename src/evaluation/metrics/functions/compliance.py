import numpy as np


def compute_compliance_rate(cycle_times: np.ndarray, threshold: float) -> float:
    """CR(L, T) = |{σ ∈ L : ct(σ) < T}| / |L|"""
    if len(cycle_times) == 0:
        return 0.0
    return float(np.mean(cycle_times < threshold))


def compute_cir(sim_cr: float, ref_cr: float) -> float:
    """CIR = (CR_sim - CR_ref) / CR_ref"""
    if ref_cr == 0:
        return float('inf') if sim_cr > 0 else 0.0
    return (sim_cr - ref_cr) / ref_cr
