import numpy as np
import pandas as pd


def compute_cycle_times(log_df: pd.DataFrame, case_col: str, start_col: str, end_col: str) -> np.ndarray:
    """Compute end-to-end cycle time per case (in seconds)."""
    cycle_times = []
    for _, group in log_df.groupby(case_col):
        st = pd.to_datetime(group[start_col], format='ISO8601').min()
        et = pd.to_datetime(group[end_col], format='ISO8601').max()
        cycle_times.append((et - st).total_seconds())
    return np.array(cycle_times)
