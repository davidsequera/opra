from typing import Optional

import numpy as np
import pandas as pd


def compute_resource_utilization_cv(
    log_df: pd.DataFrame,
    resource_col: str,
    start_col: str,
    end_col: str,
) -> Optional[float]:
    """
    CV of resource utilization = SD(utilizations) / mean(utilizations).
    Utilization = total busy time / total available time for each resource.
    Available time is approximated as (max_end - min_start) of the entire log.
    """
    if resource_col not in log_df.columns:
        return None

    log_df = log_df.copy()
    log_df[start_col] = pd.to_datetime(log_df[start_col], format='ISO8601')
    log_df[end_col] = pd.to_datetime(log_df[end_col], format='ISO8601')

    horizon_start = log_df[start_col].min()
    horizon_end = log_df[end_col].max()
    total_horizon = (horizon_end - horizon_start).total_seconds()

    if total_horizon <= 0:
        return None

    utilizations = []
    for _, group in log_df.groupby(resource_col):
        busy_time = (group[end_col] - group[start_col]).dt.total_seconds().sum()
        utilizations.append(busy_time / total_horizon)

    utilizations = np.array(utilizations)
    mean_u = np.mean(utilizations)
    if mean_u <= 0:
        return None
    return float(np.std(utilizations) / mean_u)
