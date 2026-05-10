from typing import Optional

import pandas as pd

from ..entities.similarity_result import SimilarityResult


def compute_similarity_metrics(
    original_log_path: str,
    simulated_log_path: str,
    original_col_names: Optional[dict] = None,
) -> SimilarityResult:
    """
    Compute all 7 similarity metrics using the log-distance-measures package.

    Requires: pip install log-distance-measures

    Falls back gracefully if the package is not installed.
    """
    result = SimilarityResult()

    try:
        from log_distance_measures.config import EventLogIDs
        from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
        from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
        from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
        from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
        from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
        from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
        from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
    except ImportError:
        print("  WARNING: log-distance-measures not installed. Skipping similarity metrics.")
        print("  Install with: pip install log-distance-measures")
        return result

    original = pd.read_csv(original_log_path)
    simulated = pd.read_csv(simulated_log_path)

    _orig_cols = original_col_names or {}
    original_ids = EventLogIDs(
        case=_orig_cols.get("case", "case_id"),
        activity=_orig_cols.get("activity", "activity"),
        resource=_orig_cols.get("resource", "resource"),
        start_time=_orig_cols.get("start", "start_time"),
        end_time=_orig_cols.get("end", "end_time"),
    )
    simulated_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        resource="resource",
        start_time="start_time",
        end_time="end_time",
    )

    for col in [original_ids.start_time, original_ids.end_time]:
        original[col] = pd.to_datetime(original[col], format='ISO8601', utc=True)
    for col in [simulated_ids.start_time, simulated_ids.end_time]:
        simulated[col] = pd.to_datetime(simulated[col], format='ISO8601', utc=True)

    try:
        result.ngd = n_gram_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  NGD computation failed: {e}")

    try:
        result.aed = absolute_event_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  AED computation failed: {e}")

    try:
        result.ced = circadian_event_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  CED computation failed: {e}")

    try:
        result.red = relative_event_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  RED computation failed: {e}")

    try:
        result.cwd = circadian_workforce_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  CWD computation failed: {e}")

    try:
        result.car = case_arrival_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  CAR computation failed: {e}")

    try:
        result.ctd = cycle_time_distribution_distance(
            original, original_ids, simulated, simulated_ids, bin_size=pd.Timedelta(hours=1)
        )
    except Exception as e:
        print(f"  CTD computation failed: {e}")

    return result
