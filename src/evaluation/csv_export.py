"""Export evaluation results in three CSV shapes: long, wide-per-run, wide-aggregated."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .metrics.entities.aggregated_results import AggregatedResults

_PERFORMANCE_KEYS = (
    "num_cases", "avg_ct", "median_ct", "std_ct", "min_ct", "max_ct",
    "p75_ct", "p90_ct", "p95_ct", "util_cv",
)
_SIMILARITY_KEYS = ("ngd", "aed", "ced", "red", "cwd", "car", "ctd")
_META_KEYS = ("log", "policy", "run_id")


def write_runs_wide(records: Iterable[dict], path: str) -> pd.DataFrame:
    """One row per (log, policy, run_id); columns are individual metrics."""
    df = pd.DataFrame(list(records))
    if df.empty:
        df.to_csv(path, index=False)
        return df
    # ordering: meta first, then perf, then cr_*, cir_*, then similarity
    cr_cols = sorted(c for c in df.columns if c.startswith("cr_"))
    cir_cols = sorted(c for c in df.columns if c.startswith("cir_"))
    ordered = [c for c in (*_META_KEYS, *_PERFORMANCE_KEYS, *cr_cols, *cir_cols, *_SIMILARITY_KEYS) if c in df.columns]
    df = df[ordered]
    df.to_csv(path, index=False)
    return df


def write_runs_long(records: Iterable[dict], path: str) -> pd.DataFrame:
    """One row per (log, policy, run_id, metric, value)."""
    wide = pd.DataFrame(list(records))
    if wide.empty:
        wide.to_csv(path, index=False)
        return wide
    metric_cols = [c for c in wide.columns if c not in _META_KEYS]
    long = wide.melt(
        id_vars=list(_META_KEYS),
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long.to_csv(path, index=False)
    return long


def _mean_and_ci95(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0
    sem = float(arr.std(ddof=1)) / np.sqrt(n)
    ci = float(stats.t.ppf(0.975, n - 1)) * sem
    return mean, ci


def write_aggregated(records: Iterable[dict], path: str) -> pd.DataFrame:
    """One row per (log, policy); columns = `<metric>_mean`, `<metric>_ci95`."""
    df = pd.DataFrame(list(records))
    if df.empty:
        df.to_csv(path, index=False)
        return df

    metric_cols = [c for c in df.columns if c not in _META_KEYS]
    rows = []
    for (log, policy), grp in df.groupby(["log", "policy"], sort=False):
        row = {"log": log, "policy": policy, "num_runs": len(grp)}
        for m in metric_cols:
            mean, ci = _mean_and_ci95(grp[m].tolist())
            row[f"{m}_mean"] = mean
            row[f"{m}_ci95"] = ci
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def write_all_csvs(records: list[dict], output_dir: str) -> dict:
    """Write the three CSVs into output_dir; return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "long": os.path.join(output_dir, "runs_long.csv"),
        "wide": os.path.join(output_dir, "runs_wide.csv"),
        "aggregated": os.path.join(output_dir, "aggregated.csv"),
    }
    write_runs_wide(records, paths["wide"])
    write_runs_long(records, paths["long"])
    write_aggregated(records, paths["aggregated"])
    return paths
