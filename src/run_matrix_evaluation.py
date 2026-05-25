"""
Matrix evaluation: every (log, policy) combination, K simulations each.

Run:
    python src/run_matrix_evaluation.py \
        --logs AC_CRE BPIC12 BPIC17 \
        --policies RA-RR DM-RR DM-GR DM-DRL DRL-DRL \
        --K 10 \
        --output-dir data/evaluation_results/matrix \
        [--resume]

Per-log paths and checkpoints live in LOG_REGISTRY below — edit there to add a log
or point at a different checkpoint, rather than passing 30 CLI flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Optional

from evaluation.csv_export import write_all_csvs
from evaluation.experiment import (
    DEFAULT_LOG_COLUMNS,
    evaluate_policy_on_log,
    load_per_run_records_from_disk,
)
from evaluation.metrics.policy_evaluator import PolicyEvaluator


def _ckpt(run_name: str) -> str:
    return f"data/training_models/{run_name}/checkpoints/best_model.pt"


# LOG_REGISTRY: dict[str, dict] = {
#     "AcademicCredentials": {
#         "path": "data/logs/AcademicCredentials/AcademicCredentials_train.csv",
#         "checkpoint_full": _ckpt("AcademicCredentials_DDPS_p75_300_400_tp90_tk2_pe30_b0_full"),
#         "checkpoint_resource_only": _ckpt("AcademicCredentials_DDPS_p75_300_400_tp90_tk2_pe30_b0_resource_only"),
#         "masks": dict(top_k=2, top_p=0.9, p_min_end=0.3),
#         "train_percentile": 75,
#     },
#     "BPIC_2012": {
#         "path": "data/logs/BPIC_2012/BPIC_2012_train.csv",
#         "checkpoint_full": _ckpt("BPIC_2012_DDPS_p75_300_3000_tp90_tk2_pe30_b0_full"),
#         "checkpoint_resource_only": _ckpt("BPIC_2012_DDPS_p75_300_3000_tp90_tk2_pe30_b0_resource_only"),
#         "masks": dict(top_k=2, top_p=0.9, p_min_end=0.3),
#         "train_percentile": 75,
#     },
#     "BPIC_2017": {
#         "path": "data/logs/BPIC_2017/BPIC_2017_train.csv",
#         "checkpoint_full": _ckpt("BPIC_2017_DDPS_p75_300_7400_tp90_tk3_pe10_b0_full"),
#         "checkpoint_resource_only": _ckpt("BPIC_2017_DDPS_p75_300_7400_tp90_tk3_pe10_b0_resource_only"),
#         "masks": dict(top_k=3, top_p=0.9, p_min_end=0.1),
#         "train_percentile": 75,
#     },
# }

# LOG_REGISTRY: dict[str, dict] = {
#     "AcademicCredentials": {
#         "path": "data/logs/AcademicCredentials/AcademicCredentials_train.csv",
#         "checkpoint_full": _ckpt("AcademicCredentials_DDPS_p75_300_400_tp100_tk100_pe0_b0_full"),
#         "checkpoint_resource_only": _ckpt("AcademicCredentials_DDPS_p75_300_400_tp100_tk100_pe0_b0_resource_only"),
#         "masks": dict(top_k=100, top_p=1, p_min_end=0),
#         "train_percentile": 75,
#     },
#     "BPIC_2012": {
#         "path": "data/logs/BPIC_2012/BPIC_2012_train.csv",
#         "checkpoint_full": _ckpt("BPIC_2012_DDPS_p75_300_3000_tp100_tk100_pe0_b0_full"),
#         "checkpoint_resource_only": _ckpt("BPIC_2012_DDPS_p75_300_3000_tp100_tk100_pe0_b0_resource_only"),
#         "masks": dict(top_k=100, top_p=1, p_min_end=0),
#         "train_percentile": 75,
#     },
#     "BPIC_2017": {
#         "path": "data/logs/BPIC_2017/BPIC_2017_train.csv",
#         "checkpoint_full": _ckpt("BPIC_2017_DDPS_p75_300_7400_tp100_tk100_pe0_b0_full"),
#         "checkpoint_resource_only": _ckpt("BPIC_2017_DDPS_p75_300_7400_tp100_tk100_pe0_b0_resource_only"),
#         "masks": dict(top_k=100, top_p=1, p_min_end=0),
#         "train_percentile": 75,
#     },
# }

LOG_REGISTRY: dict[str, dict] = {
    "BPIC_2012": {
        "path": "data/logs/BPIC_2012/BPIC_2012_train.csv",
        "checkpoint_full": _ckpt("BPIC_2012_DDPS_p75_300_3000_tp90_tk3_pe20_b0_full"),
        "checkpoint_resource_only": _ckpt("BPIC_2012_DDPS_p75_300_3000_tp90_tk3_pe20_b0_resource_only"),
        "masks": dict(top_k=3, top_p=0.9, p_min_end=0.2),
        "train_percentile": 75,
    }
}


#ALL_POLICIES = ("RA-RR", "DM-RR", "DM-GR", "DM-DRL", "DRL-DRL")
ALL_POLICIES = ["DRL-DRL"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPRA matrix evaluation")
    parser.add_argument("--logs", nargs="+", default=list(LOG_REGISTRY.keys()))
    parser.add_argument("--policies", nargs="+", default=list(ALL_POLICIES))
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--sla-percentiles", type=int, nargs="+", default=[95, 90, 75, 50])
    parser.add_argument("--output-dir", default="data/evaluation_results/adhoc_matrix_bpic12")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (log,policy) combos whose simulated logs already exist; re-derive metrics from disk.",
    )
    return parser.parse_args()


def _checkpoint_for(policy: str, log_cfg: dict) -> Optional[str]:
    if policy == "DRL-DRL":
        return log_cfg.get("checkpoint_full")
    if policy == "DM-DRL":
        return log_cfg.get("checkpoint_resource_only")
    return None


def _evaluator_for_log(log_cfg: dict, sla_percentiles: list[int]) -> PolicyEvaluator:
    return PolicyEvaluator(
        original_log_path=log_cfg["path"],
        original_log_names={
            "case": DEFAULT_LOG_COLUMNS["case_id"],
            "start": DEFAULT_LOG_COLUMNS["start_timestamp"],
            "end": DEFAULT_LOG_COLUMNS["end_timestamp"],
        },
        sla_percentiles=list(sla_percentiles),
    )


def _has_existing_run(output_dir: str, log_name: str, policy: str) -> bool:
    sim_log_dir = os.path.join(output_dir, log_name, policy, "simulated_logs")
    if not os.path.isdir(sim_log_dir):
        return False
    return any(f.endswith(".csv") for f in os.listdir(sim_log_dir))


def main() -> int:
    args = parse_args()

    unknown_logs = [l for l in args.logs if l not in LOG_REGISTRY]
    if unknown_logs:
        print(f"Unknown logs: {unknown_logs}. Known: {list(LOG_REGISTRY)}", file=sys.stderr)
        return 1

    unknown_policies = [p for p in args.policies if p not in ALL_POLICIES]
    if unknown_policies:
        print(f"Unknown policies: {unknown_policies}. Known: {ALL_POLICIES}", file=sys.stderr)
        return 1

    all_records: list[dict] = []
    skipped: list[tuple[str, str, str]] = []

    for log_name in args.logs:
        log_cfg = LOG_REGISTRY[log_name]
        evaluator = _evaluator_for_log(log_cfg, args.sla_percentiles)

        for policy in args.policies:
            checkpoint = _checkpoint_for(policy, log_cfg)
            if policy in ("DRL-DRL", "DM-DRL") and (not checkpoint or not os.path.isfile(checkpoint)):
                skipped.append((log_name, policy, f"missing checkpoint: {checkpoint}"))
                print(f"[skip] {log_name}/{policy}: missing checkpoint {checkpoint}")
                continue

            if args.resume and _has_existing_run(args.output_dir, log_name, policy):
                print(f"[resume] {log_name}/{policy}: re-deriving metrics from existing simulated logs")
                _, records = load_per_run_records_from_disk(args.output_dir, log_name, policy, evaluator)
                if records:
                    all_records.extend(records)
                    continue
                # else fall through and re-run
                print(f"[resume] {log_name}/{policy}: no usable cached logs, running fresh")

            try:
                _, records = evaluate_policy_on_log(
                    log_path=log_cfg["path"],
                    policy_name=policy,
                    K=args.K,
                    sla_percentiles=tuple(args.sla_percentiles),
                    train_percentile=log_cfg.get("train_percentile", 75),
                    checkpoint=checkpoint,
                    masks_cfg=log_cfg.get("masks"),
                    output_dir=args.output_dir,
                    log_name=log_name,
                    seed=args.seed,
                    evaluator=evaluator,
                )
                all_records.extend(records)
            except Exception as exc:  # noqa: BLE001
                skipped.append((log_name, policy, f"{type(exc).__name__}: {exc}"))
                print(f"[error] {log_name}/{policy}: {exc}", file=sys.stderr)

    if not all_records:
        print("No records produced; CSVs not written.", file=sys.stderr)
        return 2

    paths = write_all_csvs(all_records, args.output_dir)
    print("\nCSV summaries written:")
    for k, p in paths.items():
        print(f"  {k:>10}: {p}")

    if skipped:
        print("\nSkipped/failed combinations:")
        for log_name, policy, reason in skipped:
            print(f"  {log_name}/{policy}: {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
