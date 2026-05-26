"""
Library-style orchestrator: trains both PPO agents (full DRL-DRL + resource-only
DM-DRL) for every registered log, calling the training functions in-process.

Run names follow the format:
    {LogName}_DDPS_p{percentile}_{episodes}_{max_cases}_tp{top_p*100}_tk{top_k}_pe{p_min_end*100}_b{beta*100}_{variant}

Defaults: percentile=75, episodes=300, beta=0 (no regularization).

Usage (from the project root):
    python src/train_all.py
    python src/train_all.py --logs AcademicCredentials
    python src/train_all.py --variants full
    python src/train_all.py --episodes 5 --logs AcademicCredentials
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

from train import train_full_agent
from train_resource_only import train_resource_only_agent

## Flexibility experiment logs
# TRAINING_REGISTRY: dict[str, dict] = {
#     "AcademicCredentials": {
#         "path": "data/logs/AcademicCredentials/AcademicCredentials_train.csv",
#         "max_cases": 400,        # 398 rounded
#         "top_k": 2,
#         "top_p": 0.9,
#         "p_min_end": 0.3,
#     },
#     "BPIC_2012": {
#         "path": "data/logs/BPIC_2012/BPIC_2012_train.csv",
#         "max_cases": 3000,       # 3030 rounded
#         "top_k": 2,
#         "top_p": 0.9,
#         "p_min_end": 0.2,
#     },
#     "BPIC_2017": {
#         "path": "data/logs/BPIC_2017/BPIC_2017_train.csv",
#         "max_cases": 7400,       # 7402 rounded
#         "top_k": 3,
#         "top_p": 0.9,
#         "p_min_end": 0.1,
#     },
# }

## No Flexibility experiment
# TRAINING_REGISTRY: dict[str, dict] = {
#     "AcademicCredentials": {
#         "path": "data/logs/AcademicCredentials/AcademicCredentials_train.csv",
#         "max_cases": 400,        # 398 rounded
#         "top_k": 100,
#         "top_p": 1,
#         "p_min_end": 0,
#     },
#     "BPIC_2012": {
#         "path": "data/logs/BPIC_2012/BPIC_2012_train.csv",
#         "max_cases": 3000,       # 3030 rounded
#         "top_k": 100,
#         "top_p": 1,
#         "p_min_end": 0,
#     },
#     "BPIC_2017": {
#         "path": "data/logs/BPIC_2017/BPIC_2017_train.csv",
#         "max_cases": 7400,       # 7402 rounded
#         "top_k": 100,
#         "top_p": 1,
#         "p_min_end": 0,
#     },
# }



## Flexibility experiment logs with KL conformance regularization
# TRAINING_REGISTRY: dict[str, dict] = {
#     "AcademicCredentials": {
#         "path": "data/logs/AcademicCredentials/AcademicCredentials_train.csv",
#         "max_cases": 400,        # 398 rounded
#         "top_k": 2,
#         "top_p": 0.9,
#         "p_min_end": 0.3,
#         "kl_conformance_coef": 0.08, # Add KL conformance regularization for the full agent on this log
#     },
#     "BPIC_2012": {
#         "path": "data/logs/BPIC_2012/BPIC_2012_train.csv",
#         "max_cases": 3000,       # 3030 rounded
#         "top_k": 2,
#         "top_p": 0.9,
#         "p_min_end": 0.2,
#         "kl_conformance_coef": 0.01, # Add KL conformance regularization for the full agent on this log
#     },
#     "BPIC_2017": {
#         "path": "data/logs/BPIC_2017/BPIC_2017_train.csv",
#         "max_cases": 7400,       # 7402 rounded
#         "top_k": 3,
#         "top_p": 0.9,
#         "p_min_end": 0.1,
#         "kl_conformance_coef": 0.05, # Add KL conformance regularization for the full agent on this log
#     },
# }


## No Flexibility experiment logs with KL conformance regularization
TRAINING_REGISTRY: dict[str, dict] = {
    "AcademicCredentials": {
        "path": "data/logs/AcademicCredentials/AcademicCredentials_train.csv",
        "max_cases": 400,        # 398 rounded
        "top_k": 100,
        "top_p": 1,
        "p_min_end": 0,
        "kl_conformance_coef": 0.08, # Add KL conformance regularization for the full agent on this log
    },
    "BPIC_2012": {
        "path": "data/logs/BPIC_2012/BPIC_2012_train.csv",
        "max_cases": 3000,       # 3030 rounded
        "top_k": 100,
        "top_p": 1,
        "p_min_end": 0,
        "kl_conformance_coef": 0.01, # Add KL conformance regularization for the full agent on this log
    },
    "BPIC_2017": {
        "path": "data/logs/BPIC_2017/BPIC_2017_train.csv",
        "max_cases": 7400,       # 7402 rounded
        "top_k": 100,
        "top_p": 1,
        "p_min_end": 0,
        "kl_conformance_coef": 0.05, # Add KL conformance regularization for the full agent on this log
    },
}



#VARIANTS = ("full", "resource_only")
VARIANTS = ["full"]
DEFAULT_PERCENTILE = 75
DEFAULT_EPISODES = 250
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all PPO agents per log")
    parser.add_argument("--logs", nargs="+", default=list(TRAINING_REGISTRY.keys()))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--percentile", type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a (log, variant) if its best_model.pt already exists.")
    parser.add_argument("--dry-run", action="store_true", help="List the planned runs and exit.")
    return parser.parse_args()


def build_run_name(log_name: str, log_cfg: dict, args: argparse.Namespace, variant: str) -> str:
    return (
        f"{log_name}"
        f"_DDPS_p{args.percentile}"
        f"_{args.episodes}_{log_cfg['max_cases']}"
        f"_tp{int(round(log_cfg['top_p'] * 100))}"
        f"_tk{log_cfg['top_k']}"
        f"_pe{int(round(log_cfg['p_min_end'] * 100))}"
        f"_kl{int(round(log_cfg['kl_conformance_coef'] * 100))}"
        f"_{variant}"
    )


def checkpoint_path(run_name: str) -> str:
    return os.path.join("data/training_models", run_name, "checkpoints", "best_model.pt")


def _train_one(log_cfg: dict, variant: str, run_name: str, args: argparse.Namespace) -> str:
    """Dispatch to the right training function."""
    if variant == "full":
        return train_full_agent(
            log_path=log_cfg["path"],
            episodes=args.episodes,
            max_cases=log_cfg["max_cases"],
            percentile=args.percentile,
            top_k=log_cfg["top_k"],
            top_p=log_cfg["top_p"],
            p_min_end=log_cfg["p_min_end"],
            lr=args.lr,
            seed=args.seed,
            save_every=args.save_every,
            run_name=run_name,
            kl_conformance_coef=log_cfg["kl_conformance_coef"],
        )
    if variant == "resource_only":
        return train_resource_only_agent(
            log_path=log_cfg["path"],
            episodes=args.episodes,
            max_cases=log_cfg["max_cases"],
            percentile=args.percentile,
            top_k=log_cfg["top_k"],
            top_p=log_cfg["top_p"],
            p_min_end=log_cfg["p_min_end"],
            lr=args.lr,
            seed=args.seed,
            save_every=args.save_every,
            run_name=run_name,
        )
    raise ValueError(f"Unknown variant: {variant}")


def main() -> int:
    args = parse_args()

    unknown = [l for l in args.logs if l not in TRAINING_REGISTRY]
    if unknown:
        print(f"Unknown logs: {unknown}. Known: {list(TRAINING_REGISTRY)}", file=sys.stderr)
        return 1

    plan: list[tuple[str, str, str]] = []
    for log_name in args.logs:
        log_cfg = TRAINING_REGISTRY[log_name]
        for variant in args.variants:
            run_name = build_run_name(log_name, log_cfg, args, variant)
            if args.skip_existing and os.path.isfile(checkpoint_path(run_name)):
                print(f"[skip] {log_name}/{variant}: {run_name}/best_model.pt already present")
                continue
            plan.append((log_name, variant, run_name))

    if not plan:
        print("Nothing to train.")
        return 0

    print(f"\nTraining plan ({len(plan)} runs):")
    for log_name, variant, run_name in plan:
        print(f"  {log_name}/{variant} -> {run_name}")

    if args.dry_run:
        return 0

    durations: list[float] = []
    failures: list[tuple[str, str, str]] = []
    overall_start = time.time()
    n = len(plan)

    def _fmt(seconds: float) -> str:
        return str(timedelta(seconds=int(seconds)))

    for i, (log_name, variant, run_name) in enumerate(plan, start=1):
        avg = sum(durations) / len(durations) if durations else None
        eta_str = f" | est. remaining: {_fmt(avg * (n - i + 1))}" if avg else ""
        elapsed_str = _fmt(time.time() - overall_start)
        now = datetime.now().strftime("%H:%M:%S")

        print(
            f"\n========== [{i}/{n}] {now} | elapsed {elapsed_str}{eta_str}\n"
            f"           {log_name}/{variant} -> {run_name}\n"
            f"=========="
        )

        run_start = time.time()
        try:
            _train_one(TRAINING_REGISTRY[log_name], variant, run_name, args)
            run_elapsed = time.time() - run_start
            durations.append(run_elapsed)
            print(f"---------- [{i}/{n}] OK in {_fmt(run_elapsed)} ----------")
        except Exception as exc:  # noqa: BLE001
            run_elapsed = time.time() - run_start
            durations.append(run_elapsed)
            failures.append((log_name, variant, f"{type(exc).__name__}: {exc}"))
            print(f"---------- [{i}/{n}] FAILED in {_fmt(run_elapsed)}: {exc} ----------", file=sys.stderr)
            traceback.print_exc()

    print(f"\nTotal elapsed: {_fmt(time.time() - overall_start)}")

    if failures:
        print("\nFailed runs:")
        for log_name, variant, reason in failures:
            print(f"  {log_name}/{variant}: {reason}")
        return 2

    print("All training runs complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
