"""
OPRA Evaluation Script.

Compares a simulated event log against the original using
performance + similarity metrics from the thesis.

Usage:
    python src/evaluate.py
    python src/evaluate.py --original data/logs/LoanApp/LoanApp.csv --simulated data/simulated_logs/LoanApp/LoanApp_DDPS.csv
"""

import argparse
import glob
import os

from evaluation.metrics import PolicyEvaluator


ORIGINAL_LOG_NAMES = {
    "case":     "case_id",
    "activity": "activity",
    "resource": "resource",
    "start":    "start_time",
    "end":      "end_time",
}


def parse_args():
    parser = argparse.ArgumentParser(description="OPRA Evaluation")
    parser.add_argument("--original",  type=str, default="data/logs/LoanApp/LoanApp.csv")
    parser.add_argument("--simulated", type=str, default="data/simulated_logs/LoanApp/LoanApp_DDPS.csv")
    parser.add_argument("--sla_percentiles", type=int, nargs="+", default=[95, 90, 75])
    parser.add_argument("--policy_name", type=str, default="DDPS")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    evaluator = PolicyEvaluator(
        original_log_path=args.original,
        original_log_names=ORIGINAL_LOG_NAMES,
        sla_percentiles=args.sla_percentiles,
    )

    paths = (
        sorted(glob.glob(os.path.join(args.simulated, "*.csv")))
        if os.path.isdir(args.simulated)
        else [args.simulated]
    )

    results = evaluator.evaluate_policy(
        simulated_log_paths=paths,
        policy_name=args.policy_name,
        log_name="evaluation",
    )
    evaluator.print_results(results)

    if args.output_dir:
        evaluator.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
