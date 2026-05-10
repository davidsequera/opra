"""
Single-experiment evaluation: one (log, policy) pair, K simulations.

Edit the parameters below and run from the project root:

    python src/run_single_evaluation.py
"""

from evaluation.experiment import evaluate_policy_on_log


# --- parameters ---
LOG_PATH = "data/logs/AcademicCredentials/AcademicCredentials_train.csv"
POLICY = "DRL-DRL"                  # RA-RR | DM-RR | DM-GR | DM-DRL | DRL-DRL
K = 10
CHECKPOINT = "data/training_models/AcademicCredentials_DDPS_p75_400_400_tp90_tk3_pe10_a100/checkpoints/best_model.pt"
OUTPUT_DIR = "data/evaluation_results/single"
LOG_NAME = "AcademicCredentials"
TRAIN_PERCENTILE = 75
SLA_PERCENTILES = (95, 90, 75, 50)
MASKS = dict(top_k=3, top_p=0.95, p_min_end=0.1)
SEED = 0


def main():
    evaluate_policy_on_log(
        log_path=LOG_PATH,
        policy_name=POLICY,
        K=K,
        sla_percentiles=SLA_PERCENTILES,
        train_percentile=TRAIN_PERCENTILE,
        checkpoint=CHECKPOINT,
        masks_cfg=MASKS,
        output_dir=OUTPUT_DIR,
        log_name=LOG_NAME,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
