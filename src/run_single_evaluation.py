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
CHECKPOINT = "data/training_models/AcademicCredentials_DDPS_p75_300_400_tp0.9_tk2_pe30_b100_full/checkpoints/final_model.pt"
OUTPUT_DIR = "data/evaluation_results/AcademicCredentials_DDPS_p75_300_400_tp0.9_tk2_pe30_b100_full"
LOG_NAME = "AcademicCredentials"
TRAIN_PERCENTILE = 75
SLA_PERCENTILES = (95, 90, 75, 50)
MASKS = dict(top_k=2, top_p=0.7, p_min_end=0.3)
#MASKS = dict(top_k=100, top_p=1, p_min_end=0)
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
