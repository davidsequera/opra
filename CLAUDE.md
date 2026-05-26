# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup

```bash
conda create -n opra_env python=3.11.14
conda activate opra_env
pip install -r requirements.txt
# PyTorch (CPU or CUDA 12.6):
pip install torch torchvision  # CPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # CUDA
```

> `torch` is NOT in `requirements.txt` — it must be installed separately.

### Running

All scripts must be run from the **project root** (not from `src/`), as they use relative paths like `data/logs/...`.

```bash
# Basic DDPS simulation
python src/simulate.py

# Train the full DRL-DRL agent
python src/train.py --log_path data/logs/AcademicCredentials/AcademicCredentials_train.csv \
    --episodes 300 --max_cases 400 --percentile 75 \
    --top_k 2 --top_p 0.9 --p_min_end 0.3 --kl_conformance_coef 0.0 \
    --run_name AcademicCredentials_DDPS_p75_300_400_tp90_tk2_pe30_full

# Train the resource-only DM-DRL agent
python src/train_resource_only.py --log_path data/logs/AcademicCredentials/AcademicCredentials_train.csv \
    --episodes 300 --max_cases 400 --percentile 75 \
    --top_k 2 --top_p 0.9 --p_min_end 0.3 \
    --run_name AcademicCredentials_DDPS_p75_300_400_tp90_tk2_pe30_a0_resource_only

# Train both variants for all registered logs (orchestrator, in-process)
python src/train_all.py

# Evaluate one (log, policy) combo using the library helper
python src/run_single_evaluation.py

# Evaluate the full (log × policy) matrix and emit comparison CSVs
python src/run_matrix_evaluation.py
```

Trained checkpoints land in `data/training_models/<run_name>/checkpoints/`. Evaluation outputs live in `data/evaluation_results/<log>/<policy>/` plus aggregated CSVs at the matrix root.

## Architecture

OPRA is a **policy-based / hexagonal architecture** combining Discrete Event Simulation (SimPy) with Reinforcement Learning (Gymnasium + PyTorch PPO).

### Core data flow

```
Event log (CSV)
    -> Initializer.build()
    -> SimulationSetup (immutable config with all policies)
    -> SimulatorEngine (SimPy-based DDPS)
    -> event_log (list of dicts) -> CSV export
```

For RL:
```
SimulatorEngine (RL mode)
    -> BusinessProcessEnvironment (Gymnasium wrapper)
    -> PPOAgent selects (activity, resource) at each decision point
    -> reward = RewardFunction.compute(case_context)
```

### Key components

**`src/environment/simulator/core/engine.py` — `SimulatorEngine`**
- Two operating modes toggled by `is_rl_mode`:
  - **Standard** (`simulate()`): SimPy runs to completion, policies choose automatically.
  - **RL** (`run_until_decision()`): SimPy pauses at each routing decision, yielding control to the agent via `pending_decisions` queue and `decision_event`.
- `apply_decision(activity, resource)` resumes a paused case.
- Time advances only via `env.timeout(x > 0)` — never via Python `datetime`.

**`src/environment/simulator/core/setup.py` — `SimulationSetup`**
- Frozen dataclass holding all policies: `routing_policy`, `processing_time_policy`, `waiting_time_policy`, `arrival_policy`, `calendar_policy`, `resource_policy`.

**`src/initializer/implementations/DDPSInitializer.py` — Initializers**
- `DDPSInitializer`: builds all policies empirically from the event log (Markov routing, sampled processing times, weekly calendar grid, waiting times).
- `ParametricInitializer` (extends `DDPSInitializer`): overrides arrival (Exponential distribution) and processing time (Normal distribution) to use fitted parametric models.

**`src/environment/core/env.py` — `BusinessProcessEnvironment`**
- Gymnasium `Env` wrapping `SimulatorEngine`.
- `action_space`: `MultiDiscrete([num_activities, num_resources])`.
- State vector `s ∈ ℝ^d`, `d = 3|R| + 2|A| + 5`, structured in four blocks:
  - **Global** (3|R|): per resource — utilization `u_i`, assignment encoding `η_i`, queue pressure `q_i`
  - **Demand** (|A|): per activity — pending demand `κ_j` (cases awaiting execution)
  - **Case** (|A|+3): branching probabilities `b_c`, last activity `ℓ_c`, trace length `λ_c`, SLA urgency `φ_c`
  - **Temporal** (2): hour of day `τ_h`, day of week `τ_d`
- Reward computed by pluggable `RewardFunction` — see **Reward functions** section below.
- Activity/resource masks enforce valid transitions and skill constraints.

**`src/environment/core/reward.py` — Reward Functions**
- `RewardFunction` (ABC): base class for reward computation.
- `CaseRewardContext`: encapsulates case metrics (cycle_time, sla_threshold, num_events, is_completed, chosen_activity_prob).
- `SLARewardFunction` (H=1.0): two-part reward with intermediate and terminal signals.
  - Intermediate (case not yet completed): `(H/1000) × (1 − ct/T)` when on track, `−(H/100) × (ct/T)` when overdue.
  - Terminal (case completed): `+H` if `ct < T`, else `−H`.
- `BinaryRewardFunction`: simple baseline (`+1` if SLA met, `0` otherwise).

**`src/agent/JointAgent/agent.py` — `PPOAgent` / `PPOPolicy`**
- Full DRL-DRL agent. Hierarchical action selection: activity head first, then resource head conditioned on the chosen activity via embedding.
- Activity and resource masks applied as `-1e9` logit fill before sampling.
- `PPOAgent(kl_conformance_coef=0.0)`: optional KL conformance auxiliary loss added to the PPO objective. Penalises divergence of the agent's activity distribution π(a|s) from the empirical routing distribution P(a|a_last) extracted from the state vector (Block B, first `|A|` elements). Applied over unmasked activities only. Set to `0.0` to disable (default).
- `PPOPolicy.evaluate()` returns `(log_prob, entropy, value, activity_dist.probs)` — the fourth element is used by the KL conformance loss in `update()`.

**`src/agent/ResourcesOnlyAgent/resource_only_agent.py` — `PPOResourceOnlyAgent` / `PPOResourceOnlyPolicy`**
- DM-DRL backbone: same architecture minus the activity head. Activity is supplied externally each step (during training, sampled from `EmpiricalDMActivitySelector`).
- Exposes the same `policy_old.get_resource_logits(state, activity)` shape as the full agent so the evaluation `DRLResourceSelector` is agnostic to which agent it holds.

### Policy interfaces are a hexagonal boundary

Policies under `src/environment/simulator/policies/` are the **ports** of the hexagonal architecture — abstract contracts between the simulator core and any pluggable implementation (empirical sampler, parametric distribution, future ML predictor, …). Treat them as stable.

Rules of thumb:

- **Do NOT add methods to a policy ABC just because a single consumer wants something specific.** Adding `get_expected_duration` to `ProcessingTimePolicy` to support one heuristic baseline silently breaks every existing implementation and forces every future implementation (e.g. an LSTM-based duration predictor) to provide it — even when "expected value" isn't a natural concept for that model.
- **Build the consumer's logic outside the policy**, using only the existing public interface. Example: `GreedyProcessingTimeResourceSelector` in `src/evaluation/selectors/resource.py` estimates the expected duration by averaging N draws of `policy.get_activity_duration(...)`. The policy stays unaware of the consumer; consumers can be added, removed, or specialized freely.
- **If the interface really needs to change**, propose it explicitly. An interface change is a contract change: every existing subclass must be updated, and the abstract method must remain meaningful for plausible future implementations (otherwise it's a leaky abstraction). Don't sneak the change in through a default implementation either — defaults hide the cost of an API decision and the next implementation will either ignore the method or reimplement it inconsistently.

This applies to all policy ABCs: `RoutingPolicy`, `ProcessingTimePolicy`, `ArrivalPolicy`, `CalendarPolicy`, `WaitingTimePolicy`, `ResourceAllocationPolicy`. The selector layer (`src/evaluation/selectors/`) and any future heuristic should respect the same boundary.

### Policies

All policies live under `src/environment/simulator/policies/` (abstract base classes) with implementations in `src/environment/simulator/implementations/`:

| Policy | Empirical | Parametric |
|---|---|---|
| Routing | `ProbabilisticRoutingPolicy` (1st-order Markov), `SecondOrderRoutingPolicy` (2nd-order Markov) | — |
| Processing Time | `EmpiricalResourceActivityProcessingTimePolicy` | `NormalProcessingTimePolicy` |
| Arrival | `EmpiricalArrivalPolicy`, `WeeklyArrivalPolicy` | `ExponentialArrivalPolicy` |
| Calendar | `WeeklyCalendarPolicy` (7×24 global grid), `WeeklyResourceCalendarPolicy` (per-resource availability) | — |
| Waiting Time | `ExtraneousWaitingTimePolicy` (stratified by activity-resource pair) | — |
| Resource | `SkillBasedResourcePolicy` | — |

To add a new policy: implement the abstract base in `policies/`, place the implementation in `implementations/`, and wire it in the relevant `Initializer.build()`.

### Time semantics

- SimPy `env.now` is a numeric scalar in the configured `time_unit` (`"seconds"`, `"minutes"`, `"hours"`).
- Absolute timestamps are only used during initialization (log parsing) and when exporting results (`convert_to_absolute_time=True`), anchored by `start_timestamp`.
- **Never** pass Python `datetime` objects into the SimPy engine.

### Input data format

CSV event logs require columns (names are configurable via `LogColumnNames`):
- `caseid`, `Activity_1`, `Resource_1`, `start_timestamp`, `end_timestamp`

Simulated output logs use: `case`, `activity`, `resource`, `start`, `end`.

## Research context

This is a master's thesis project at Universidad de los Andes. Understanding the research goal is important for making correct design decisions.

### Problem framing

The process optimization problem is framed as an MDP:
- **State**: `s ∈ ℝ^d` snapshot at each decision point — global resource utilization/assignment/queue pressure, per-activity pending demand, case-specific features (branching probs, last activity, trace length, SLA urgency), and temporal features (hour, day). Dimension `d = 3|R| + 2|A| + 5`.
- **Action**: a joint `(activity, resource)` pair — the agent selects *both* what to do next *and* who does it. This is the key novelty over prior work, which fixes control-flow and only optimizes resource assignment.
- **Reward**: SLA compliance — a case is a success if its cycle time falls below a threshold `T` (defined as a percentile of the original log's cycle time distribution, e.g. p75 or p90).

### Masking

Activity masks use **top-k / top-p** (nucleus) filtering over learned branching probabilities to keep agent decisions within plausible process behavior. Resource masks enforce skill constraints (`resource.skills` must contain the chosen activity). Both masks are applied as `-1e9` logit fill before softmax, which is the standard approach.

### Reward function

The thesis targets a two-part reward encouraging SLA compliance:
- **Intermediate** (case not yet completed): directional signal based on cycle time proximity to SLA threshold `T`.
- **Terminal** (case completed): binary bonus `+H` if `ct < T`, else penalty `-H`.

**Current implementations** (in `src/environment/core/reward.py`):

1. **`SLARewardFunction`** — the primary implementation. For intermediate states: linear scaling `(H/1000) × (1 − ct/T)` when on track, or `−(H/100) × (ct/T)` when overdue. Terminal: `+H` / `−H`.

2. **`BinaryRewardFunction`** — legacy simplified version (kept for baselines and ablations).

**KL conformance regularization** is applied at the PPO loss level (not the reward level) via `PPOAgent(kl_conformance_coef=...)`. See the agent section above.


### Evaluation framework

Library-style — the matrix script is just an orchestrator over composable parts. All evaluation code lives in `src/evaluation/`:

```
src/evaluation/
    selectors/
        activity.py     # ActivitySelector + Random / GreedyProb / EmpiricalDM / DRL
        resource.py     # ResourceSelector + Random / DRL
        factory.py      # build_policy(name, agent, rng) -> (act_sel, res_sel)
    runner.py           # run_episode(env, simulator, act_sel, res_sel) — one episode loop
    experiment.py       # evaluate_policy_on_log(...) — K runs for ONE (log, policy)
    csv_export.py       # writes runs_long.csv / runs_wide.csv / aggregated.csv
    metrics/            # PolicyEvaluator + per-run PerformanceResult / SimilarityResult
                        # + aggregation (mean ± 95% CI via t-distribution)
    training/           # TrainingMetricsTracker, EpisodeMetrics, UpdateMetrics, etc.
```

**Selector abstraction.** Every decision policy is an `(ActivitySelector, ResourceSelector)` pair. `run_episode` is the single integration point — the same loop drives RA-RR, DM-RR, DM-GR, DM-DRL, and DRL-DRL. To add a policy, implement one or both interfaces; no env or runner changes needed.

**Single (log, policy) experiment.** `evaluate_policy_on_log(...)` builds the setup, loads the right checkpoint (full PPOAgent for DRL-DRL, PPOResourceOnlyAgent for DM-DRL, none for the rest), runs K simulations, exports per-run CSV logs, and returns `(AggregatedResults, list[per_run_record])`. Use it from `run_single_evaluation.py` or directly in a notebook.

**Matrix.** `run_matrix_evaluation.py` loops `(log × policy)`, sharing one `PolicyEvaluator` per log so reference compliance baselines for CIR are identical across policies. Combinations whose checkpoints don't exist are skipped (logged at the end). `--resume` re-derives metrics from previously-exported simulated logs without re-simulating.

`LOG_REGISTRY` in `run_matrix_evaluation.py` mirrors the experimental conditions in `train_all.py` — four commented/active blocks, one active at a time:

| Registry | Mask | KL reg | Episodes |
|---|---|---|---|
| Flexibility — no KL | top-k/p nucleus | 0 | 300 |
| Flexibility — with KL | top-k/p nucleus | per-log | 250 |
| No Flexibility — no KL | top_k=100, top_p=1 | 0 | 300 |
| **No Flexibility — with KL** *(active)* | top_k=100, top_p=1 | per-log | 250 |

Checkpoint names in `LOG_REGISTRY` must match the run name format produced by `train_all.py` exactly. KL-regularized registries omit `checkpoint_resource_only` (no resource-only models were trained with KL); DM-DRL is skipped automatically with a `[skip]` message for those configs.

**Three CSV shapes**, all from the same in-memory record list:
- `runs_long.csv` — `log, policy, run_id, metric, value` for melt/groupby plotting.
- `runs_wide.csv` — one row per `(log, policy, run_id)`, one column per metric.
- `aggregated.csv` — one row per `(log, policy)` with `<metric>_mean` and `<metric>_ci95` columns. The paper table.

**Per-policy `results.json`** is also written under `output_dir/<log>/<policy>/` for backward compat with the prior pipeline.

**Performance metrics** (`src/evaluation/metrics/functions/performance_metrics.py`): per-threshold CR (T95/T90/T75/T50), CIR (compliance improvement ratio vs. original log), cycle time (mean/median/std/quantiles), resource utilization CV.

**Similarity metrics** (`src/evaluation/metrics/functions/similarity_metrics.py`) wrap the `log-distance-measures` package:
- **Control-flow**: N-gram distance (NGD)
- **Temporal**: Absolute (AED), Circadian (CED), and Relative (RED) event distributions
- **Resource**: Circadian Workforce Distribution (CWD)
- **Congestion**: Case Arrival Rate (CAR) and Cycle Time Distribution (CTD)

### Baselines

The five evaluation policies (`src/evaluation/selectors/factory.py::build_policy`):

Names follow `{activity}-{resource}`:

| Name    | Activity selection                                | Resource selection                                  | Checkpoint           |
|---------|---------------------------------------------------|-----------------------------------------------------|----------------------|
| RA-RR   | Random uniform over feasible activities            | Random over feasible                                 | —                    |
| DM-RR   | Sample from routing-policy probabilities (the "DM")| Random                                              | —                    |
| DM-GR   | Sample from routing-policy probabilities           | Greedy: argmin estimated processing time (heuristic) | —                    |
| DM-DRL  | Sample from routing-policy probabilities           | `PPOResourceOnlyAgent`                              | resource-only ckpt   |
| DRL-DRL | `PPOAgent` activity head                           | `PPOAgent` resource head                            | full ckpt            |

DM-GR is the domain-knowledge heuristic baseline: stochastic activity sampling (avoids loops on argmax-collapse) paired with a `GreedyProcessingTimeResourceSelector` that estimates expected duration by averaging N draws of the existing `processing_time_policy.get_activity_duration(activity, resource)` interface (no API change to the policy), and picks the resource with the lowest estimate. Empirically this collapses cycle times into a tight band positioned between T50 and T95 — best T95 compliance, worst T50 — because the heuristic ignores congestion and overloads the historically-fast resources.

The "decision model" (DM) for DM-RR / DM-DRL is the empirical `ProbabilisticRoutingPolicy` (1st-order Markov), not an LSTM — that simplification was deliberate to avoid bringing in a separate next-activity predictor.

### Training scripts

- **`src/train.py::train_full_agent(...)`** — full DRL-DRL PPO training. CLI `main()` is a thin wrapper around the function. Always uses `SLARewardFunction()`. Accepts `--kl_conformance_coef` (default `0.0`) to enable the KL conformance auxiliary loss in the PPO update.
- **`src/train_resource_only.py::train_resource_only_agent(...)`** — analog for `PPOResourceOnlyAgent`. Activity at each decision is sampled from `EmpiricalDMActivitySelector` so the agent only learns to optimize resource allocation under a fixed control-flow distribution. Always uses plain `SLARewardFunction()`.
- **`src/train_all.py`** — orchestrator. Imports both training functions and calls them in-process (no subprocess) for every `(log, variant)` combo in `TRAINING_REGISTRY`. Per-log `kl_conformance_coef` can be set directly in the registry entry. Prints `[N/total]` progress with ETA between runs. Run names follow `{LogName}_DDPS_p{pctile}_{episodes}_{max_cases}_tp{top_p*100}_tk{top_k}_pe{p_min_end*100}_kl{kl_conformance_coef*100}_{variant}`. The matrix runner's `LOG_REGISTRY` references checkpoints under those exact names.

  `TRAINING_REGISTRY` has four commented/active blocks (one active at a time) matching the four experimental conditions:

  | Registry | Mask | KL reg |
  |---|---|---|
  | Flexibility — no KL | top-k/p nucleus | 0 |
  | Flexibility — with KL | top-k/p nucleus | per-log |
  | No Flexibility — no KL | top_k=100, top_p=1 | 0 |
  | **No Flexibility — with KL** *(active)* | top_k=100, top_p=1 | per-log |

## Known issues / stubs

- `ResourceAllocationPolicy` and `StoppingPolicy` are planned but not fully implemented.
- The engine guards against infinite loops via `max_cases`, but zero-duration activities can still cause issues.
- The "decision model" used by DM-RR / DM-DRL is the empirical 1st-order Markov routing policy, not an LSTM. Replacing it with a learned next-activity model would be a `RoutingPolicy` subclass plugged into the same `EmpiricalDMActivitySelector` slot.
