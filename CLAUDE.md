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

# RL experiment (PPO agent)
python src/main.py

# Evaluate simulated logs vs original
python src/evaluate.py
```

Output logs land in `data/simulated_logs/PurchasingExample/`.

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
- `SLARewardFunction` (K=1.0): two-part reward with intermediate and terminal signals. 
  - Intermediate (case not yet completed): `r = ±K/10 × (ct/T)` directional signal based on SLA proximity.
  - Terminal (case completed): `r = +K` if `ct < T`, else `-K`.
- `RegularizedSLARewardFunction` (K=1.0, alpha=0.0): extends `SLARewardFunction` with distributional regularization. Positive rewards are scaled by `α(Prob(A) − 1) + 1`, where `Prob(A)` is the as-is routing probability of the chosen activity. This encourages the agent to stay close to the empirical routing distribution.
- `BinaryRewardFunction`: simple baseline (`+1` if SLA met, `0` otherwise).

**`src/agent/agent.py` — `PPOAgent` / `PPOPolicy`**
- Hierarchical action selection: activity head first, then resource head conditioned on chosen activity via embedding.
- Activity and resource masks applied as `-1e9` logit fill before sampling.

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
- **Terminal** (case completed): binary bonus `+K` if `ct < T`, else penalty `-K`.

**Current implementations** (in `src/environment/core/reward.py`):

1. **`SLARewardFunction`** — the primary implementation, which replaces the prior simplified version. For intermediate states: linear scaling `(K/1000) × (1 − ct/T)` when on track, or `−(K/10) × (ct/T)` when overdue. Matches the thesis definition's intent.

2. **`RegularizedSLARewardFunction`** — extends `SLARewardFunction` with an optional distributional regularization term (`alpha` parameter). Scales positive rewards by the empirical routing probability of the chosen activity, encouraging the agent to stay near the learned process distribution while rewarding SLA improvements. Set `alpha=0` to disable regularization (default behavior = `SLARewardFunction`).

3. **`BinaryRewardFunction`** — legacy simplified version (kept for baselines and ablations).


### Evaluation

Simulated logs are compared against the original using the `log-distance-measures` package (`src/evaluate.py`) across:
- **Control-flow**: N-gram distance (NGD)
- **Temporal**: Absolute (AED), Circadian (CED), and Relative (RED) event distributions
- **Resource**: Circadian Workforce Distribution (CWD)
- **Congestion**: Case Arrival Rate (CAR) and Cycle Time Distribution (CTD)

### Baselines (for comparison when implementing new policies)

| Name | Activity selection | Resource selection |
|---|---|---|
| RA-RR | Proportional to branching probs | Random |
| GP-RR | Greedy (argmax branching prob) | Random |
| DM-RR | ML model (LSTM) | Random |
| DM-DRL | ML model (LSTM) | DRL agent |
| DRL-AR | DRL agent (joint) | DRL agent (joint) |

## Known issues / stubs

- `ResourceAllocationPolicy` and `StoppingPolicy` are planned but not fully implemented.
- The engine guards against infinite loops via `max_cases`, but zero-duration activities can still cause issues.
- `RegularizedSLARewardFunction` contains debug print statements (line 136 of `reward.py`) that should be removed in production use.
