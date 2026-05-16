# OPRA: Optimal Predictor of Resources and Activities

## Overview

OPRA (Optimal Predictor of Resources and Activities) is a hybrid simulation-optimization framework designed to improve business processes. At its core, it implements a **plug-and-play discrete-event process simulator** built on top of **SimPy**, driven primarily by **event logs**. It combines a data-driven Discrete Event Simulator (DDPS) with cutting-edge Machine Learning (ML) and Reinforcement Learning (RL) techniques to find optimal resource allocation policies.

The core idea is to:
- Take an event log
- Infer behavioral components (routing, timing, arrivals, calendars, etc.)
- Assemble them into a simulation using interchangeable **policies**
- Allow mixing paradigms: classic DDPS, empirical distributions, and ML-based models

While providing a powerful "what-if" analysis tool to model existing business processes, the primary objective of this project is to leverage RL agents to explore and discover new, more efficient policies for allocating resources, routing cases, and managing temporal aspects of the process. The simulator is the core of this framework, providing the environment for the RL agent to learn and improve. It's intended to be easy to use for beginners (reasonable defaults) and highly extensible for advanced users and research use cases.

## Problem Statement

In many business processes, allocating the right resource to the right task at the right time is critical for efficiency, cost reduction, and meeting service-level agreements (SLAs). However, finding the optimal allocation policy is a complex combinatorial problem, especially in dynamic environments with stochastic events and complex constraints (like resource calendars). Traditional analytical methods often fall short.

OPRA addresses this by framing the resource allocation problem as a reinforcement learning task. An RL agent learns a policy by interacting with a realistic simulation of the business process.

## Key Features

*   **Powerful Discrete Event Simulation Core:** The simulator, built on SimPy, offers a flexible and robust core for modeling complex business processes.
*   **Pluggable and Extensible Policies:** All behavioral aspects of the simulation (e.g., routing, processing times, resource allocation, arrivals, calendars) are defined by interchangeable policies. These can range from simple rule-based policies to empirical distributions, or advanced ML/RL-based models.
*   **Data-Driven Initialization:** The `Initializer` automatically configures simulation parameters (e.g., arrival rates, activity durations, routing probabilities) from real-world event logs (XES/CSV format), facilitating easy setup from existing data.
*   **Realistic Resource Calendars:** Model and incorporate realistic resource availability, including shifts, breaks, and holidays, influencing activity execution.
*   **Reinforcement Learning Environment:** The framework provides a standard interface (similar to OpenAI Gym) for an RL agent to interact with the simulation, receive observations, take actions, and get rewards, enabling learning of optimal policies.

## Architecture

### High-level Architecture

The simulator follows a **policy-based / hexagonal architecture**.

### Core Concepts

-   **SimulatorEngine**: Orchestrates the simulation loop, owns the SimPy environment (`env`), executes cases and activities, and collects an event log as output.
-   **SimulationSetup**: An immutable configuration object that assembles all policies and global settings, passed into the simulator at initialization.
-   **Initializer**: Builds a `SimulationSetup` from an event log, encapsulating all log-to-behavior inference. Different initializers may exist (DDPS, ML-based, hybrid).

The project is structured into several key components:
*   `src/environment`: Contains the core simulation logic, including entities, the simulator engine, and policies.
*   `src/agent`: Contains the implementation of the RL agent that learns the optimal policy.
*   `src/initializer`: Responsible for reading data (like event logs) and setting up the simulation environment.
*   `main.py`: The main entry point to run a simulation experiment.

### Time Semantics (Important)

*   **Internal simulation time**: Represented as a numeric scalar (`env.now`), with configurable units (seconds, minutes, hours). All policies operate in internal time.
*   **Absolute time**: Only used for calendar inference, arrival inference, and exporting results back to real timestamps, anchored via a single `start_timestamp`.
    *   SimPy time MUST NOT depend on Python `datetime`.
    *   Calendars translate absolute time → internal availability.
    *   Processing and waiting times are durations, not datetimes.

### Policies (Core Extension Points)

All behavior is expressed via **policies**. Policies must be stateless or internally self-contained, replaceable without modifying the engine, and deterministic given a random seed (when applicable). Key policies include: `RoutingPolicy`, `ProcessingTimePolicy`, `WaitingTimePolicy` (planned), `ArrivalPolicy`, `CalendarPolicy`, `ResourceAllocationPolicy` (planned), and `Stopping/Termination Policy`.

## How it Works

1.  **Initialization:** The `Initializer` reads an event log and a process model to configure the simulation parameters.
2.  **Simulation Loop:** The `simulator/core/engine.py` runs the simulation by processing events from an event queue.
3.  **Decision Points:** When a decision is needed, the simulator calls the corresponding policy.
4.  **Agent Interaction:** If an RL-based policy is used, the simulator passes the current state to the `agent`. The agent selects an action.
5.  **State Transition:** The simulator executes the action, and the simulation state changes.
6.  **Feedback:** The simulator calculates a reward based on the outcome and new state, sent back to the agent for learning.
7.  **Learning:** The agent uses this feedback to update its internal model to make better decisions.

This cycle of `state -> action -> reward -> new state` continues, allowing the agent to learn an optimal policy for resource management within the simulated environment.

## Design Principles (Non-negotiable)

-   Separation of concerns
-   No hard-coded behavior
-   Policies over conditionals
-   Internal time is numeric and consistent
-   Logs ≠ process semantics
-   Defaults must be safe for beginners
-   Advanced users can override everything

## Repository Structure

```text
opra/
├── CLAUDE.md
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── logs/                        # Input event logs (CSV)
│   ├── training_models/             # Checkpoints and training metrics
│   └── evaluation_results/          # Evaluation CSVs and aggregated results
├── docs/
│   ├── architecture.dsl
│   ├── environment.md
│   └── Thesis_Proposal.pdf
└── src/
    ├── agent/
    │   ├── JointAgent/              # Full DRL-DRL PPO agent
    │   └── ResourcesOnlyAgent/      # Resource-only DM-DRL agent
    ├── environment/
    │   ├── core/
    │   │   ├── env.py               # Gymnasium wrapper (BusinessProcessEnvironment)
    │   │   ├── mask.py              # Nucleus / top-k masking
    │   │   └── reward.py            # SLA / Regularized / Binary reward functions
    │   ├── simulator/
    │   │   ├── core/                # SimulatorEngine, SimulationSetup
    │   │   ├── implementations/     # Empirical, parametric, distribution impls
    │   │   └── policies/            # Abstract policy interfaces (hexagonal boundary)
    │   └── entities/                # Activity, Case, Resource, Events
    ├── evaluation/
    │   ├── selectors/               # ActivitySelector / ResourceSelector impls + factory
    │   ├── metrics/                 # Performance + similarity metric functions
    │   ├── training/                # TrainingMetricsTracker, episode/update metrics
    │   ├── runner.py                # run_episode — single episode loop
    │   ├── experiment.py            # evaluate_policy_on_log — K runs for one (log, policy)
    │   └── csv_export.py            # runs_long / runs_wide / aggregated CSV writers
    ├── initializer/
    │   ├── implementations/
    │   │   ├── DDPSInitializer.py
    │   │   └── ParametricInitializer.py
    │   └── Initializer.py
    ├── simulate.py                  # Basic DDPS simulation
    ├── train.py                     # Train full DRL-DRL agent
    ├── train_resource_only.py       # Train resource-only DM-DRL agent
    ├── train_all.py                 # Orchestrator: trains all (log × variant) combos
    ├── run_single_evaluation.py     # Evaluate one (log, policy) combo
    └── run_matrix_evaluation.py     # Evaluate full (log × policy) matrix
```

## Getting Started

### Prerequisites

- Python 3.11 (recommended via conda)
- PyTorch (CPU or CUDA — installed separately, see below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdaptiveBProcess/Optimal-Predictor-of-Resources-and-Activities.git
   cd opra
   ```

2. **Create and activate the environment:**
   ```bash
   conda create -n opra_env python=3.11.14
   conda activate opra_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch** (not in `requirements.txt`):
   ```bash
   # CPU
   pip install torch torchvision
   # CUDA 12.6
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

> All scripts must be run from the **project root**, not from `src/`, as they use relative paths like `data/logs/...`.

---

## Running

### 1. Basic Discrete-Event Simulation

Runs a DDPS and writes a simulated event log to `data/simulated_logs/`:

```bash
python src/simulate.py
```

### 2. Train the Full DRL-DRL Agent

Trains a joint `(activity, resource)` PPO agent:

```bash
python src/train.py \
    --log_path data/logs/AcademicCredentials/AcademicCredentials_train.csv \
    --episodes 300 \
    --max_cases 400 \
    --percentile 75 \
    --top_k 2 \
    --top_p 0.9 \
    --p_min_end 0.3 \
    --alpha 0.0 \
    --run_name AcademicCredentials_run01
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--log_path` | required | Path to the training event log CSV |
| `--episodes` | 300 | Number of training episodes |
| `--max_cases` | log size | Cases simulated per episode |
| `--percentile` | 75 | SLA threshold percentile (T75, T90, etc.) |
| `--top_k` | 2 | Max activities kept by nucleus filter |
| `--top_p` | 0.9 | Cumulative probability threshold for nucleus filter |
| `--p_min_end` | 0.3 | Minimum probability to treat an activity as a valid end |
| `--alpha` | 0.0 | Regularization weight (0 = plain SLA reward) |
| `--run_name` | auto | Identifier for the run; determines checkpoint folder |

Checkpoints and metrics land in:

```
data/training_models/<run_name>/
├── checkpoints/
│   ├── checkpoint_ep0010.pt
│   ├── best_model.pt
│   └── final_model.pt
├── episode_metrics.csv
├── update_metrics.csv
└── summary.json
```

### 3. Train the Resource-Only DM-DRL Agent

Trains a resource-only PPO agent where the activity is sampled from the empirical routing policy (same DM used at evaluation):

```bash
python src/train_resource_only.py \
    --log_path data/logs/AcademicCredentials/AcademicCredentials_train.csv \
    --episodes 300 \
    --max_cases 400 \
    --percentile 75 \
    --top_k 2 \
    --top_p 0.9 \
    --p_min_end 0.3 \
    --run_name AcademicCredentials_run01_resource_only
```

Accepts the same arguments as `train.py` except `--alpha` (always uses plain SLA reward).

### 4. Train All Registered Logs

Runs both variants (DRL-DRL and DM-DRL) for every log registered in `TRAINING_REGISTRY` inside `train_all.py`, in-process:

```bash
python src/train_all.py
```

Prints `[N/total]` progress with ETA between runs. Skips combinations whose checkpoints already exist.

### 5. Evaluate a Single (Log, Policy) Combination

Runs K simulations for one log + policy pair and exports metrics:

```bash
python src/run_single_evaluation.py
```

Edit the configuration at the top of the script (log path, policy name, checkpoint, K) before running.

### 6. Evaluate the Full (Log × Policy) Matrix

Runs all five evaluation policies across all registered logs and writes comparison CSVs:

```bash
python src/run_matrix_evaluation.py
```

Add `--resume` to re-derive metrics from previously-exported simulated logs without re-simulating. Combinations whose checkpoints don't exist are skipped and reported at the end.

**Output** (under `data/evaluation_results/`):

| File | Shape | Description |
|---|---|---|
| `runs_long.csv` | `log, policy, run_id, metric, value` | Tidy format for plotting |
| `runs_wide.csv` | one row per `(log, policy, run_id)` | One column per metric |
| `aggregated.csv` | one row per `(log, policy)` | `<metric>_mean` + `<metric>_ci95` — the paper table |

Per-policy `results.json` files are also written under `data/evaluation_results/<log>/<policy>/`.

### Evaluation Policies

| Name | Activity | Resource |
|---|---|---|
| RA-RR | Random uniform | Random |
| DM-RR | Empirical routing (Markov) | Random |
| DM-GR | Empirical routing (Markov) | Greedy min processing time |
| DM-DRL | Empirical routing (Markov) | `PPOResourceOnlyAgent` |
| DRL-DRL | `PPOAgent` activity head | `PPOAgent` resource head |
