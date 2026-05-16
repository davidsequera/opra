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
├── .git/
├── .gitignore
├── CLAUDE.md
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── logs/
│   └── simulated_logs/
├── docs/
│   ├── architecture.dsl
│   ├── environment.md
│   └── Thesis_Proposal.pdf
└── src/
    ├── agent/
    │   └── agent.py
    ├── environment/
    │   ├── core/
    │   │   ├── env.py
    │   │   ├── mask.py
    │   │   └── reward.py
    │   ├── simulator/
    │   │   ├── adapters/
    │   │   ├── core/
    │   │   ├── implementations/
    │   │   └── policies/
    │   └── entities/
    │       ├── Activity.py
    │       ├── Case.py
    │       ├── Events.py
    │       └── Resource.py
    ├── initializer/
    │   ├── implementations/
    │   │   ├── DDPSInitializer.py
    │   │   └── ParametricInitializer.py
    │   └── Initializer.py
    ├── metrics/
    │   ├── __init__.py
    │   ├── training_metrics.py
    │   └── evaluation_metrics.py
    ├── evaluate.py
    ├── evaluate_policy.py
    ├── main.py
    ├── simulate.py
    └── train.py
```

## Getting Started

### Prerequisites
*   Python 3.8+

### Installation
1.  **Clone the repository:**
    ```bash
    git clone  https://github.com/AdaptiveBProcess/Optimal-Predictor-of-Resources-and-Activities.git
    cd opra
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running Simulations

#### 1. Basic Discrete-Event Simulation
To run a basic DDPS and generate a simulated event log:
```bash
python src/simulate.py
```
This will output a simulated event log to `data/simulated_logs/PurchasingExample/PurchasingExample.csv`.

#### 2. Reinforcement Learning Experiment
To run a single-episode RL simulation (quick test):
```bash
python src/main.py
```

#### 3. Training a Policy
To train a PPO agent over multiple episodes with metric tracking and checkpointing:
```bash
python src/train.py \
    --log_path data/logs/LoanApp/LoanApp.csv \
    --episodes 200 \
    --max_cases 50 \
    --percentile 90 \
    --lr 3e-4 \
    --save_every 10 \
    --run_name experiment_01
```


Key arguments:
- `--episodes`: Number of training episodes (default: 100)
- `--max_cases`: Cases simulated per episode (default: 20)
- `--percentile`: SLA threshold percentile (default: 95)
- `--top_p`: Nucleus filtering threshold for activity masking (default: 0.9)
- `--update_every`: PPO update frequency in episodes (default: 1)

Output structure:
```
data/training_models/experiment_01/
├── checkpoints/
│   ├── checkpoint_ep0010.pt
│   ├── checkpoint_ep0020.pt
│   ├── best_model.pt
│   └── final_model.pt
├── episode_metrics.csv
├── update_metrics.csv
└── summary.json
```

#### 4. Evaluating a Trained Policy
To evaluate a trained checkpoint over K independent runs and compute the full metrics suite (performance + similarity):
```bash
python src/evaluate_policy.py \
    --checkpoint data/training_models/experiment_01/checkpoints/best_model.pt \
    --log_path data/logs/LoanApp/LoanApp.csv \
    --K 10 \
    --policy_name DRL-AR \
    --log_name LoanApp
```

This runs K greedy (deterministic) simulations, exports each as a CSV with absolute timestamps, and computes:
- **Performance**: Compliance rates at T95/T90/T75/T50, compliance improvement ratios, cycle time statistics, resource utilization CV
- **Similarity** (requires `log-distance-measures`): NGD, AED, CED, RED, CWD, CAR, CTD

Results are aggregated as mean ± 95% CI and saved to `data/evaluation_results/`.
