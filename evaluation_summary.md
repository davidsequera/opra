# OPRA — Evaluation Briefing

Self-contained brief for handing to a fresh Claude conversation that will help write the **Evaluation** chapter of the thesis. Everything needed to reason about the design, the hypotheses, the baselines, the metrics, and the early results is in here.

---

## 1. What OPRA is

OPRA is a master's thesis project (Universidad de los Andes) that frames **business process optimization** as an MDP and solves it with **PPO** on top of a **SimPy-based Discrete Event Simulator**. The novelty vs. prior work: the agent jointly decides **both the next activity *and* the resource that will execute it** at every decision point. Prior work fixes control-flow and only optimizes resource allocation.

- **State** `s ∈ ℝ^d`, `d = 3|R| + 2|A| + 5`: per-resource utilization / assignment / queue pressure, per-activity pending demand, case-specific features (branching probs, last activity, trace length, SLA urgency), and temporal features (hour, day).
- **Action**: joint `(activity, resource)` pair. Activity masks use top-k / top-p nucleus filtering over learned routing probabilities; resource masks enforce skill constraints. Both via `-1e9` logit fill.
- **Reward**: SLA compliance — a case is a success if `cycle_time < T`, where `T` is a percentile of the original log's cycle-time distribution (training uses **p75** in all current runs). Implementation: `SLARewardFunction` (intermediate + terminal), with optional `RegularizedSLARewardFunction` that scales positive rewards by the empirical routing probability of the chosen activity (`alpha=0` disables; current runs use `alpha=0`).

---

## 2. Research hypotheses

The evaluation is organised around three hypotheses:

- **H1 — SLA Compliance.** A DRL agent with joint (activity, resource) control achieves higher SLA compliance than baselines that use random selection, empirical sampling, or heuristic resource choice.
- **H2 — Behavioural Realism.** A DRL policy can improve SLA compliance while remaining within an interpretable distance of realistic process executions (control-flow, temporal, resource and congestion distributions).
- **H3 — Incremental Benefit.** Progressively expanding the agent's scope of control — from random, to empirical-DM activity + random resource, to empirical-DM activity + learned resource, to fully joint learned — yields monotonic improvements in SLA compliance.

---

## 3. The five evaluation policies

Every policy is an `(ActivitySelector, ResourceSelector)` pair. Names use the convention `{activity}-{resource}`.

| Name    | Activity selection                                       | Resource selection                                                  | Checkpoint           |
|---------|----------------------------------------------------------|---------------------------------------------------------------------|----------------------|
| RA-RR   | Random uniform over feasible activities                  | Random over feasible                                                | —                    |
| DM-RR   | Sample from the empirical routing policy (the "DM")      | Random                                                              | —                    |
| DM-GR   | Sample from the empirical routing policy                 | Greedy: argmin estimated processing time (heuristic)                | —                    |
| DM-DRL  | Sample from the empirical routing policy                 | `PPOResourceOnlyAgent` (learned resource head only)                 | resource-only ckpt   |
| DRL-DRL | `PPOAgent` activity head                                 | `PPOAgent` resource head                                            | full ckpt            |

**Notes**

- The "Decision Model" (DM) is the empirical **1st-order Markov `ProbabilisticRoutingPolicy`** — *not* an LSTM. This is a deliberate simplification to avoid pulling in a separate next-activity predictor.
- **DM-GR** is the domain-knowledge heuristic baseline: it estimates expected processing time by averaging *N* draws of the existing `ProcessingTimePolicy.get_activity_duration(activity, resource)` interface and picks the lowest. No change to the policy interface — hexagonal boundary preserved. The averaging lives in `GreedyProcessingTimeResourceSelector._estimate_duration()` in `src/evaluation/selectors/resource.py`.
- **Stochastic activity** in DM-RR / DM-GR / DM-DRL avoids loop-collapse that argmax routing would cause on some processes.

### How each policy maps to a hypothesis

- **H1**: Compared against RA-RR, DM-RR, DM-GR. DM-GR is the critical anti-baseline — it pre-empts "you didn't try the obvious heuristic" reviewer pushback by showing that *naive* domain-knowledge optimization can actually *hurt* compliance.
- **H2**: Similarity metrics (NGD, AED, CED, RED, CWD, CAR, CTD) on all policies vs. the original log. Same metric family across the matrix; DM-* policies anchor what "realistic" looks like.
- **H3**: The ladder RA-RR → DM-RR → DM-DRL → DRL-DRL. DM-GR sits **off-ladder** at the same scope as DM-DRL — it isolates "is the win from learning, or from any non-random resource rule?".

---

## 4. Three event logs

| Log                  | Cases (train) | `max_cases` | top_k | top_p | p_min_end |
|----------------------|---------------|-------------|-------|-------|-----------|
| AcademicCredentials  | 398           | 400         | 2     | 0.9   | 0.3       |
| BPIC_2012            | 3030          | 3000        | 2     | 0.9   | 0.3       |
| BPIC_2017            | 7402          | 7400        | 3     | 0.9   | 0.1       |

Training: 300 episodes per (log, variant), SLA training percentile = **75**, lr = 3e-4, gamma = 0.99, seed = 42, `alpha = 0` (no regularization).

Run-name format (used everywhere — checkpoint paths, eval matrix lookup):

```
{LogName}_DDPS_p{percentile}_{episodes}_{max_cases}_tp{top_p*100}_tk{top_k}_pe{p_min_end*100}_a{alpha*100}_{variant}
```

Example: `AcademicCredentials_DDPS_p75_300_400_tp90_tk2_pe30_a0_full`.

---

## 5. Metrics

### Performance (per simulated run)

- **CR(Tₚ)** for p ∈ {95, 90, 75, 50}: fraction of cases with cycle time below the *p*-th percentile of the *original* log.
- **CIR(Tₚ)**: compliance improvement ratio vs. the original log baseline at the same threshold.
- **Cycle time**: mean, median, std, min, max, p75, p90, p95.
- **Resource utilization CV**: coefficient of variation of per-resource utilization (load-balancing proxy).

### Similarity (per simulated run, computed against the original log via `log-distance-measures`)

- **Control-flow**: NGD (N-gram distance).
- **Temporal**: AED (absolute), CED (circadian), RED (relative event distribution).
- **Resource**: CWD (circadian workforce distribution).
- **Congestion**: CAR (case arrival rate), CTD (cycle time distribution).

### Aggregation

K independent runs per (log, policy). Aggregation reports mean ± 95% CI via the t-distribution. Three CSV shapes emitted from the same in-memory record list:

- `runs_long.csv` — `log, policy, run_id, metric, value` (melt-friendly).
- `runs_wide.csv` — one row per `(log, policy, run_id)`, one column per metric.
- `aggregated.csv` — one row per `(log, policy)`, columns `<metric>_mean` / `<metric>_ci95`. **This is the table that goes in the paper.**

Per-policy `results.json` is also written for backward compatibility.

---

## 6. Architecture (one diagram, plain prose)

```
Event log (CSV)
  -> Initializer.build()                              [empirical or parametric]
  -> SimulationSetup  (frozen dataclass of policies)
  -> SimulatorEngine  (SimPy-based DDPS)
       - Standard mode: policies choose automatically, simulate() runs to completion
       - RL mode: pauses at each decision; agent calls apply_decision(activity, resource)
  -> BusinessProcessEnvironment (Gymnasium wrapper)
  -> (ActivitySelector, ResourceSelector) chosen via build_policy(name, ...)
  -> run_episode(env, simulator, act_sel, res_sel)
       - the single integration loop used by RA-RR, DM-RR, DM-GR, DM-DRL, DRL-DRL
  -> per-run simulated CSV + PolicyEvaluator metrics
  -> aggregate over K runs -> CSVs
```

**Hexagonal boundary.** Policy ABCs under `src/environment/simulator/policies/` (`RoutingPolicy`, `ProcessingTimePolicy`, `ArrivalPolicy`, `CalendarPolicy`, `WaitingTimePolicy`, `ResourceAllocationPolicy`) are stable contracts. Consumer-specific behaviour (like the greedy heuristic's expected-duration estimation) lives in the consumer, **never** added to the ABC. This keeps future ML-based policy implementations (e.g. LSTM routing, learned processing time) feasible without back-edits to every existing implementation.

**Selector abstraction.** Adding a new policy means implementing one or both selectors. No env, runner, or training changes needed.

---

## 6b. Evaluation policy definitions

Each evaluation policy is an `(ActivitySelector, ResourceSelector)` pair. The selector layer lives in `src/evaluation/selectors/`. At every decision point in the simulation, `run_episode` queries the activity selector for an activity, then the resource selector for a resource (conditioned on that activity), then calls `env.step((activity, resource))`. Activity and resource masks (top-k / top-p nucleus filtering and skill constraints, respectively) are applied uniformly across all policies — no policy is allowed to pick an infeasible action.

### RA-RR — Random Activity, Random Resource

- **Activity:** uniform random over the activities allowed by the activity mask (i.e. the top-k / top-p filtered set of plausible next activities for the case).
- **Resource:** uniform random over the resources allowed by the resource mask (i.e. resources whose `skills` include the chosen activity).
- **Checkpoint:** none.
- **Role:** absolute lower bound. Establishes the worst-case behaviour of the system when *no* domain knowledge or learning is applied; the only structure imposed is feasibility.

### DM-RR — Decision-Model Activity, Random Resource

- **Activity:** sampled stochastically from the empirical `ProbabilisticRoutingPolicy` (1st-order Markov over activity transitions, conditioned on the last activity of the case). Sampling — not argmax — is deliberate, because argmax on some processes collapses into loops between the two most frequent activities.
- **Resource:** uniform random over feasible resources.
- **Checkpoint:** none.
- **Role:** isolates the contribution of *realistic control-flow alone*. By construction it should match the empirical process's activity statistics; the only randomness on top of the original log is the resource assignment.

### DM-GR — Decision-Model Activity, Greedy Resource

- **Activity:** same stochastic sampling from the empirical routing policy as DM-RR.
- **Resource:** greedy on estimated processing time. The `GreedyProcessingTimeResourceSelector` calls `processing_time_policy.get_activity_duration(activity, resource)` *N* times for each feasible resource, averages, and picks the resource with the lowest estimated duration. The policy ABC is **not** modified — the averaging happens in the selector.
- **Checkpoint:** none.
- **Role:** the domain-knowledge heuristic baseline. Tests whether a sensible non-learned rule ("pick the historically-fast resource") is enough to improve SLA compliance. Empirically it is *not*: it saturates the top tail (CR T95) but collapses the cycle-time distribution into a narrow band that fails the median thresholds, because it ignores congestion and overloads fast resources.

### DM-DRL — Decision-Model Activity, DRL Resource

- **Activity:** same stochastic sampling from the empirical routing policy.
- **Resource:** `PPOResourceOnlyAgent`. A PPO agent with the same backbone as the full agent minus the activity head. At each decision step it receives the state and the externally-supplied activity (embedded), and outputs a resource distribution masked to feasible resources.
- **Checkpoint:** `*_resource_only`, trained by `src/train_resource_only.py`. During training, the activity at each decision is drawn from the same `EmpiricalDMActivitySelector` used at evaluation, so the training and inference activity distributions match.
- **Role:** isolates the value of *learning* the resource policy under a fixed control-flow distribution. Direct comparison with DM-GR answers "is the win from learning, or from any non-random resource rule?".

### DRL-DRL — DRL Activity, DRL Resource (the thesis novelty)

- **Activity:** `PPOAgent` activity head — a learned policy over activities, masked by the same top-k / top-p nucleus filter used by every other policy. The agent is therefore constrained to plausible next activities according to the empirical routing distribution, but is free to deviate from the *probabilities* within that set.
- **Resource:** `PPOAgent` resource head, conditioned on an embedding of the chosen activity. Hierarchical action selection: activity first, then resource.
- **Checkpoint:** `*_full`, trained by `src/train.py`.
- **Role:** the full proposal. By jointly choosing activity and resource, the agent can re-order work and bias the case toward faster paths *and* faster resources simultaneously. This is the only policy in the matrix that can change *which activities happen*, not just who executes them.

### Summary

| Policy   | Activity                       | Resource                              | Learning | Checkpoint              |
|----------|--------------------------------|---------------------------------------|----------|-------------------------|
| RA-RR    | Random (uniform feasible)      | Random (uniform feasible)             | none     | —                       |
| DM-RR    | Stochastic empirical routing   | Random                                | none     | —                       |
| DM-GR    | Stochastic empirical routing   | Greedy on est. processing time        | none     | —                       |
| DM-DRL   | Stochastic empirical routing   | Learned (PPO, resource-only)          | resource | `*_resource_only`       |
| DRL-DRL  | Learned (PPO activity head)    | Learned (PPO resource head)           | joint    | `*_full`                |

---

## 7. Current state of training and evaluation

- **AcademicCredentials**: ✅ both checkpoints trained (full and resource-only). Matrix evaluation completed for K = 10.
- **BPIC_2012**: ✅ both checkpoints trained. Matrix evaluation completed for K = 10.
- **BPIC_2017**: ✅ both checkpoints trained. Matrix evaluation completed for K = 10.

The full (log × policy) matrix is complete. All results below are K = 10 runs, training SLA percentile = p75.

---

### AcademicCredentials results (K = 10, training p75)

| policy  | CR T95 | CR T90 | CR T75 | CR T50 | avg cycle time | NGD  |
|---------|--------|--------|--------|--------|----------------|------|
| RA-RR   | 92.9%  | 86.8%  | 66.3%  | 28.6%  | 894k s         | 0.37 |
| DM-RR   | 94.4%  | 89.3%  | 71.0%  | 35.4%  | 772k s         | 0.26 |
| DM-GR   | 100.0% | 99.7%  | 63.5%  | 25.2%  | 720k s         | 0.27 |
| DM-DRL  | 98.4%  | 97.1%  | 90.2%  | 66.0%  | 346k s         | 0.27 |
| DRL-DRL | 100.0% | 99.7%  | 99.4%  | 94.0%  | 60k s          | 0.46 |

**CIR — Compliance Improvement Ratio vs. the original log** (positive = lift, negative = the policy is *worse* than the historical process at that threshold).

| policy  | CIR T95  | CIR T90  | CIR T75  | CIR T50  |
|---------|----------|----------|----------|----------|
| RA-RR   | −2.17%   | −3.55%   | −11.41%  | −42.76%  |
| DM-RR   | −0.56%   | −0.70%   | −5.13%   | −29.15%  |
| DM-GR   | +5.29%   | +10.81%  | −15.23%  | −49.70%  |
| DM-DRL  | +3.62%   | +7.96%   | +20.44%  | +31.96%  |
| DRL-DRL | +5.24%   | +10.84%  | +32.79%  | +88.04%  |

---

### BPIC_2012 results (K = 10, training p75)

| policy  | CR T95 | CR T90 | CR T75 | CR T50 | avg cycle time | NGD  |
|---------|--------|--------|--------|--------|----------------|------|
| RA-RR   | 95.1%  | 91.9%  | 79.0%  | 29.7%  | 614k s         | 0.38 |
| DM-RR   | 84.9%  | 79.2%  | 65.0%  | 35.0%  | 1150k s        | 0.36 |
| DM-GR   | 84.4%  | 77.3%  | 61.6%  | 33.7%  | 948k s         | 0.36 |
| DM-DRL  | 84.8%  | 79.0%  | 65.6%  | 35.3%  | 1093k s        | 0.36 |
| DRL-DRL | 100.0% | 100.0% | 99.9%  | 97.2%  | 9k s           | 0.75 |

| policy  | CIR T95  | CIR T90  | CIR T75  | CIR T50  |
|---------|----------|----------|----------|----------|
| RA-RR   | +0.14%   | +2.06%   | +5.32%   | −40.57%  |
| DM-RR   | −10.58%  | −12.04%  | −13.32%  | −30.07%  |
| DM-GR   | −11.13%  | −14.11%  | −17.83%  | −32.65%  |
| DM-DRL  | −10.71%  | −12.27%  | −12.54%  | −29.35%  |
| DRL-DRL | +5.28%   | +11.10%  | +33.17%  | +94.36%  |

---

### BPIC_2017 results (K = 10, training p75)

| policy  | CR T95 | CR T90 | CR T75 | CR T50 | avg cycle time | NGD  |
|---------|--------|--------|--------|--------|----------------|------|
| RA-RR   | 99.5%  | 98.2%  | 91.9%  | 72.0%  | 614k s         | 0.42 |
| DM-RR   | 96.3%  | 92.1%  | 80.1%  | 56.8%  | 868k s         | 0.18 |
| DM-GR   | 87.0%  | 79.2%  | 62.4%  | 39.4%  | 1346k s        | 0.18 |
| DM-DRL  | 97.4%  | 94.2%  | 83.8%  | 62.1%  | 771k s         | 0.18 |
| DRL-DRL | 100.0% | 100.0% | 100.0% | 99.6%  | 112k s         | 0.68 |

| policy  | CIR T95  | CIR T90  | CIR T75  | CIR T50  |
|---------|----------|----------|----------|----------|
| RA-RR   | +4.71%   | +9.17%   | +22.59%  | +43.94%  |
| DM-RR   | +1.41%   | +2.40%   | +6.78%   | +13.67%  |
| DM-GR   | −8.41%   | −12.01%  | −16.78%  | −21.29%  |
| DM-DRL  | +2.54%   | +4.66%   | +11.68%  | +24.27%  |
| DRL-DRL | +5.28%   | +11.12%  | +33.32%  | +99.24%  |

---

### Cross-log observations

**H1 — SLA Compliance (DRL-DRL vs. baselines):**
- DRL-DRL achieves near-perfect CR at all thresholds across all three logs (T95 ≈ 100%, T50 ≥ 94%). The gap is most dramatic at T50: DRL-DRL reaches 94% (AC), 97% (BPIC_2012), and 99.6% (BPIC_2017) while the next-best policy rarely exceeds 66%.
- H1 is strongly supported across all three logs.

**H3 — Incremental Benefit (ladder RA-RR → DM-RR → DM-DRL → DRL-DRL):**
- The ladder holds on AcademicCredentials and BPIC_2017. On BPIC_2012 the DM-* family (DM-RR, DM-GR, DM-DRL) performs *worse* than RA-RR at every threshold — the empirical Markov routing produces longer cycle times than random activity selection on this log, suggesting the 1st-order Markov policy is poorly calibrated for BPIC_2012's structure. DRL-DRL still dominates. The rungs between DM-RR and DM-DRL are still monotone within the DM-* family on BPIC_2012.
- RA-RR on BPIC_2017 outperforms DM-RR at every threshold — the same Markov-routing pathology is present to a lesser degree.

**DM-GR — consistent negative result:**
- DM-GR is the worst or near-worst non-random policy at T50/T75 on every log. On BPIC_2017 it is the worst policy overall. The congestion-collapse pattern (fast-resource overload → inflated cycle-time floor) is consistent. Strong thesis point: the naïve domain-knowledge heuristic reliably fails.

**H2 — Behavioural Realism:**
- DRL-DRL NGD ranges from 0.46 (AC) to 0.75 (BPIC_2012) vs. 0.26–0.42 for DM-* policies. The behavioural deviation from the original process is largest on BPIC_2012 where DRL-DRL also achieves the most extreme cycle-time compression (~9k s avg vs. 614k s for RA-RR). The realism–compliance trade-off is sharpest on the largest log.
- DM-* NGD values are tightly clustered (~0.18 on BPIC_2017, ~0.36 on BPIC_2012, ~0.26 on AC), confirming they preserve control-flow better than DRL-DRL as expected.

**BPIC_2012 anomaly (DM routing underperforms random):**
- All DM-* policies have CIR < 0 at every threshold on BPIC_2012 (except DM-RR at T50 which is close to zero). RA-RR meanwhile has positive CIR at T95/T90/T75. This is likely because the empirical 1st-order Markov routing on BPIC_2012 systematically routes cases through slow paths. Worth a paragraph in the thesis under "Threats to Validity" or "Limitations of the Decision Model".

---

## 8. Things a fresh Claude should keep in mind when drafting the chapter

1. **Activity selection in DM-* is stochastic, not argmax.** The text should say "sampled from the empirical routing distribution", not "the most likely activity". Argmax would collapse processes into loops on some logs.
2. **DM is empirical Markov, not LSTM.** Avoid implying a learned next-activity model is in scope.
3. **DM-GR's value is the *negative* result.** Don't write it up as "another strong baseline". Frame it as: "a sensible domain-knowledge heuristic is shown to be insufficient — congestion-awareness matters."
4. **Training SLA threshold = p75 for all logs.** Earlier drafts used T90/T80; we standardised on p75.
5. **Reward is `SLARewardFunction` with `alpha = 0`** in the reported runs. The regularised variant exists but isn't currently active.
6. **Realism is reported per dimension** (control-flow / temporal / resource / congestion), not as a single aggregate score. Don't average the similarity metrics into one number.
7. **K = 10 runs per (log, policy), aggregated as mean ± 95% CI (t-distribution).** Every metric in the paper table has a CI.
8. **The matrix is `log × policy`**, not `log × policy × hyperparameter`. Hyperparameter ablations (e.g. `alpha`, `top_k`, training percentile) are *separate* studies and only AC_CRE has the full matrix today.
9. **The thesis novelty is the joint (activity, resource) action.** When discussing H3, emphasise that DM-DRL → DRL-DRL is the step where the action space expands; the earlier rungs of the ladder differ only in how the resource is chosen.

---

## 9. Suggested chapter outline

1. **Problem statement.** Recap the MDP framing and the SLA-compliance objective. Anchor the three hypotheses.
2. **Experimental setup.** Logs, training hyperparameters, mask parameters, K, SLA thresholds, seeds.
3. **Baselines.** The five policies in the table above. One paragraph per policy explaining *what role it plays in which hypothesis* (use the mapping in §3).
4. **Metrics.** Performance + similarity, why CIR is reported alongside CR, why CV proxies load-balancing, why per-dimension similarity rather than aggregate.
5. **Results.**
   - 5a — H1: cross-policy compliance table at all thresholds, per log.
   - 5b — H3: the ladder, with cycle-time distribution plots showing the collapse from RA-RR to DRL-DRL.
   - 5c — H2: similarity heatmap or radar chart per policy per log. Highlight DRL-DRL vs. DM-DRL gap on NGD.
   - 5d — DM-GR analysis: the cycle-time-band finding, framed as "why congestion-awareness matters".
6. **Discussion.** When does DRL win, when does it not, what does the realism trade-off cost. Limitations: DM is Markov-1, no learned next-activity model; SLA is a single scalar.
7. **Threats to validity.** Stochasticity (mitigated by K=10 + CI), choice of training percentile, mask hyperparameters chosen per log.

---

## 10. Where the code lives (for cross-referencing while drafting)

- Reward: `src/environment/core/reward.py` — `SLARewardFunction`, `RegularizedSLARewardFunction`, `BinaryRewardFunction`.
- Env: `src/environment/core/env.py` — `BusinessProcessEnvironment`.
- Agents: `src/agent/JointAgent/` (full DRL-DRL), `src/agent/ResourcesOnlyAgent/` (DM-DRL).
- Selectors: `src/evaluation/selectors/{activity,resource,factory}.py`.
- Runner: `src/evaluation/runner.py`.
- Single experiment: `src/evaluation/experiment.py::evaluate_policy_on_log`.
- Matrix runner: `src/run_matrix_evaluation.py`.
- Metrics: `src/evaluation/metrics/` (performance, similarity, aggregation, `PolicyEvaluator`).
- Training: `src/train.py` (full), `src/train_resource_only.py` (resource-only), `src/train_all.py` (orchestrator).
