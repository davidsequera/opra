import math
import gymnasium as gym
import numpy as np
import pandas as pd

from environment.simulator.core.engine import SimulatorEngine
from environment.core.reward import (
    RewardFunction,
    SLARewardFunction,
    RegularizedSLARewardFunction,
    CaseRewardContext,
)
from environment.core.mask import (
    ActivityMaskFunction,
    ResourceMaskFunction,
    NucleusMaskFunction,
    SkillBasedMaskFunction,
    ActivityMaskContext,
    ResourceMaskContext,
)

class BusinessProcessEnvironment(gym.Env):

    def __init__(self, simulator: "SimulatorEngine", sla_threshold, max_cases,
                 reward_function: RewardFunction = None,
                 activity_mask_function: ActivityMaskFunction = None,
                 resource_mask_function: ResourceMaskFunction = None):
        super().__init__()

        self.simulator = simulator
        self.sla_threshold = sla_threshold
        self.max_cases = max_cases
        self.reward_function = reward_function or SLARewardFunction()
        self.activity_mask_function = activity_mask_function or NucleusMaskFunction()
        self.resource_mask_function = resource_mask_function or SkillBasedMaskFunction()

        self.action_space = gym.spaces.MultiDiscrete(
            [simulator.num_activities, simulator.num_resources]
        )

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
 
    def set_agent_activity_probs(self, probs: list) -> None:
        """Call once before env.step() with the agent's masked activity distribution (π)."""
        self._pending_agent_activity_probs = probs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.simulator.reset(max_cases=self.max_cases)
        self.completed_cases = 0
        self._case_log_prob_sum: dict = {}
        self._case_num_decisions: dict = {}
        self._case_agent_probs: dict = {}
        self._case_routing_probs: dict = {}
        self._pending_agent_activity_probs = None

        state, _ = self._advance_to_next_decision()
        return state, {}

    def step(self, action):
        act_idx, res_idx = action

        # Map indices to actual activity and resource
        # Note: In RL mode, act_idx might represent 'END' if defined in all_activities
        activity_type = self.simulator.all_activities[act_idx]
        resource = self.simulator.all_resources[res_idx]

        # Capture the case receiving this decision before advancing
        current_case = self.simulator.get_case_needing_decision()

        # Capture routing distribution before applying the decision
        probs_dict = self.simulator.setup.routing_policy.get_activity_probabilities(current_case)
        chosen_prob = float(probs_dict.get(activity_type, 0.0))
        routing_probs = [float(probs_dict.get(act, 0.0)) for act in self.simulator.all_activities]
        agent_probs = self._pending_agent_activity_probs
        self._pending_agent_activity_probs = None

        case_id = id(current_case)
        self._case_log_prob_sum[case_id] = self._case_log_prob_sum.get(case_id, 0.0) + math.log(chosen_prob + 1e-8)
        self._case_num_decisions[case_id] = self._case_num_decisions.get(case_id, 0) + 1
        self._case_agent_probs[case_id] = agent_probs
        self._case_routing_probs[case_id] = routing_probs

        # Apply decision to simulator (it will resume the process_case)
        self.simulator.apply_decision(activity_type, resource)

        state, completed = self._advance_to_next_decision()

        now = self.simulator.state()["internal_time"]
        completed_ids = {id(c) for c in completed}

        reward = 0.0

        # Intermediate reward for the decided case if it did not complete
        if current_case is not None and id(current_case) not in completed_ids:
            elapsed = now - current_case.start_time
            ctx = CaseRewardContext(
                cycle_time=elapsed,
                sla_threshold=self.sla_threshold,
                num_events=len(current_case.activity_history),
                start_time=current_case.start_time,
                end_time=now,
                is_completed=False,
                chosen_activity_prob=chosen_prob,
                activity_probs_agent=agent_probs,
                activity_probs_routing=routing_probs,
                mean_routing_prob=1.0,
            )
            reward += self.reward_function.compute(ctx)

        # Terminal reward for all completed cases
        for case in completed:
            cid = id(case)
            log_sum = self._case_log_prob_sum.pop(cid, 0.0)
            n = self._case_num_decisions.pop(cid, 1)
            mean_routing_prob = math.exp(log_sum / max(n, 1))
            ctx = CaseRewardContext(
                cycle_time=case.cycle_time,
                sla_threshold=self.sla_threshold,
                num_events=len(case.activity_history),
                start_time=case.start_time,
                end_time=case.end_time,
                is_completed=True,
                chosen_activity_prob=1.0,
                activity_probs_agent=self._case_agent_probs.pop(cid, None),
                activity_probs_routing=self._case_routing_probs.pop(cid, None),
                mean_routing_prob=mean_routing_prob,
            )
            reward += self.reward_function.compute(ctx)
            self.completed_cases += 1

        terminated = self.completed_cases >= self.max_cases
        truncated = False

        return state, reward, terminated, truncated, {}


    def _advance_to_next_decision(self):
        completed_cases = self.simulator.run_until_decision()

        state = self._compute_state()

        return state, completed_cases

    def vectorize_state(self):
        """
        Builds a normalized state vector in [0, 1] with three blocks.

        Block A — Global Process Snapshot  (3*R + A features)
        --------------------------------------------------------
        Per resource (ordered by simulator.all_resources):
          · utilization       : count / capacity
          · assignment_enc    : normalized index of activity being executed (0 = idle)
          · queue_pressure    : queue length / 10, clamped to [0, 1]
        Per activity (ordered by simulator.all_activities):
          · pending_count     : cases queued for this activity / 10, clamped to [0, 1]

        Block B — Case-Specific Features  (A + 3 features)
        --------------------------------------------------------
          · branching_probs   : P(next_activity | current_position), one float per activity
          · last_activity_enc : (index_of_last_activity + 1) / num_activities (0 = new case)
          · trace_length_norm : len(history) / 20, clamped to [0, 1]
          · sla_urgency       : elapsed_time / sla_threshold, clamped to [0, 1]

        Block C — Temporal Features  (2 features)
        --------------------------------------------------------
          · hour_of_day  : (now % 86400) / 86400
          · day_of_week  : ((now // 86400) % 7) / 6
        """
        sim_state = self.simulator.state()
        now = sim_state["internal_time"]
        num_act = self.simulator.num_activities

        # ── Block A: Global Process Snapshot ────────────────────────────────────
        res_features = []
        for res in self.simulator.all_resources:
            occ = sim_state["resource_occupancy"].get(
                res.id, {"in_use": 0, "capacity": 1, "waiting": 0}
            )
            capacity = max(occ["capacity"], 1)

            utilization = occ["in_use"] / capacity

            current_act = sim_state["resource_current_activity"].get(res.id)
            if current_act is not None and current_act in self.simulator.all_activities:
                assignment_enc = self.simulator.all_activities.index(current_act) / num_act
            else:
                assignment_enc = 0.0

            queue_pressure = min(occ["waiting"] / 10.0, 1.0)

            res_features.extend([utilization, assignment_enc, queue_pressure])

        # Per-activity: cases in waiting_requests (resource-queue) for each activity
        act_pending = {}
        for _, (_, act) in self.simulator.waiting_requests.items():
            act_pending[act] = act_pending.get(act, 0) + 1

        act_features = [
            min(act_pending.get(act, 0) / 10.0, 1.0)
            for act in self.simulator.all_activities
        ]

        # ── Block B: Case-Specific Features ─────────────────────────────────────
        case = self.simulator.get_case_needing_decision()

        if case is None:
            # Simulation ended or between decisions — safe zero vector
            case_features = [0.0] * (num_act + 3)
        else:
            history = case.activity_history
            current_activity = history[-1] if history else None

            # Last activity encoded as (idx+1)/num_act so that 0 unambiguously means "new case"
            if history and current_activity in self.simulator.all_activities:
                last_act_enc = (self.simulator.all_activities.index(current_activity) + 1) / num_act
            else:
                last_act_enc = 0.0

            trace_length_norm = min(len(history) / 20.0, 1.0)
            sla_urgency = min((now - case.start_time) / max(self.sla_threshold, 1.0), 1.0)

            # Branching probabilities for current routing position
            probs_dict = self.simulator.setup.routing_policy.get_activity_probabilities(case)

            branching_probs = [
                float(probs_dict.get(act, 0.0))
                for act in self.simulator.all_activities
            ]

            case_features = branching_probs + [last_act_enc, trace_length_norm, sla_urgency]

        # ── Block C: Temporal Features ───────────────────────────────────────────
        time_features = [
            (now % 86400) / 86400.0,           # hour_of_day
            ((now // 86400) % 7) / 6.0,        # day_of_week
        ]

        return np.clip(
            np.array(res_features + act_features + case_features + time_features, dtype=np.float32),
            0.0, 1.0,
        )

    @property
    def state_dim(self):
        # Block A: 3 per resource + 1 per activity
        # Block B: num_activities (branching probs) + 3 (last_act, trace_len, sla_urgency)
        # Block C: 2 (hour_of_day, day_of_week)
        return 3 * self.simulator.num_resources + 2 * self.simulator.num_activities + 5

    def _compute_state(self):
        return self.vectorize_state()

    def get_activity_mask(self, case):
        """Returns a binary mask for feasible next activities."""
        probs_dict = self.simulator.setup.routing_policy.get_activity_probabilities(case)
        ctx = ActivityMaskContext(
            probabilities=probs_dict,
            all_activities=self.simulator.all_activities,
        )
        return self.activity_mask_function.compute(ctx)

    def get_resource_mask(self, activity_name, case=None):
        """Returns a binary mask of feasible resources for a given activity."""
        ctx = ResourceMaskContext(
            activity_name=activity_name,
            all_resources=self.simulator.all_resources,
        )
        return self.resource_mask_function.compute(ctx)
