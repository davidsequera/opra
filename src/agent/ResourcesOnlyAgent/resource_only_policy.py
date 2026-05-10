"""Resource-only PPO policy network.

Same backbone + activity_embedding + resource_head + value_head as the full
PPOPolicy, but with NO activity head — the activity is supplied externally
(typically sampled from the empirical routing policy at each decision).

This lets the agent learn to optimize resource allocation under a fixed
control-flow distribution (the DM-DRL baseline of the thesis).
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOResourceOnlyPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_activities: int,
        num_resources: int,
        activities_embedding_dim: int = 32,
    ):
        super().__init__()
        hidden = 256

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.activity_embedding = nn.Embedding(num_activities, activities_embedding_dim)
        self.resource_head = nn.Linear(hidden + activities_embedding_dim, num_resources)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state, activity, resource_mask=None):
        features = self.backbone(state)
        act_emb = self.activity_embedding(activity)

        res_input = torch.cat([features, act_emb], dim=-1)
        resource_logits = self.resource_head(res_input)
        if resource_mask is not None:
            resource_logits = resource_logits.masked_fill(resource_mask == 0, -1e9)

        resource_dist = Categorical(logits=resource_logits)
        resource = resource_dist.sample()
        log_prob = resource_dist.log_prob(resource)
        value = self.value_head(features).squeeze(-1)

        return resource, log_prob, value

    def evaluate(self, state, activity, resource, resource_mask=None):
        features = self.backbone(state)
        act_emb = self.activity_embedding(activity)

        res_input = torch.cat([features, act_emb], dim=-1)
        resource_logits = self.resource_head(res_input)
        if resource_mask is not None:
            resource_logits = resource_logits.masked_fill(resource_mask == 0, -1e9)

        resource_dist = Categorical(logits=resource_logits)
        log_prob = resource_dist.log_prob(resource)
        entropy = resource_dist.entropy()
        value = self.value_head(features).squeeze(-1)

        return log_prob, entropy, value

    def get_resource_logits(self, state, activity):
        """Same signature as PPOPolicy.get_resource_logits — used by DRLResourceSelector."""
        features = self.backbone(state)
        act_emb = self.activity_embedding(activity)
        res_input = torch.cat([features, act_emb], dim=-1)
        return self.resource_head(res_input)
