import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOPolicy(nn.Module):

    def __init__(self, state_dim, num_activities, num_resources, activities_embedding_dim=32):
        super().__init__()

        hidden = 256

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Heads
        self.activity_head = nn.Linear(hidden, num_activities)

        # Activity embedding for conditioning
        self.activity_embedding = nn.Embedding(num_activities, activities_embedding_dim)

        self.resource_head = nn.Linear(hidden + activities_embedding_dim, num_resources)

        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state, activity_mask=None, resource_mask=None):

        features = self.backbone(state)

        # --- Activity ---
        activity_logits = self.activity_head(features)

        if activity_mask is not None:
            activity_logits = activity_logits.masked_fill(
                activity_mask == 0, -1e9
            )

        activity_dist = Categorical(logits=activity_logits)
        activity = activity_dist.sample()

        # --- Resource (conditional) ---
        act_emb = self.activity_embedding(activity)

        res_input = torch.cat([features, act_emb], dim=-1)
        resource_logits = self.resource_head(res_input)

        if resource_mask is not None:
            resource_logits = resource_logits.masked_fill(
                resource_mask == 0, -1e9
            )

        resource_dist = Categorical(logits=resource_logits)
        resource = resource_dist.sample()

        # Log prob
        log_prob = (
            activity_dist.log_prob(activity)
            + resource_dist.log_prob(resource)
        )

        value = self.value_head(features).squeeze(-1)

        return activity, resource, log_prob, value

    def evaluate(self, state, activity, resource,
                 activity_mask=None, resource_mask=None):

        features = self.backbone(state)

        activity_logits = self.activity_head(features)
        if activity_mask is not None:
            activity_logits = activity_logits.masked_fill(
                activity_mask == 0, -1e9
            )

        activity_dist = Categorical(logits=activity_logits)

        act_emb = self.activity_embedding(activity)
        res_input = torch.cat([features, act_emb], dim=-1)

        resource_logits = self.resource_head(res_input)
        if resource_mask is not None:
            resource_logits = resource_logits.masked_fill(
                resource_mask == 0, -1e9
            )

        resource_dist = Categorical(logits=resource_logits)

        log_prob = (
            activity_dist.log_prob(activity)
            + resource_dist.log_prob(resource)
        )

        entropy = (
            activity_dist.entropy()
            + resource_dist.entropy()
        )

        value = self.value_head(features).squeeze(-1)

        return log_prob, entropy, value

    def get_activity_logits(self, state):
        features = self.backbone(state)
        return self.activity_head(features)

    def get_resource_logits(self, state, activity):
        features = self.backbone(state)
        act_emb = self.activity_embedding(activity)
        res_input = torch.cat([features, act_emb], dim=-1)
        return self.resource_head(res_input)