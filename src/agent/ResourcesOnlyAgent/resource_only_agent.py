"""Resource-only PPO agent.

Mirror of PPOAgent for training a resource-allocation-only policy: the activity
at each decision is supplied externally (e.g. sampled from the empirical routing
policy), and the agent only learns the resource head + value + backbone.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .resource_only_policy import PPOResourceOnlyPolicy


class PPOResourceOnlyAgent:
    def __init__(
        self,
        state_dim: int,
        num_activities: int,
        num_resources: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        K_epochs: int = 4,
        eps_clip: float = 0.2,
        device: str = "cpu",
        activities_embedding_dim: int = 32,
    ):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = PPOResourceOnlyPolicy(
            state_dim, num_activities, num_resources,
            activities_embedding_dim=activities_embedding_dim,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOResourceOnlyPolicy(
            state_dim, num_activities, num_resources,
            activities_embedding_dim=activities_embedding_dim,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.buffer = ResourceOnlyRolloutBuffer()

    def select_action(self, state, activity_idx: int, resource_mask, deterministic: bool = False) -> int:
        self.policy_old.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            res_mask_t = torch.FloatTensor(resource_mask).unsqueeze(0).to(self.device)
            act_t = torch.tensor([activity_idx], dtype=torch.long, device=self.device)

            res_logits = self.policy_old.get_resource_logits(state_t, act_t)
            res_logits = res_logits.masked_fill(res_mask_t == 0, -1e9)
            res_dist = Categorical(logits=res_logits)

            if deterministic:
                resource_idx = torch.argmax(res_logits, dim=-1)
            else:
                resource_idx = res_dist.sample()

            log_prob = res_dist.log_prob(resource_idx)

            features = self.policy_old.backbone(state_t)
            value = self.policy_old.value_head(features).squeeze(-1)

        self.buffer.states.append(state_t)
        self.buffer.activities.append(act_t)
        self.buffer.resources.append(resource_idx)
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(value)
        self.buffer.resource_masks.append(res_mask_t)

        return resource_idx.item()

    def update(self):
        if not self.buffer.rewards:
            return None

        # Monte-Carlo returns
        rewards = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        rewards_std = rewards.std()
        if rewards_std > 0.1:
            rewards = (rewards - rewards.mean()) / (rewards_std + 1e-7)
        else:
            print("Warning: Low reward variance, skipping normalization to avoid amplifying noise.")
            rewards = rewards - rewards.mean()

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_activities = torch.squeeze(torch.stack(self.buffer.activities, dim=0)).detach().to(self.device)
        old_resources = torch.squeeze(torch.stack(self.buffer.resources, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        old_resource_masks = torch.squeeze(torch.stack(self.buffer.resource_masks, dim=0)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss_val = 0.0

        for _ in range(self.K_epochs):
            logprobs, entropy, state_values = self.policy.evaluate(
                old_states, old_activities, old_resources, old_resource_masks,
            )
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_bonus = 0.01 * entropy.mean()

            loss = policy_loss + value_loss - entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_loss_val += loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / self.K_epochs,
            "value_loss": total_value_loss / self.K_epochs,
            "entropy": total_entropy / self.K_epochs,
            "total_loss": total_loss_val / self.K_epochs,
        }


class ResourceOnlyRolloutBuffer:
    def __init__(self):
        self.states = []
        self.activities = []
        self.resources = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.resource_masks = []

    def clear(self):
        for lst in (
            self.states, self.activities, self.resources, self.logprobs,
            self.rewards, self.state_values, self.is_terminals, self.resource_masks,
        ):
            del lst[:]
