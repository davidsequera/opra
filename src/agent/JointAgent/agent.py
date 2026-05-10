import torch
from .policy import PPOPolicy
import torch.nn as nn
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, state_dim, num_activities, num_resources, 
                 lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, device="cpu", activities_embedding_dim=32):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = PPOPolicy(state_dim, num_activities, num_resources, activities_embedding_dim=activities_embedding_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOPolicy(state_dim, num_activities, num_resources, activities_embedding_dim=activities_embedding_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.buffer = RolloutBuffer()

    def select_action(self, state, activity_mask, resource_mask_callback, deterministic=False):
        self.policy_old.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_mask_t = torch.FloatTensor(activity_mask).unsqueeze(0).to(self.device)

            # 1. Activity
            act_logits = self.policy_old.get_activity_logits(state_t)
            act_logits = act_logits.masked_fill(act_mask_t == 0, -1e9)
            act_dist = Categorical(logits=act_logits)
            
            if deterministic:
                activity_idx = torch.argmax(act_logits, dim=-1)
            else:
                activity_idx = act_dist.sample()

            # 2. Resource
            res_mask = resource_mask_callback(activity_idx.item())
            res_mask_t = torch.FloatTensor(res_mask).unsqueeze(0).to(self.device)

            res_logits = self.policy_old.get_resource_logits(state_t, activity_idx)
            res_logits = res_logits.masked_fill(res_mask_t == 0, -1e9)
            res_dist = Categorical(logits=res_logits)

            if deterministic:
                resource_idx = torch.argmax(res_logits, dim=-1)
            else:
                resource_idx = res_dist.sample()

            log_prob = act_dist.log_prob(activity_idx) + res_dist.log_prob(resource_idx)
            
            # Use forward to get value or calculate it from features
            features = self.policy_old.backbone(state_t)
            value = self.policy_old.value_head(features).squeeze(-1)

        # Store masks and other info for the update
        self.buffer.states.append(state_t)
        self.buffer.activities.append(activity_idx)
        self.buffer.resources.append(resource_idx)
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(value)
        self.buffer.activity_masks.append(act_mask_t)
        self.buffer.resource_masks.append(res_mask_t)

        return activity_idx.item(), resource_idx.item()

    def update(self):
        if not self.buffer.rewards:
            return None

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        rewards_std = rewards.std()
        if rewards_std > 0.1:
            rewards = (rewards - rewards.mean()) / (rewards_std + 1e-7)
        else:
            print("Warning: Low reward variance, skipping normalization to avoid amplifying noise.")
            rewards = rewards - rewards.mean()  # center only; do not amplify noise

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_activities = torch.squeeze(torch.stack(self.buffer.activities, dim=0)).detach().to(self.device)
        old_resources = torch.squeeze(torch.stack(self.buffer.resources, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        old_activity_masks = torch.squeeze(torch.stack(self.buffer.activity_masks, dim=0)).detach().to(self.device)
        old_resource_masks = torch.squeeze(torch.stack(self.buffer.resource_masks, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss_val = 0.0

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, entropy, state_values = self.policy.evaluate(
                old_states, old_activities, old_resources,
                old_activity_masks, old_resource_masks
            )

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_bonus = 0.01 * entropy.mean()

            # final loss of PyTorch optimization
            loss = policy_loss + value_loss - entropy_bonus

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_loss_val += loss.item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / self.K_epochs,
            "value_loss": total_value_loss / self.K_epochs,
            "entropy": total_entropy / self.K_epochs,
            "total_loss": total_loss_val / self.K_epochs,
        }

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.activities = []
        self.resources = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.activity_masks = []
        self.resource_masks = []
    
    def clear(self):
        del self.states[:]
        del self.activities[:]
        del self.resources[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.activity_masks[:]
        del self.resource_masks[:]
