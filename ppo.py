import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, input_dim=259, action_dim=2, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))
        log_std = torch.clamp(self.log_std_head(x), -2, 2)
        std = log_std.exp()
        return Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, input_dim=259, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value


class PPOConfig:
    def __init__(self, ppo_eps=0.2, ppo_grad_descent_steps=10):
        self.ppo_eps = ppo_eps
        self.ppo_grad_descent_steps = ppo_grad_descent_steps


def compute_ppo_loss(actor, obs, actions, advantages, old_log_probs, clip_eps):
    dist = actor(obs)
    log_probs = dist.log_prob(actions).sum(dim=-1)
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss, log_probs


def train_ppo(actor, critic, actor_optimizer, critic_optimizer, obs_batch, act_batch, adv_batch, rtg_batch, config):
    obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=actor.fc1.weight.device)
    act_batch = torch.tensor(act_batch, dtype=torch.float32, device=actor.fc1.weight.device)
    adv_batch = torch.tensor(adv_batch, dtype=torch.float32, device=actor.fc1.weight.device)
    rtg_batch = torch.tensor(rtg_batch, dtype=torch.float32, device=actor.fc1.weight.device)

    old_log_probs = actor(obs_batch).log_prob(act_batch).sum(dim=-1).detach()
    for _ in range(config.ppo_grad_descent_steps):
        actor_loss, _ = compute_ppo_loss(actor, obs_batch, act_batch, adv_batch, old_log_probs, config.ppo_eps)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

    value_preds = critic(obs_batch).squeeze()
    critic_loss = F.mse_loss(value_preds, rtg_batch)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()
