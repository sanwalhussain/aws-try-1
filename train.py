import torch
import numpy as np
from env import create_env_with_ppo_policy
from ppo import Actor, Critic, PPOConfig, train_ppo
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(259, 2).to(device)
critic = Critic(259).to(device)

env = create_env_with_ppo_policy(actor, device)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

config = PPOConfig(ppo_eps=0.2, ppo_grad_descent_steps=10)

epochs = 300
episodes_per_batch = 2
gamma = 0.99

all_rewards = []
actor_losses = []
critic_losses = []

for epoch in range(epochs):
    obs_batch, act_batch, adv_batch, rtg_batch = [], [], [], []
    batch_rewards = []

    for _ in range(episodes_per_batch):
        raw_obs, _ = env.reset()
        obs = raw_obs
        trajectory_obs, trajectory_acts, trajectory_rewards, trajectory_log_probs = [], [], [], []

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = actor(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = action.squeeze().cpu().numpy()

            next_raw_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = next_raw_obs

            trajectory_obs.append(obs)
            trajectory_acts.append(action)
            trajectory_rewards.append(reward)
            trajectory_log_probs.append(log_prob.detach().cpu().numpy())
            obs = next_obs

            if terminated or truncated:
                break

        rtg = np.zeros_like(trajectory_rewards, dtype=np.float32)
        adv = np.zeros_like(trajectory_rewards, dtype=np.float32)
        running_total = 0

        for t in reversed(range(len(trajectory_rewards))):
            running_total = trajectory_rewards[t] + gamma * running_total
            rtg[t] = running_total

        obs_tensor = torch.tensor(np.array(trajectory_obs), dtype=torch.float32, device=device)
        with torch.no_grad():
            values = critic(obs_tensor).squeeze().cpu().numpy()
        adv = rtg - values

        obs_batch.extend(trajectory_obs)
        act_batch.extend(trajectory_acts)
        rtg_batch.extend(rtg)
        adv_batch.extend(adv)
        batch_rewards.append(sum(trajectory_rewards))

    actor_loss, critic_loss = train_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        obs_batch,
        act_batch,
        adv_batch,
        rtg_batch,
        config,
    )

    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    all_rewards.append(np.mean(batch_rewards))

    print(f"Epoch {epoch + 1}/{epochs}, Avg Reward: {np.mean(batch_rewards):.2f}, Actor Loss: {actor_loss:.2f}, Critic Loss: {critic_loss:.2f}")

torch.save(actor.state_dict(), "ppo_actor.pth")
torch.save(critic.state_dict(), "ppo_critic.pth")

plt.plot(all_rewards, label="Rewards")
plt.xlabel("Epochs")
plt.ylabel("Average Reward")
plt.legend()
plt.savefig("training_rewards.png")

plt.plot(actor_losses, label="Actor Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("actor_training_losses.png")

plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("critic_training_losses.png")
