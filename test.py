import torch
import numpy as np
from env import create_env_with_ppo_policy
from ppo import Actor

# Helper function for observation processing
def process_observation(obs):
    """
    Process observations from the environment.
    This method ensures that the observation aligns with the expected format for the policy.
    """
    try:
        if not isinstance(obs, np.ndarray):
            raise ValueError(f"Observation must be a numpy array, got {type(obs)}")
        expected_size = 259  # Match the expected observation size
        if obs.shape[0] != expected_size:
            raise ValueError(f"Observation size mismatch! Expected {expected_size}, got {obs.shape[0]}")
        return obs
    except Exception as e:
        print(f"Error in observation processing: {e}")
        return None

# Load PPO-trained actor model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_dim = 259  # Update based on processed observation size
action_dim = 2  # Action space size for steering and throttle

actor = Actor(obs_dim, action_dim).to(device)
actor.load_state_dict(torch.load("ppo_actor.pth", map_location=device))
actor.eval()  # Set to evaluation mode

print("Environment and Trained PPO Model loaded.\n")

# Create the environment with the trained actor model
env = create_env_with_ppo_policy(actor, device)

# Attach the custom observation processor to the environment
env.process_observation = process_observation

# Test loop
print("Starting Test...\n")
num_episodes = 5  # Number of episodes to test
total_rewards = []
success_count = 0  # Count episodes where the agent reaches the destination

for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes}")
    raw_obs, _ = env.reset()
    obs = env.process_observation(raw_obs)
    if obs is None:
        print("Skipping episode due to observation processing error.")
        continue

    total_reward = 0
    step_count = 0

    while True:
        # Render the environment in topdown mode with useful text overlay
        env.render(
            mode="topdown",
            text={
                "Episode": episode + 1,
                "Timestep": env.episode_step,
                "Traffic Density": env.config.get("traffic_density", "N/A"),
                "Total Reward": f"{total_reward:.2f}",
            },
        )

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist = actor(obs_tensor)
            action = dist.sample().squeeze().cpu().numpy()  # Sample action from PPO policy

        next_raw_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = env.process_observation(next_raw_obs)

        if next_obs is None:
            print("Error in processing observation. Terminating episode.")
            break

        total_reward += reward
        step_count += 1

        # Debugging information for each step
        print(f"Step {step_count}:")
        print(f" - Action Taken: {action}")
        print(f" - Reward: {reward:.4f}")
        print(f" - Total Reward: {total_reward:.4f}")
        print(f" - Terminated: {terminated}, Truncated: {truncated}")
        print(f" - Info: {info}\n")

        if terminated or truncated:
            if info.get("arrive_dest", False):  # Check if the agent successfully reached the destination
                success_count += 1
            print(f"Episode Ended. Total Reward: {total_reward:.4f}")
            break

        obs = next_obs

    total_rewards.append(total_reward)

# Summary of test results
print("\nTest Summary:")
for i, reward in enumerate(total_rewards, 1):
    print(f" - Episode {i}: Total Reward = {reward:.4f}")

average_reward = np.mean(total_rewards)
success_rate = (success_count / num_episodes) * 100
print(f"\nAverage Reward over {num_episodes} episodes: {average_reward:.4f}")
print(f"Success Rate: {success_rate:.2f}% ({success_count}/{num_episodes})")

# Close the environment
env.close()
print("\nEnvironment closed. Test completed.")
