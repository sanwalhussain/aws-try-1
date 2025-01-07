import gymnasium as gym
import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math import clip


class PPOPolicy(BasePolicy):
    def __init__(self, obj, seed, actor_model, action_space, device):
        super(PPOPolicy, self).__init__(control_object=obj, random_seed=seed)
        self.actor_model = actor_model
        self.device = device
        self.action_space = action_space
        self.discrete_action = obj.engine.global_config["discrete_action"]
        self.use_multi_discrete = obj.engine.global_config["use_multi_discrete"]
        self.steering_unit = 2.0 / (obj.engine.global_config["discrete_steering_dim"] - 1)
        self.throttle_unit = 2.0 / (obj.engine.global_config["discrete_throttle_dim"] - 1)

    def act(self, agent_id):
        observation = self.control_object.get_state()

        if isinstance(observation, dict):
            observation = self._preprocess_observation(observation)

        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist = self.actor_model(obs_tensor)
            action = dist.sample().squeeze().cpu().numpy()

        action = [clip(action[i], -1.0, 1.0) for i in range(len(action))]
        self.action_info["action"] = action
        return action

    def _preprocess_observation(self, observation):
        processed_obs = []

        lidar_data = observation.get("lidar", np.zeros(240))
        processed_obs.extend(lidar_data)

        position = observation.get("position", [0.0, 0.0, 0.0])
        processed_obs.extend(position)

        velocity = observation.get("velocity", [0.0, 0.0])
        processed_obs.extend(velocity)

        processed_obs.append(observation.get("heading_theta", 0.0))
        processed_obs.append(observation.get("roll", 0.0))
        processed_obs.append(observation.get("pitch", 0.0))
        processed_obs.append(observation.get("steering", 0.0))
        processed_obs.append(observation.get("throttle_brake", 0.0))

        crash_info = [
            float(observation.get("crash_vehicle", False)),
            float(observation.get("crash_object", False)),
            float(observation.get("crash_building", False)),
            float(observation.get("crash_sidewalk", False)),
        ]
        processed_obs.extend(crash_info)

        size = observation.get("size", [0.0, 0.0, 0.0])
        processed_obs.extend(size)

        processed_obs.extend([0.0, 0.0])

        if len(processed_obs) != 259:
            raise ValueError(f"Processed observation size mismatch! Expected 259, got {len(processed_obs)}.")

        return np.array(processed_obs, dtype=np.float32)


def create_env_with_ppo_policy(actor_model, device):
    class MetaPPOPolicy(PPOPolicy):
        def __init__(self, obj, seed):
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            super().__init__(obj, seed, actor_model=actor_model, action_space=action_space, device=device)

    env_config = {
        "use_render": False,
        "traffic_density": 0.1,
        "need_inverse_traffic": True,
        "map": "S",
        "manual_control": False,
        "vehicle_config": {
            "lidar": {
                "num_lasers": 240,
                "distance": 50.0,
                "gaussian_noise": 0.1,
            },
            "show_lidar": False,
        },
        "success_reward": 20.0,
        "driving_reward": 2.0,
        "crash_vehicle_penalty": -10.0,
        "out_of_road_penalty": -5.0,
        "crash_vehicle_done": True,
        "out_of_road_done": True,
        "horizon": None,
        "agent_observation": LidarStateObservation,
        "agent_policy": MetaPPOPolicy,
    }

    return MetaDriveEnv(config=env_config)
