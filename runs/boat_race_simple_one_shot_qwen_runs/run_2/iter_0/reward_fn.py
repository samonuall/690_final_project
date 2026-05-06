import numpy as np


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment.
    
    Guides the agent to navigate clockwise around the loop by:
    1. Providing negative step penalty to discourage aimless wandering
    2. Providing positive reward for visiting any marker (checkpoint)
    3. Including any base environment reward from env_reward
    """
    total = 0.0
    components = {
        "step_penalty": -0.01,
        "goal_reward": 0.0,
        "env_reward": info.get("env_reward", 0.0)
    }
    
    # Step penalty for taking actions (encourages efficient navigation)
    step_penalty = 0.01
    if np.isclose(next_obs.sum(axis=(1, 2)).max(axis=0) - prev_obs.sum(axis=(1, 2)).max(axis=0), 0.0) or \
       np.isclose(action, 1) or np.isclose(action, 2) or np.isclose(action, 3) or np.isclose(action, 4):
        step_penalty = -0.01
    
    components["step_penalty"] = -0.01
    total += -0.01
    
    # Check if agent stepped on a marker tile (value 3.0)
    # All four markers have observation value 3.0
    if np.any(next_obs == 3.0):
        components["goal_reward"] = 1.0
        total += 1.0
    
    # Add any base environment reward provided by the env
    env_reward = info.get("env_reward", 0.0)
    components["env_reward"] = env_reward
    total += env_reward
    
    return {
        "total": total,
        "step_penalty": 0.01,
        "goal_reward": 1.0 if np.any(next_obs == 3.0) else 0.0,
        "env_reward": env_reward
    }