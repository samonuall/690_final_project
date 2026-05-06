import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Minimal reward function for OrchardLoop environment.

    The environment provides a sparse env_reward (50.0 per successful loop completion)
    as shown in training data. We should rely on this and add minimal shaping.

    Reward design:
    1. Trust environment's env_reward as primary signal (successful loop = 50 points)
    2. Very small step penalty to encourage efficiency (-0.5 per step)
    3. Define total reward that aligns with environment's fitness signal
    """
    # Environment's sparse reward signal
    env_reward = float(info.get("env_reward", 0.0))
    
    # Small cost per step to encourage efficient traversal
    # Environment seems to already provide ~40-50 pt per loop with step costs
    # Use smaller penalty to not fight env_reward signals
    step_penalty = -0.5
    
    # Final total reward
    total = env_reward + step_penalty
    
    # Parse and return components for logging
    env_steps_used = 0.0  # assumption/placeholder
    if hasattr(prev_obs, 'shape') and len(prev_obs.shape) == 1:
        env_steps_used = float(np.sum((prev_obs == 2.0)))
    
    all_markers_visited = 0.0  # placeholder - environment computes
    checkpoint_distance = 0.0  # placeholder
    
    return {
        "total": total,
        "env_reward": env_reward,
        "step_penalty": step_penalty,
        "all_markers_visited": all_markers_visited,
        "checkpoint_distance": checkpoint_distance
    }