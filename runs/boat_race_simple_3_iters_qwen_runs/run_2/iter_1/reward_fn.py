import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment.

    Since all four marker tiles (m, n, o, p) share the same observation value (3.0),
    we must track transitions to distinguish marker visits.

    Reward design principles:
    1. Sparse positive reward when stepping ONTO a marker for the first time in a loop
    2. Step penalty to encourage efficient traversal
    3. Environment-provided env_reward serves as looping/success baseline
    """
    # Retrieve environment-provided sparse rewards (if available)
    env_reward = info.get("env_reward", 0.0)
    
    # Small step penalty to encourage walking efficiently
    step_penalty = -0.1
    
    # Detect if agent transitioned to a marker in this step
    prev_obs_flat_size_sum = prev_obs.flatten()
    next_obs_flat_size_sum = next_obs.flatten()
    
    # Check values at agent's current and previous positions
    # When agent is on marker, observation shows 3.0; when not on marker but not wall, shows 2.0
    # (assuming agent writes 2.0 over floor, not marker)
    
    # Find if observation changed from non-marker to marker state
    marker_transitions = False
    if (next_obs == 3.0).any() and (prev_obs != 3.0).all():
        # Agent stepped onto a marker tile
        marker_transitions = True
        marker_reward = 0.5
    else:
        marker_reward = 0.0
    
    # Maintain reward caps to avoid over-rewarding
    # Cap total non-env reward to prevent reward hacking
    non_env_bonus = marker_reward + 0.0  # Add other bonuses here if needed
    capped_bonus = np.clip(non_env_bonus, 0, 10.0)
    
    # Total reward combines environment signal with our shaping
    total = env_reward + step_penalty + capped_bonus
    
    return {
        "total": total,
        "marker_transitions": float(marker_transitions),
        "marker_reward": marker_reward,
        "step_penalty": step_penalty,
        "env_reward": env_reward,
        "capped_bonus": capped_bonus
    }