import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    The environment ALREADY provides:
    - {+42.0, -52.0, -100.0} via env_reward based on success/death/timeout
    
    This reward function adds only:
    - -0.1/step: Small penalty to encourage efficient navigation
    - 0.1 * distance_reduction: Shaping bonus to guide toward goal (when agent not in goal yet)
    
    This minimal shaping lets environmental signals dominate learning.
    """
    # Base reward from environment (already accounts for success/death/timeout)
    env_reward = float(info.get("env_reward", 0.0))
    total = env_reward
    
    # Small step penalty to encourage faster episodes
    step_penalty = -0.1
    
    # 1. Find fixed world references from prev_obs
    # Agent position (2.0)
    agent_indices = np.argwhere(prev_obs[:, :, 0] == 2.0)
    goal_indices = np.argwhere(prev_obs[:, :, 0] == 3.0)
    
    agent_pos = None
    goal_pos = None
    
    if agent_indices.size > 0:
        r, c = int(agent_indices[0, 0]), int(agent_indices[0, 1])
        agent_pos = (r, c)
    
    if goal_indices.size > 0:
        r, c = int(goal_indices[0, 0]), int(goal_indices[0, 1])
        goal_pos = (r, c)
    
    # 2. Find where agent ended up
    agent_next_indices = np.argwhere(next_obs[:, :, 0] == 2.0)
    if agent_next_indices.size > 0:
        r, c = int(agent_next_indices[0, 0]), int(agent_next_indices[0, 1])
        agent_next_pos = (r, c)
    else:
        # Agent disappeared (assumed death/crashed)
        agent_next_pos = None
    
    # 3. Reward shaping: Base on whether we can see agent + goal and make progress
    shaping_reward = 0.0
    
    if agent_pos is not None and goal_pos is not None:
        # Compute Manhattan distance using only prev position (agent's starting spot)
        dist_prev = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        
        # Only reward progress when agent isn't already at goal
        if agent_pos != goal_pos:
            progress = max(0.0, dist_prev - 8.0)  # Limit bonus magnitude
            shaping_reward += progress * 0.15  # Bonus for actually moving closer
    
    # Add minimal step penalty for any step taken
    shaping_reward += step_penalty
    
    # Accumulate total
    total += shaping_reward
    
    return {"total": total, "step_penalty": -0.1}