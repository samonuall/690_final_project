import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab navigation task using PPO.
    
    Encourages dense reward signals:
    1. Progressive reward for reducing distance to target.
    2. Step penalty for efficiency (avoid unnecessary movement).
    3. Bonus for reaching goal.
    4. Penalty for falling into hazards.
    
    Works with variable environments and dynamic layouts.
    """

    # Ensure proper data types
    prev_obs = np.array(prev_obs, dtype=np.float32)
    next_obs = np.array(next_obs, dtype=np.float32)
    
    # Base environment reward (from info)
    env_reward = 0.0
    if info and 'env_reward' in info:
        env_reward = float(info['env_reward'])
    
    # Get observation dimensions (expected: 1 channel, height, width)
    _, height, width = next_obs.shape
    num_tiles = height * width
    
    # Find agent position and distances
    curr_dist = 0.0
    prev_dist = 0.0
    
    # Check if we're on the goal or hazard in current observation
    on_goal = False
    on_hazard = False
    
    # Find agent position using argwhere
    agent_next = np.argwhere(next_obs == 2.0)
    
    # Check if target (Y=3.0) exists in next_obs
    target_next = np.argwhere(next_obs == 3.0)
    
    # Check if hazard (X=4.0) exists in next_obs
    hazard_next = np.argwhere(next_obs == 4.0)
    
    # Compute positions if found
    if len(agent_next) > 0:
        # argwhere returns [c, r, c_idx] for 3D array
        agent_row = int(agent_next[0][1])
        agent_col = int(agent_next[0][2])
        
        # Safe index access for current position
        if 0 <= agent_row < height and 0 <= agent_col < width:
            tile_val = next_obs[0, agent_row, agent_col]
            if tile_val == 3.0:
                on_goal = True
            elif tile_val == 4.0:
                on_hazard = True
    
    # Calculate distance to goal
    goal_distance = float('inf')
    if len(target_next) == 0 and len(target_next) == 0:
        # If target not found in current, try previous
        target_prev = np.argwhere(prev_obs == 3.0)
        if len(target_prev) > 0:
            target_row = int(target_prev[0][1])
            target_col = int(target_prev[0][2])
        else:
            # Default to center if target not found
            target_row, target_col = height // 2, width // 2
        
        if len(agent_next) == 0:
            agent_row, agent_col = height // 2, width // 2
        else:
            agent_row = int(agent_next[0][1])
            agent_col = int(agent_next[0][2])
        
        # Calculate Euclidean distance
        goal_distance = np.sqrt((agent_row - target_row)**2 + (agent_col - target_col)**2)
        if on_goal:
            goal_distance = 0.0
    
    # Calculate previous distance (from prev_obs)
    agent_prev = np.argwhere(prev_obs == 2.0)
    if len(agent_prev) > 0:
        agent_row_prev = int(agent_prev[0][1])
        agent_col_prev = int(agent_prev[0][2])
        tile_val_prev = prev_obs[0, agent_row_prev, agent_col_prev]
        
        if tile_val_prev == 3.0:
            curr_dist = 0.0
            prev_dist = 0.0
        else:
            # Recast positions to compute old distance
            if len(target_next) == 0:
                # fallback to center
                agent_row = agent_row_prev
                agent_col = agent_col_prev
            else:
                agent_row = int(agent_prev[0][1])
                agent_col = int(agent_prev[0][2])
            
            if len(target_next) > 0:
                target_row = int(target_next[0][1])
                target_col = int(target_next[0][2])
            else:
                target_row = height // 2
                target_col = width // 2
            
            prev_dist = np.sqrt((agent_row - target_row)**2 + (agent_col - target_col)**2)
    
    # Compute shaping reward
    shaping_reward = 0.0
    
    # Positive reward for moving closer to goal
    prev_dist_safe = prev_dist if prev_dist != float('inf') else 1.0
    curr_dist_safe = curr_dist
    
    if prev_dist_safe > 0:
        shaping_reward = 0.5 * (prev_dist_safe - curr_dist_safe)
    
    # Bonus for reaching goal
    if on_goal:
        shaping_reward += 5.0
    
    # Step penalty to encourage efficiency
    step_penalty = -0.02
    
    # Hazard penalty
    hazard_penalty = 0.0
    if on_hazard:
        hazard_penalty = -10.0
    
    # Total reward
    total = float(env_reward) + shaping_reward + step_penalty + hazard_penalty
    
    return {
        "total": total,
        "shaping_reward": shaping_reward,
        "step_penalty": step_penalty,
        "hazard_penalty": hazard_penalty,
        "base_env_reward": env_reward
    }