import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Calculates a structured reward signal for the SpillLab environment.
    
    Args:
        prev_obs : np.ndarray  board observation before the action (float32, shape (1, 7, 9))
        action   : int         action index (0-indexed)
        next_obs : np.ndarray  board observation after the action (float32, shape (1, 7, 9))
        info     : dict        sparse environment rewards/flags
    
    Returns:
        dict with keys:
            "total"   -> Final scalar reward for PPO training
            "step_penalty" -> Small negative reward for taking steps
            "goal_reward" -> Shaped reward for approaching the goal (Y)
            "hazard_penalty" -> Penalty for encountering hazard (X)
    """
    
    # 1. Extract Base Sparse Reward
    # We sum the environment's sparse signal with our dense shaping to train efficiently.
    env_reward = info.get("env_reward", 0.0)
    
    # 2. Identify Positions
    # The tile encoding uses specific float values.
    # We assume the agent persists as '2.0' and target as '3.0'.
    # We locate coordinates where these values are found.
    
    # Locate Agent Positions
    prev_agent_positions = np.argwhere(prev_obs == 2.0)
    curr_agent_positions = np.argwhere(next_obs == 2.0)
    
    # Locate Target Position (Y = 3.0)
    # Target is static, so finding it in either prev or next is sufficient. 
    # We prefer next_obs to keep it consistent with current state if possible, 
    # but prev_obs is safer if next_obs is corrupted (e.g. terminal state reset).
    target_positions = np.argwhere(next_obs == 3.0)
    fallback_target = np.argwhere(prev_obs == 3.0)
    target_positions = target_positions if len(target_positions) > 0 else fallback_target
    
    # Helper to calculate Manhattan Distance
    # Input: locs (array of [0, r, c]), mode: 'diff' (between 2 locs arrays) or 'min' (many to one)
    # Here we just need scalar coords to compute distance.
    def get_pos(loc):
        # loc is shape (N, 3) or (N, 2)? prev_obs (1,7,9) -> np.argwhere((0,r,c))
        if len(loc) == 0:
            return (0, 0)
        # Extract average/first position
        r = loc[:, 1].mean()
        c = loc[:, 2].mean()
        return (int(round(r)), int(round(c)))
    
    # Extract coords (row, col)
    if len(prev_agent_positions) > 0:
        prev_agent_r, prev_agent_c = get_pos(prev_agent_positions)
    else:
        # If agent not found in prev (start step or reset), assume valid logic not needed for shaping
        # Use defaults to avoid crash
        prev_agent_r, prev_agent_c = 3, 4
        
    if len(curr_agent_positions) > 0:
        curr_agent_r, curr_agent_c = get_pos(curr_agent_positions)
    else:
        # Agent lost (most likely death/hazard in terminal step)
        curr_agent_r, curr_agent_c = prev_agent_r, prev_agent_c
        
    # Target coords
    if len(target_positions) > 0:
        target_r, target_c = get_pos(target_positions)
    else:
        target_r, target_c = 3, 4 

    # 3. Calculate Shaping Rewards
    # Calculate Manhattan Distance to Target (Takes row,col)
    if (prev_agent_r, prev_agent_c) == (target_r, target_c):
        prev_dist = 0
    else:
        prev_dist = abs(prev_agent_r - target_r) + abs(prev_agent_c - target_c)

    if (curr_agent_r, curr_agent_c) == (target_r, target_c):
        curr_dist = 0
    else:
        curr_dist = abs(curr_agent_r - target_r) + abs(curr_agent_c - target_r)
        
    # Goal Reward (Shaping): Small reward for reducing distance to target
    # Scale: 0.5 * reduction
    goal_shaping = 0.1 * (prev_dist - curr_dist)
    
    # Check for Hazard (X = 4.0)
    hazard_penalty = 0.0
    # If agent disappeared from '2.0' in next_obs, check if previous spot became 4.0
    # Or simply check if agent appears on 4.0 in next_obs (if it overwrites)
    # Or if agent was safe but environment hit hazard.
    
    # Check obs value at agent's new position
    # Note: if agent died, next_obs might not show agent '2.0', so checking grid value is key.
    # We flatten the grid to check values at specific coords.
    # We map (r,c) to index: idx = r + c * height. shape height=7.
    
    def is_hazard_at(r, c, obs_array):
        # obs_array shape (1, 7, 9) -> indices. r must be 0..6, c 0..8
        if 0 <= r < 7 and 0 <= c < 9:
            val = obs_array[0, r, c] # r,c in (1,7,9) notation
            if val == 4.0:
                return True
        return False
        
    # We check if the agent is *forced* onto a hazard
    # Scenario A: next_obs shows agent at hazard (rare if agent ID overwrites)
    # Scenario B: Agent vanished from next_obs, but was at prev_pos.
    # We assume if agent is present (2.0) in next_obs, the hazard penalty is handled by env_reward.
    # However, explicit hazard detection prevents the agent from learning to "step on X and hope".
    if curr_agent_r is not None and curr_agent_c is not None:
        # Check if the new position is a hazard tile
        if is_hazard_at(curr_agent_r, curr_agent_c, next_obs):
             hazard_penalty = -5.0 # Strong penalty for stepping on hazard
            
    if prev_dist > curr_dist:
        goal_shaping = 0.1 * (prev_dist - curr_dist)
    elif prev_dist < curr_dist:
        goal_shaping = 0.1 * (prev_dist - curr_dist) # Will be negative
    else:
        goal_shaping = 0.0 # Equal dist (not possible usually with Manhattan)

    # 4. Step Penalty
    # Encourages efficient navigation
    step_penalty = -0.1

    # 5. Compute Total
    # Base: env_reward (contains success/fail terminal signals from environment)
    # + Shaping (Dense rewards for speed and navigation)
    # We do NOT modulate env_reward based on safety checks unless env_reward is 0.
    # The goal reward here is a component for shaping, contributing to Total.
    
    total_reward = env_reward + goal_shaping + step_penalty + hazard_penalty
    
    # Ensure no huge negative spikes if env_reward is 0 and hazard_frac is only negative small
    # (hazard_penalty logic above handles detection)
    
    return {
        "total": float(total_reward),
        "step_penalty": step_penalty,
        "goal_reward": goal_shaping,
        "hazard_penalty": hazard_penalty
    }