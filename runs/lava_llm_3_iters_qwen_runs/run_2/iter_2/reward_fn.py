import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Improved SpillLab reward function with balanced shaping signals.

    Performance analysis from training data:
    - By step 155,000 agent achieving success (fitness: 42.0)
    - Need to strengthen goal signal for faster learning
    - Reduce excessive base penalties that delay exploration
    """
    
    # THRESHOLDS
    AGENT_ID = 2.0
    TARGET_ID = 3.0
    HAZARD_ID = 4.0
    WALL_ID = 0.0
    
    # REWARD SCALING
    STEP_PENALTY = -0.1          # Small penalty for exploration
    DENSITY_REWARD = 0.5         # More aggressive goal shaping
    
    # 1. EXTRACT SPARSE ENVIRONMENT REWARD
    env_reward = info.get("env_reward", 0.0)
    
    # 2. LOCATE TILE POSITIONS
    height, width = 7, 9
    
    def get_positions(obs_arr, value, default=None):
        """Return first (r,c) where obs[r,c] == value"""
        for r in range(height):
            for c in range(width):
                if obs_arr[0, r, c] == value:
                    return (r, c)
        return default
    
    prev_agent = get_positions(prev_obs, AGENT_ID)
    curr_agent = get_positions(next_obs, AGENT_ID)
    target = get_positions(next_obs, TARGET_ID)
    
    # If agent vanished (terminated), keep last position
    if curr_agent is None:
        curr_agent = prev_agent
    
    # 3. DISTANCE CALCULATION (MANHATTAN)
    if target is None:
        curr_dist = prev_dist = 0
    else:
        prev_dist = abs(prev_agent[0] - target[0]) + abs(prev_agent[1] - target[1]) if prev_agent else -1
        curr_dist = abs(curr_agent[0] - target[0]) + abs(curr_agent[1] - target[1]) if curr_agent else -1
    
    # 4. GOAL REWARD SHAPING (Positive progress signal)
    if prev_dist >= 0 and curr_dist >= 0:
        distance_reduction = prev_dist - curr_dist
        goal_reward = DENSITY_REWARD * distance_reduction
    else:
        goal_reward = 0.0
    
    # 5. HAZARD DETECTION (Immediate failure signal)
    hazard_penalty = 0.0
    if curr_agent is not None and curr_agent != (0,0):
        curr_r, curr_c = curr_agent
        if 0 <= curr_r < height and 0 <= curr_c < width:
            if next_obs[0, curr_r, curr_c] == HAZARD_ID:
                hazard_penalty = -10.0
    
    # 6. FAILSAFE - If agent disappeared entirely (possible reset)
    if curr_agent is None and prev_agent is None:
        curr_reward = 0.0
    else:
        base_penalty = STEP_PENALTY
    
    # 7. ASSEMBLE TOTAL REWARD
    total_reward = env_reward + goal_reward + base_penalty + hazard_penalty
    
    return {
        "total": float(total_reward),
        "step_penalty": base_penalty,
        "goal_reward": float(goal_reward),
        "hazard_penalty": float(hazard_penalty)
    }