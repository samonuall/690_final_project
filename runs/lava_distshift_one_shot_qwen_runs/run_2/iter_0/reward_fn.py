import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Rewards:
      - Reward for reaching the Goal (Y) tile.
      - Penalty for each step to encourage shortest path.
      - Penalty for triggering a Hazard (X) tile.
      - Goal shaping about reducing distance to target.
    """
    # Extract environment reward if available (default to 0 if missing)
    env_reward = info.get("env_reward", 0.0)
    
    # Constants for shaping and rewards
    STEP_PENALTY = -1.0       # Encourage shortest path
    GOAL_REWARD = 10.0        # Large bonus for actually hitting the goal
    GOAL_SHAPING = -1.0       # Proximity shaping to encourage direction toward goal
    HAZARD_PENALTY = -10.0    # Penalty for stepping onto hazard X
    
    total_accum = env_reward
    goal_shaping = 0.0
    safety_penalty = 0.0
    
    # Resolve dimensions of observation
    H, W = next_obs.shape[1], next_obs.shape[2]
    
    # Locate Goal (Tile Y = 3.0)
    goal_coords = np.argwhere(next_obs == 3.0)
    
    # Locate Agent (Tile A = 2.0)
    agent_coords = np.argwhere(next_obs == 2.0)
    
    # Check if Hazard (Tile X = 4.0) exists
    hazard_exists = np.any(next_obs == 4.0)
    
    agent_found = len(agent_coords) > 0
    goal_found = len(goal_coords) > 0
    
    if goal_found:
        # Agent has reached goal (Y tile)
        goal_shaping = GOAL_REWARD
        total_accum += GOAL_REWARD
    
    elif hazard_exists:
        # Agent stepped on hazard (X tile)
        safety_penalty = HAZARD_PENALTY
        total_accum += HAZARD_PENALTY
    
    # Apply step penalty regardless (encourage efficiency; if goal/hazard
    # is reached, override bonuses may be dominated by this, but still
    # encourage direct routing)
    total_accum += STEP_PENALTY
    
    # Log components for analysis
    goal_shaping_val = 0.0
    safety_penalty_val = 0.0
    
    # If goal found, goal_shaping is GOAL_REWARD
    if goal_found:
        goal_shaping_val = GOAL_REWARD
    # If hazard exists, safety_penalty is HAZARD_PENALTY
    if hazard_exists:
        safety_penalty_val = HAZARD_PENALTY
    
    return {
        "total": float(total_accum),
        "env_reward": float(env_reward),
        "step_penalty": float(STEP_PENALTY),
        "goal_shaping": float(goal_shaping_val),
        "safety_penalty": float(safety_penalty_val)
    }