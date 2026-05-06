import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    Encourages reaching extraction point (Y) as agent while avoiding
    chemical spills (X). Uses sparse shaping at destination tile level.
    
    Args:
        prev_obs  : np.ndarray - board observation before action (1, 7, 9)
        action    : int       - discrete action index (0=NOOP, others=directions)
        next_obs  : np.ndarray - board observation after action (1, 7, 9)
        info      : dict      - contains at minimum: "env_reward": float
    
    Returns:
        dict with "total" and breakdowns (goal/hazard/step)
    """
    # Extract environment reward
    env_reward = info.get("env_reward", 0.0)
    total = env_reward
    
    # Locate the agent's position in prev_obs (value 2.0)
    agent_indices = np.where(prev_obs == 2.0)
    
    if len(agent_indices[0]) == 0:
        return {
            "total": float(total),
            "has_goal": 0.0,
            "has_hazard": 0.0,
            "stationary": 1.0,
            "env_reward": env_reward
        }
    
    prev_y = int(agent_indices[0][0])
    prev_x = int(agent_indices[1][0])
    
    # Locate agent in next_obs (verify update)
    next_indices = np.where(next_obs == 2.0)
    
    is_stationary = len(next_indices[0]) == 0 or (len(next_indices[0]) > 0 and 
                                         next_indices[0][0] == prev_y and 
                                         next_indices[1][0] == prev_x)
    
    # Get tile value at agent's current position BEFORE moving
    # Ensure scalar extraction by flattening first
    tile_at_agent_dest = float(prev_obs.flatten()[prev_y * 9 + prev_x])
    tile_at_new_pos = float(next_obs.flatten()[prev_y * 9 + prev_x])
    
    component = 0.0
    
    # Check tile category at destination tile
    if tile_at_agent_dest == 3.0:
        # Agent stepped onto Goal (Y)
        goal_bonus = 50.0
        total += goal_bonus
        component = goal_bonus
    elif tile_at_agent_dest == 4.0:
        # Agent stepped onto Hazard (X)
        hazard_penalty = 50.0
        total -= hazard_penalty
        component = -hazard_penalty
    
    # Step penalty proportional to grid size (encourages efficiency)
    step_penalty = -1.0 / 7.0  
    total += step_penalty
    
    # Detect episode end from environment
    terminated = abs(env_reward) > 100.0
    
    return {
        "total": float(total),
        "step_penalty": float(step_penalty),
        "goal_bonus": float(goal_bonus if tile_at_agent_dest == 3.0 else 0.0),
        "hazard_penalty": float(hazard_penalty if tile_at_agent_dest == 4.0 else 0.0),
        "is_stationary": float(is_stationary),
        "terminated": float(terminated)
    }