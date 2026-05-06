import numpy as np

AGENT_TILE = 2.0
TARGET_TILE = 3.0  # Y = Extraction point
HAZARD_TILE = 4.0  # X = Chemical spill

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab chemical lab navigation.
    
    Design principles:
    - Encourage reaching extraction point (Y=3.0) quickly with high positive reward
    - Strongly penalize entering chemical spill zones (X=4.0)
    - Apply consistent step penalty to encourage efficient pathing
    - Works regardless of layout configuration or start/target positions
    
    Args:
        prev_obs : np.ndarray                   board observation before the action
        action   : int                         action index (0-indexed)
        next_obs : np.ndarray                   board observation after the action
        info     : dict                        contains at minimum: "env_reward": float
    
    Returns:
        dict with required key "total" (float used for PPO training) plus component keys
    """
    
    goal_reward = 0.0
    hazard_penalty = 0.0
    step_penalty = -0.1
    
    # Flatten observations to 1D for easier indexing
    next_flat = next_obs.flatten()
    
    # Check if extraction point (Y = 3.0) is present in grid
    target_exists = np.any(next_flat == TARGET_TILE)
    
    # Check if hazard (X = 4.0) is present in grid  
    hazard_exists = np.any(next_flat == HAZARD_TILE)
    
    # Award reward for reaching extraction point
    if hazard_exists and target_exists:
        # Agent reached target but collided with hazard
        goal_reward = 15.0
        hazard_penalty = -15.0
    elif hazard_exists:
        # Agent failed (hit chemical spill)
        goal_reward = 0.0
        hazard_penalty = -10.0
    elif target_exists:
        # Agent succeeded (reached extraction point)
        goal_reward = 15.0
        hazard_penalty = 0.0
    else:
        # Agent is still navigating, no goal/hazard reached
        goal_reward = 0.0
        hazard_penalty = 0.0
    
    # Add step penalty to encourage efficient navigation
    step_penalty = -0.1
    
    # Compute total reward
    total = goal_reward + hazard_penalty + step_penalty
    
    return {
        "total": total,
        "goal_reward": goal_reward,
        "step_penalty": step_penalty
    }