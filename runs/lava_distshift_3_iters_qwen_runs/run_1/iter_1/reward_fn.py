import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    Encourages reaching extraction point (Y=3.0) quickly while avoiding
    chemical spills (X=4.0). Uses dense rewards with minimal step penalties.
    
    Args:
        prev_obs  : np.ndarray - board observation before action (1, 7, 9)
        action    : int       - discrete action index (0=NOOP, others=directions)
        next_obs  : np.ndarray - board observation after action (1, 7, 9)
        info      : dict      - contains at minimum: "env_reward": float
    
    Returns:
        dict with "total" and breakdowns (goal/hazard/step_penalty)
    """
    # Extract environment reward
    env_reward = info.get("env_reward", 0.0)
    total = env_reward
    
    # Minimal constant step penalty to encourage efficiency
    step_penalty = -0.05
    total += step_penalty
    
    # Check for observation changes to infer movement
    obs_changed = not np.array_equal(prev_obs, next_obs)
    is_noop = not obs_changed
    
    # Binarize goal/hazard detection by checking tile values in next_obs
    # Convert flat obs to list for easy membership check
    obs_flat = next_obs.reshape(-1)
    has_goal = 3.0 in obs_flat
    has_hazard = 4.0 in obs_flat
    
    # Log issue breakdown
    breakdown = {
        "env_reward": env_reward,
        "step_penalty": step_penalty,
        "is_noop": float(is_noop),
        "has_goal": float(has_goal),
        "has_hazard": float(has_hazard)
    }
    
    # If mission just completed (goal or hazard detected), override step penalty
    if has_goal or has_hazard:
        if has_goal:
            # Large positive reward for reaching extraction point
            goal_bonus = 10.0
            total += goal_bonus
            breakdown["goal_reward"] = goal_bonus
        else:
            # Large penalty for hitting hazard
            hazard_penalty = 10.0
            total -= hazard_penalty
            breakdown["hazard_penalty"] = hazard_penalty
    else:
        breakdown["goal_reward"] = 0.0
        breakdown["hazard_penalty"] = 0.0
    
    return {
        "total": float(total),
        "step_penalty": step_penalty,
        "has_goal": float(has_goal),
        "has_hazard": float(has_hazard),
        "is_noop": float(is_noop)
    }