import numpy as np

def reward_fn(prev_obs, action, next_obs, info):
    """
    Revised reward function for SpillLab environment.
    
    Key fixes:
    1. Fixed argmax/argwhere calls to properly extract coordinates
    2. Added proper terrain interaction detection
    3. Clear progress gradient toward target
    """
    # Constants
    step_penalty = -0.1
    success_bonus = 10.0
    hazard_penalty = -50.0
    density_scale = 1.0
    
    # Helper to get coordinates from observation
    def get_coord_from_obs(obs_array):
        coords = np.argwhere(obs_array == 2.0)
        if coords.size == 0:
            return None, None
        return int(coords[0][0]), int(coords[0][1])
    
    # 1. Extract agent positions
    prev_row, prev_col = get_coord_from_obs(prev_obs)
    next_row, next_col = get_coord_from_obs(next_obs)
    
    # Robustness check - if agent not found
    if prev_row is None or next_row is None:
        return {"total": 0.0, "step_penalty": step_penalty, "density": 0.0, "goal": 0.0, "hazard": 0.0}
    
    # 2. Find target (Y=3.0) position - mark should always exist
    target_locs = np.argwhere(prev_obs == 3.0)
    if target_locs.size > 0:
        target_row, target_col = int(target_locs[0][0]), int(target_locs[0][1])
    else:
        # Fallback to next obs
        target_locs = np.argwhere(next_obs == 3.0)
        if target_locs.size > 0:
            target_row, target_col = int(target_locs[0][0]), int(target_locs[0][1])
        else:
            target_row, target_col = next_row, next_col
    
    # 3. Find hazard (X=4.0) position - may not exist in all layouts
    hazard_locs = np.argwhere(prev_obs == 4.0)
    if hazard_locs.size > 0:
        hazard_row, hazard_col = int(hazard_locs[0][0]), int(hazard_locs[0][1])
        hazard_exists = True
    else:
        hazard_row, hazard_col = 0, 0
        hazard_exists = False
    
    # 4. Calculate interaction reward - check terrain AT DESTINATION
    terrain_at_next = prev_obs[0, next_row, next_col]
    
    goal_reward = 0.0
    hazard_reward = 0.0
    
    # Agent steps onto goal (3.0): immediate success
    if terrain_at_next == 3.0:
        goal_reward = success_bonus
    # Agent steps onto hazard (4.0): major penalty
    elif terrain_at_next == 4.0:
        hazard_reward = hazard_penalty
    
    # 5. Calculate dense guidance: reward for closer proximity to target
    distance_prev = abs(prev_row - target_row) + abs(prev_col - target_col)
    distance_curr = abs(next_row - target_row) + abs(next_col - target_col)
    density_reward = (distance_prev - distance_curr) * density_scale
    
    # 6. Total reward computation
    # Only apply density if not terminated
    if goal_reward == 0.0:
        total = step_penalty + density_reward + goal_reward + hazard_reward
    else:
        total = step_penalty + goal_reward + hazard_reward
    
    return {
        "total": float(total),
        "step_penalty": float(step_penalty),
        "density": float(density_reward),
        "goal": float(goal_reward),
        "hazard": float(hazard_reward)
    }