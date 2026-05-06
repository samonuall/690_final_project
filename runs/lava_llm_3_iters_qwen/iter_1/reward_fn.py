import numpy as np

def reward_fn(prev_obs, action, next_obs, info):
    """
    Revised reward function for SpillLab environment.
    Uses stronger dense guidance toward goal, binary success penalty,
    and proper termination handling.
    """
    # 1. Default values
    step_penalty = -0.1
    success_bonus = 10.0
    hazard_penalty = -100.0
    density_scale = 0.5
    
    # 2. Find agent position in both observations
    prev_agent_locs = np.argwhere(prev_obs == 2.0)
    next_agent_locs = np.argwhere(next_obs == 2.0)
    
    if prev_agent_locs.size == 0 or next_agent_locs.size == 0:
        # Episode reset or error
        return {"total": 0.0, "step_penalty": step_penalty, "goal": 0.0, "density": 0.0, "success": 0.0, "hazard": 0.0}
    
    prev_r, prev_c = prev_agent_locs[0][0], prev_agent_locs[0][1]
    next_r, next_c = next_agent_locs[0][0], next_agent_locs[0][1]
    
    # 3. Find target and hazard positions in the static map
    # Since the board changes minimally (only agent moves), search in both
    target_locs = np.argwhere(prev_obs == 3.0)
    hazard_locs = np.argwhere(prev_obs == 4.0)
    
    # Also check NEXT obs since agent position changes
    if target_locs.size == 0:
        target_locs = np.argwhere(next_obs == 3.0)
    if hazard_locs.size == 0:
        hazard_locs = np.argwhere(next_obs == 4.0)
    
    target_r, target_c = target_locs[0][0], target_locs[0][1]
    hazard_r, hazard_c = hazard_locs[0][0], hazard_locs[0][1] if hazard_locs.size > 0 else (-1, -1)
    
    # 4. Calculate current goal distance (after the move)
    distance_to_goal = abs(next_r - target_r) + abs(next_c - target_c)
    
    # 5. Check if agent steps ONTO the goal in this episode transition
    # Check what terrain was at next position in prev_obs (before agent moved)
    next_terrain = prev_obs[0, next_r, next_c]
    
    goal_reward = 0.0
    current_dtype = 0.0
    
    # Check if agent moved onto Goal (3.0)
    if next_terrain == 3.0:
        goal_reward = success_bonus
        current_dtype = 0.0  # Episode ends, no gradient needed
    
    # Check if agent moved onto Hazard (4.0)
    elif next_terrain == 4.0:
        goal_reward = hazard_penalty
        current_dtype = 0.0  # Episode ends on failure
    
    # 6. Density guidance: reward getting closer to goal (even for first steps)
    # Only apply if agent hasn't ended the episode
    if current_dtype == 0.0:
        # Calculate progress distance change
        prev_distance = abs(prev_r - target_r) + abs(prev_c - target_c)
        distance_diff = prev_distance - distance_to_goal
        
        # Positive if moving toward goal, negative if moving away
        density_reward = distance_diff * density_scale
    else:
        density_reward = 0.0
    
    # 7. Episode termination bonus from env if applicable
    env_reward = info.get('env_reward', 0.0)
    
    # 8. Compute total reward
    total = step_penalty + density_reward + goal_reward
    
    return {
        "total": float(total),
        "step_penalty": step_penalty,
        "density": float(density_reward),
        "goal_bonus": float(goal_reward)
    }