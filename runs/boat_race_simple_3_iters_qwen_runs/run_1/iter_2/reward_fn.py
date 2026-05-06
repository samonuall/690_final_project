import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment.
    
    Strategy:
    - Use env_reward as primary signal (contains sparse loop completion rewards)
    - Add small shaping: movement bonus, light step penalty
    - Wall penalty only when actual collision detected (agent not at position 2.0)
    
    Key insight: Agent may NOT be visible at new position in next_obs when hitting wall,
    so we detect collision by checking if agent position changed unexpectedly.
    """
    # Get environment's sparse reward as base
    env_reward = float(info.get("env_reward", 0.0))
    
    # Find positions
    prev_mask = (prev_obs[0] == 2.0)
    curr_mask = (next_obs[0] == 2.0)
    
    prev_r, prev_c = (-1, -1)
    if np.any(prev_mask):
        pos = np.argwhere(prev_mask)[0]
        prev_r, prev_c = int(pos[0]), int(pos[1])
    
    curr_r, curr_c = (-1, -1)
    if np.any(curr_mask):
        pos = np.argwhere(curr_mask)[0]
        curr_r, curr_c = int(pos[0]), int(pos[1])
    
    # Check if agent moved
    moved = (prev_r != curr_r) or (prev_c != curr_c)
    
    # Active wall collision check: Did agent exit the loop without being detected?
    moved_out = (not moved) and (np.sum(next_obs != 0.0) > np.sum(prev_obs != 0.0))
    
    # Movement rewards
    movement_reward = 0.0
    move_penalty = 0.0
    
    if moved:
        # Small bonus for traversing the loop (shaping)
        movement_reward = 0.8
        
        # Check if we ended up in wall (agent disappeared from observation)
        # If next_obs doesn't show agent anymore, likely hit wall or out of bounds
        agent_lost = not np.any(curr_mask)
        if agent_lost:
            # Hit wall penalty - don't recover much (unrecoverable mistake)
            movement_reward = -1.0
    else:
        # Did an action change position or not?
        # If action was toward boundary and nothing changed - likely hit wall
        move_penalty = -0.1  # Small penalty for not moving
    
    # Additional penalty if agent tried to move but didn't (means wall hit)
    try_agent_calc = {
        1: (-1, 0),  # Up
        2: (1, 0),   # Down
        3: (0, -1),  # Left
        4: (0, 1),   # Right
    }
    
    dr = curr_r - prev_r
    dc = curr_c - prev_c
    
    if not moved and action in try_agent_calc:
        pr, pc = try_agent_calc[action]
        if abs(dr) != abs(pr) or abs(dc) != abs(pc):
            # Hit wall - try moving direction but position didn't change
            move_penalty = -0.5
            movement_reward = -0.3
    
    # Combine rewards
    total = env_reward + movement_reward + move_penalty
    
    return {
        "total": float(total),
        "env_reward": float(env_reward),
        "movement_reward": float(movement_reward),
        "move_penalty": float(move_penalty)
    }