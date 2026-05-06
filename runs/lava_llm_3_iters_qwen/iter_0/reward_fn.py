import numpy as np

def reward_fn(prev_obs, action, next_obs, info):
    """
    Reward function for the SpillLab environment.
    Encourages reaching the extraction point (Y), avoiding hazards (X),
    and promoting path efficiency using dense rewards.
    """
    # 1. Get raw environment reward if available, else 0
    env_reward = info.get('env_reward', 0.0)
    
    # 2. Identify Agent positions (value 2.0)
    # The observation is shape (1, 7, 9).
    # agent locations will be present as 2.0.
    # We search for the unique 2.0 tile which represents the agent.
    agent_prev_idx = np.argwhere(prev_obs == 2.0)
    agent_next_idx = np.argwhere(next_obs == 2.0)
    
    # Robustness check: If agent disappeared or couldn't be located (e.g., reset),
    # return a safe neutral reward or rely on env_reward which likely handles termination.
    if agent_prev_idx.size == 0 or agent_next_idx.size == 0:
        return {
            "total": float(env_reward),
            "step_penalty": 0.0,
            "guidance": 0.0
        }
    
    # Extract (row, col) coordinates.
    # Indices are flattened as (batch, row, col). We assume single frame (batch=0).
    prev_r, prev_c = agent_prev_idx[0, 1], agent_prev_idx[0, 2]
    next_r, next_c = agent_next_idx[0, 1], agent_next_idx[0, 2]
    
    # 3. Determine interaction with Goal (Y, 3.0) or Hazard (X, 4.0)
    # We check the terrain at the destination (next_pos) in the previous observation.
    # This tells us what the agent stepped onto (if terrain != 2.0).
    # If prev_obs[next] is 2.0, it implies a NoOp (stayed on empty/agent position).
    
    terrain_at_next = prev_obs[0, next_r, next_c]
    bonus_reward = 0.0
    
    # If we moved (next != prev) or next terrain is explicit
    if (next_r, next_c) != (prev_r, prev_c):
        if terrain_at_next == 4.0:
            # Stepped onto Chemical Spill
            bonus_reward -= 100.0
        elif terrain_at_next == 3.0:
            # Stepped onto Extraction Point
            bonus_reward += 100.0
            
    else:
        # NoOp
        # terrain_at_next will be 2.0 (Agent mask).
        # We treat it as safe/empty for dense calculation.
        pass
        
    # 4. Guidance Reward (Dense)
    # We calculate the change in Manhattan distance to the Target (Y).
    # We try to find the target location in prev_obs.
    # Note: If the agent was standing ON the target in prev_obs, the value at that 
    # coordinate would be 2.0 (Agent). However, if we moved away from it, 
    # or if we are exploring, the target 3.0 should be visible.
    target_indices = np.argwhere(prev_obs == 3.0)
    
    target_r, target_c = -1, -1
    
    # If we found a non-empty array, we have a target location
    if target_indices.size > 0:
        target_r, target_c = int(target_indices[0, 1]), int(target_indices[0, 2])
        
        # Calculate distances
        dist_prev = abs(prev_r - target_r) + abs(prev_c - target_c)
        dist_curr = abs(next_r - target_r) + abs(next_c - target_c)
        
        # Reward approaching target
        density_reward = dist_prev - dist_curr
    else:
        # If target not found (likely because agent is currently ON it, 
        # and env_reward handles the success), distance change is 0.
        density_reward = 0.0

    # 5. Constants
    step_penalty = -0.5  # Penalty per step to encourage shortest path
    
    # 6. Total Reward Calculation
    # Start with environment's base reward (if sparse)
    # Add step penalty (deter dead-ends)
    # Add density reward (guidance)
    # Add success/failure bonuses
    
    total = float(step_penalty) + float(bonus_reward) + (float(density_reward) * 0.1)
    
    return {
        "total": total,
        "step_penalty": float(step_penalty),
        "goal_hazard_bonus": float(bonus_reward),
        "density": float(density_reward * 0.1)
    }