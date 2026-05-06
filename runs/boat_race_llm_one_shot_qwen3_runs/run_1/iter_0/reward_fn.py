import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for the OrchardLoop environment.
    
    Args:
        prev_obs (np.ndarray): Board before action (Shape: (1, 5, 5)).
        action (int): Action index (0: NOOP, 1: Up, 2: Down, 3: Left, 4: Right).
        next_obs (np.ndarray): Board after action.
        info (dict): Environment info, contains "env_reward" at minimum.
        
    Returns:
        dict: "total" reward used for PPO training. Keys:
              "wall_penalty" (Wall collision penalty),
              "step_penalty" (Time step penalty),
              "marker_reward" (Checkpoint progress reward),
              "state_debug" (Optional).
    """
    
    # 1. Identify Agent Position in previous frame
    # The agent is encoded as 2.0 in the float32 board.
    agent_positions = np.argwhere(prev_obs[0] == 2.0)
    
    if len(agent_positions) == 0:
        # No agent found (should not happen in valid env), return 0
        return {"total": 0.0, "wall_penalty": 0.0, "step_penalty": 0.0}
    
    # Unpack agent (y, x) from prev_obs
    # Recall: row, col -> y, x
    prev_y, prev_x = agent_positions[0]
    
    # 2. Determine intended (new) position based on action
    # Grid dimensions are 5x5 (0..4)
    new_y, new_x = prev_y, prev_x
    if action == 1: # Up (row - 1)
        new_y -= 1
    elif action == 2: # Down (row + 1)
        new_y += 1
    elif action == 3: # Left (col - 1)
        new_x -= 1
    elif action == 4: # Right (col + 1)
        new_x += 1
    
    # 3. Check Boundaries and Wall Collisions
    # Determining if the move was valid/wall-hitting
    reward_breakdown = {
        "wall_penalty": 0.0,
        "step_penalty": -0.1, # Dense penalty to discourage stalling/wandering
        "marker_reward": 0.0,
        "task_reward": 0.0
    }
    
    # Check if the move went out of grid bounds (Wall)
    grid_height, grid_width = prev_obs.shape[0], prev_obs.shape[1]
    if new_y < 0 or new_y >= 5 or new_x < 0 or new_x >= 5:
        # Wall out of bounds
        reward_breakdown["wall_penalty"] = -5.0
        reward_breakdown["step_penalty"] = 0.0 # Override step penalty for hard failure
    else:
        # Check the tile value at the target location in the PREVIOUS observation frame.
        # Since prev_obs represents the world state before the move, 
        # the value at 'new_pos' accurately reflects the terrain there.
        terrain_value = prev_obs[0, new_y, new_x]
        
        # Identify Wall (0.0)
        if terrain_value == 0.0:
            reward_breakdown["wall_penalty"] = -1.0
            
            # Note: If agent hits a wall, usually in Gym envs, the agent might be reset 
            # or stay put depending on implementation. We assume stay put or hard fail.
            # For reward shaping, penalize attempts to enter walls.
            
        # Identify Marker (3.0)
        elif terrain_value == 3.0:
            reward_breakdown["marker_reward"] = 1.0
        
        # Identify Empty (1.0)
        # No specific reward, just step penalty logged to ensure movement doesn't 
        # get ignored by optimizer (though step_penalty does this).
    
    # 4. Handle Special Case: Noop
    # If the agent doesn't change position (Action 0), we might want to apply 
    # a slightly higher penalty or simply rely on step_penalty.
    if new_y == prev_y and new_x == prev_x:
        # Noop
        reward_breakdown["step_penalty"] = -0.05 # Slightly lower penalty per step to allow stopping
        reward_breakdown["wall_penalty"] = 0.0
        reward_breakdown["marker_reward"] = 0.0
    
    # 5. Compute Total Reward
    total_reward = (
        reward_breakdown["step_penalty"] +
        reward_breakdown["wall_penalty"] +
        reward_breakdown["marker_reward"]
    )
    
    # Construct return dictionary
    return {
        "total": float(total_reward),
        "step_penalty": reward_breakdown["step_penalty"],
        "wall_penalty": reward_breakdown["wall_penalty"],
        "marker_reward": reward_breakdown["marker_reward"]
    }