import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for the SpillLab environment.
    
    Encourages the agent to reach the extraction point (Y) efficiently
    while penalizing steps and time. It relies on sparse rewards from
    info['env_reward'] but adds dense shaping for trajectory optimization.
    """
    # Start with the environment's base reward signal
    # This handles success/fail termination if the environment provides it (e.g. +100/-100)
    env_reward = info.get("env_reward", 0.0)
    total = env_reward

    # Step penalty to encourage brevity and discourage staying in place
    # without reducing distance to goal or if step is NOOP.
    step_penalty = -0.1
    shaping_reward = step_penalty

    # Helper to find coordinates of a specific tile value
    # Returns a tuple (r, c) or None
    def find_tile_pos(board, value):
        # board is float32 (1, 7, 9). We look for the single instance of value.
        coords = np.argwhere(board[:, :, 0] == value)
        if coords.size == 0:
            return None
        return (int(coords[0, 0]), int(coords[0, 1]))

    # Locate Agent
    # Agent is always represented by 2.0 in the observation grid
    # In prev_obs, this is where the agent started the step
    # In next_obs, this is where the agent ended the step
    agent_prev_pos = find_tile_pos(prev_obs, 2.0)
    agent_next_pos = find_tile_pos(next_obs, 2.0)

    # Locate Goal (Y) - We use prev_obs positions to predict where the agent is relative to the goal
    # The goal position is constant in the world, so prev_obs is a valid map reference
    goal_pos = find_tile_pos(prev_obs, 3.0)

    # Distance shaping: Encourage moving closer to the goal continuously
    # This prevents the agent from drifting or waiting unnecessarily
    if agent_prev_pos is not None and goal_pos is not None:
        # Calculate Manhattan distance
        dist_prev = abs(agent_prev_pos[0] - goal_pos[0]) + abs(agent_prev_pos[1] - goal_pos[1])
        
        dist_next = 0.0
        if agent_next_pos is not None:
            dist_next = abs(agent_next_pos[0] - goal_pos[0]) + abs(agent_next_pos[1] - goal_pos[1])
        
        # Reward for getting closer (reduction in distance)
        # Avoid over-rewarding if dist_prev is already 0
        if dist_prev > 0:
            dist_diff = dist_prev - dist_next
            if dist_diff > 0:
                # Normalize reward based on max possible distance (approx 7+9=16)
                # This gives a strong signal for approach
                shaping_reward += dist_diff * 0.1 

    # Detecting Goal Arrival: 
    # The goal reward might be in env_reward, but we double-check proximity 
    # to ensure the agent receives a distinct 'maximization' signal upon arrival.
    # We also guard against the case where the environment doesn't signal goal success.
    # Even if we are on the goal (dist_next == dist_prev == 0), we might need a bonus 
    # to encourage finishing the task immediately rather than spinning.
    # Logic: If agent_next_pos matches goal_pos (where 3.0 was in prev_obs), treat as Success.
    if agent_next_pos is not None and agent_next_pos == goal_pos:
        shaping_reward += 10.0
        
    # Hazard/Death Detection (Optional fallback if env_reward is unreliable)
    # If the agent disappeared from next_obs (meaning it likely crashed into hazard),
    # or applies if the tile at next_pos was 4.0 (depending on render logic).
    # Assuming env_reward handles hazard termination correctly.
    if agent_next_pos is None and agent_prev_pos is not None:
        # If agent is missing in next obs while prev_e had agent, 
        # assume death. Add penalty if info didn't catch it.
        if env_reward >= 0 and env_reward < 10:
             shaping_reward += -50.0

    # Accumulate shaping reward
    total += shaping_reward

    return {"total": float(total)}