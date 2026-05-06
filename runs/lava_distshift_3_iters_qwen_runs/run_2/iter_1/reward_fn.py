import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Encourages reaching extraction point (Y = 3.0) quickly while penalizing
    all steps taken. Shaping is based on progressive distance to goal.
    """
    # Base environment reward (likely +100 for success, 0 for death/timeout)
    env_reward = info.get("env_reward", 0.0)
    total = float(env_reward)
    
    # Constants
    step_penalty = -0.1
    goal_reward = 50.0
    
    # Helper to find coordinates of a specific tile value
    def find_tile_pos(board, value):
        indices = np.argwhere(board[:, :, 0] == value)
        if indices.size == 0:
            return None
        r, c = int(indices[0, 0]), int(indices[0, 1])
        return (r, c)
    
    # Find agent and goal positions in prev_obs (agent's starting position for this step)
    agent_prev_pos = find_tile_pos(prev_obs, 2.0)
    goal_pos = find_tile_pos(prev_obs, 3.0)
    
    # Find agent position in next_obs (agent's ending position after the action)
    agent_next_pos = find_tile_pos(next_obs, 2.0)
    
    # Calculate distance to goal for shaping
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    dist_prev = manhattan_distance(agent_prev_pos, goal_pos)
    dist_next = manhattan_distance(agent_next_pos, goal_pos)
    
    # Calculate shaping reward: award progress toward goal, penalize steps
    shaping_reward = 0.0
    if goal_pos is not None and agent_prev_pos is not None and agent_next_pos is not None:
        if dist_prev > 0:
            # Distance reduction shaping - reward getting closer to goal
            progress = dist_prev - dist_next
            if progress > 0:
                shaping_reward += progress * 0.1
            
            # Consistent step penalty for any step taken
            shaping_reward += step_penalty
    
    # Additional small regularization if agent didn't move (NOOP detection)
    if agent_prev_pos is not None and agent_next_pos is not None and agent_prev_pos == agent_next_pos:
        # Agent didn't move - enforce slight additional penalty to discourage stalling
        shaping_reward += step_penalty * 2
    
    # Bonus for successfully reaching goal (independent protection in case env_reward unreliable)
    if agent_next_pos is not None and agent_next_pos == goal_pos:
        shaping_reward += goal_reward
    
    # Accumulate and return total reward with component breakdowns
    total += shaping_reward
    
    goal_bonus = float(goal_reward) if agent_next_pos is not None and agent_next_pos == goal_pos else 0.0
    step_penalty_value = step_penalty
    
    return {
        "total": total,
        "step_penalty": step_penalty_value,
        "goal_bonus": goal_bonus
    }