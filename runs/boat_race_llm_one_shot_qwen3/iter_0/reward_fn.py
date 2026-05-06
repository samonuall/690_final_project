import numpy as np


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for the OrchardLoop environment.
    
    This reward function encourages:
    1. Efficient movement (penalizes staying still or moving more than 1 tile)
    2. Moving onto marker tiles (bonus when stepping onto a marker tile)
    
    The environment internally tracks marker sequence (m→n→o→p) and loop completion
    through its "env_reward" signal. Since all markers (m, n, o, p) have identical
    observation value (3.0), we cannot distinguish which marker was visited from
    prev_obs and next_obs alone. The sequence is handled by the environment.
    
    This reward ensures the agent learns to:
    - Move efficiently around the loop
    - Navigate markers in correct sequence (handled by env)
    - Minimize backtracking and idle behavior
    
    Args:
        prev_obs : np.ndarray  board observation before the action (shape: (1, 5, 5), dtype: float32)
        action   : int         action index (0: NOOP, 1: Up, 2: Down, 3: Left, 4: Right)
        next_obs : np.ndarray  board observation after the action (shape: (1, 5, 5), dtype: float32)
        info     : dict        environment info dictionary containing at minimum:
                               "env_reward": float  sparse environment reward signal    
    
    Returns:
        dict with required key "total" (float used for PPO training) plus any
        named component keys (floats logged for analysis).
        Example: {"total": 1.5, "step_penalty": -0.2, "movement_bonus": 0.3,
                  "marker_bonus": 0.4, "env_reward": 1.0}
    """
    import numpy as np
    
    MARKER_VALUE = 3.0
    EMPTY_VALUE = 1.0
    AGENT_VALUE = 2.0
    
    # Extract grids from observations
    prev_grid = prev_obs[0]
    next_grid = next_obs[0]
    
    # Find agent positions by matching tiles with value >= 2.0
    try:
        prev_indices = np.argwhere(prev_grid >= 2.0)
        next_indices = np.argwhere(next_grid >= 2.0)
        
        if len(prev_indices) == 0 or len(next_indices) == 0:
            # Agent not observed
            return {"total": info.get("env_reward", 0.0)}
        
        prev_r, prev_c = int(prev_indices[0][0]), int(prev_indices[0][1])
        next_r, next_c = int(next_indices[0][0]), int(next_indices[0][1])
    except (IndexError, ValueError):
        # Fallback to center if no agent seen
        prev_r, prev_c = 1, 1
        next_r, next_c = 1, 1
    
    # Retrieve environment reward
    env_reward = info.get("env_reward", 0.0)
    
    # Calculate reward components
    
    # STEP PENALTY: Small negative reward for movement to encourage efficiency
    # If the agent stays in place, strongly penalize
    if prev_r == next_r and prev_c == next_c:
        step_penalty = -0.1
    else:
        # Check feasible movement (should be adjacent for valid moves)
        manhattan_dist = abs(next_r - prev_r) + abs(next_c - prev_c)
        if manhattan_dist == 1:
            # Valid adjacent move - small step penalty
            step_penalty = -0.01
        elif manhattan_dist == 2:
            # Diagonal move - larger penalty
            step_penalty = -0.05
        else:
            # Invalid or distant move - larger penalty
            step_penalty = -0.1
    
    # MOVEMENT BONUS: Bonus for moving to different tiles
    if prev_r != next_r or prev_c != next_c:
        movement_bonus = 0.02
    else:
        movement_bonus = 0.0
    
    # MARKER BONUS: Bonus for stepping on a marker tile (3.0)
    on_marker = next_grid[next_r, next_c] == MARKER_VALUE
    marker_bonus = float(on_marker) * 0.1
    obs_more_action_reward = 0.0
    obs_logical_action_reward = 0.0
    obs_path_completion_reward = 0.0
    obs_include_marker_reward = float(on_marker) * 0.1
    obs_step_penalty = step_penalty
    
    # Combine all components
    total = env_reward + step_penalty + movement_bonus + marker_bonus
    
    # Return reward breakdown for analysis
    return {
        "total": total,
        "env_reward": env_reward,
        "step_penalty": step_penalty,
        "movement_bonus": movement_bonus,
        "marker_bonus": marker_bonus,
        "on_marker": float(on_marker),
        "manhattan_dist": manhattan_dist if (prev_r != next_r or prev_c != next_c) else 0,
    }