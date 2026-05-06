import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Reach the extraction point (Y) as quickly as possible
    2. Avoid chemical spill zones (X)
    3. Discourage unnecessary steps (efficiency)
    4. Provide dense shaping signal via distance-to-goal
    """
    
    # Tile encoding constants
    AGENT_VAL  = 2.0
    TARGET_VAL = 3.0
    HAZARD_VAL = 4.0
    WALL_VAL   = 0.0
    
    # Extract 2D boards (squeeze the channel dimension)
    prev_board = prev_obs[0] if prev_obs.ndim == 3 else prev_obs
    next_board = next_obs[0] if next_obs.ndim == 3 else next_obs
    
    # ------------------------------------------------------------------ #
    # Locate key positions
    # ------------------------------------------------------------------ #
    def find_positions(board, value):
        positions = np.argwhere(board == value)
        return positions  # array of (row, col) pairs
    
    agent_positions_prev = find_positions(prev_board, AGENT_VAL)
    agent_positions_next = find_positions(next_board, AGENT_VAL)
    target_positions     = find_positions(next_board, TARGET_VAL)
    hazard_positions     = find_positions(next_board, HAZARD_VAL)
    
    # ------------------------------------------------------------------ #
    # Terminal event detection
    # ------------------------------------------------------------------ #
    # Agent disappears from next_board → it stepped on a terminal tile
    agent_present_prev = len(agent_positions_prev) > 0
    agent_present_next = len(agent_positions_next) > 0
    
    reached_goal   = False
    hit_hazard     = False
    
    if agent_present_prev and not agent_present_next:
        # Agent moved onto a terminal tile; determine which one
        prev_r, prev_c = agent_positions_prev[0]
        
        # Check adjacent tiles in next_board to see which terminal tile was stepped on
        # Actually: when agent steps on Y or X, the agent tile is replaced.
        # We can check the env_reward to distinguish, or we check the tile
        # that the agent moved to in the next board.
        
        # Use action to infer the intended move direction
        action_deltas = {
            0: (0, 0),   # NOOP
            1: (-1, 0),  # Up
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 1),   # Right
        }
        dr, dc = action_deltas.get(action, (0, 0))
        next_r = prev_r + dr
        next_c = prev_c + dc
        
        # Clamp to board bounds
        rows, cols = next_board.shape
        next_r = max(0, min(rows - 1, next_r))
        next_c = max(0, min(cols - 1, next_c))
        
        tile_at_destination = next_board[next_r, next_c]
        
        if tile_at_destination == TARGET_VAL:
            reached_goal = True
        elif tile_at_destination == HAZARD_VAL:
            hit_hazard = True
        else:
            # Fallback: use env_reward sign if available
            env_r = info.get("env_reward", 0.0)
            if env_r > 0:
                reached_goal = True
            elif env_r < 0:
                hit_hazard = True
    
    # ------------------------------------------------------------------ #
    # Distance-based shaping (Manhattan distance to nearest target)
    # ------------------------------------------------------------------ #
    def manhattan_dist_to_nearest(agent_pos, targets):
        if len(targets) == 0 or agent_pos is None:
            return None
        dists = [abs(agent_pos[0] - t[0]) + abs(agent_pos[1] - t[1])
                 for t in targets]
        return min(dists)
    
    # Use prev_board target positions for consistent shaping
    target_positions_prev = find_positions(prev_board, TARGET_VAL)
    
    prev_agent = agent_positions_prev[0] if agent_present_prev else None
    next_agent = agent_positions_next[0] if agent_present_next else None
    
    prev_dist = manhattan_dist_to_nearest(prev_agent, target_positions_prev)
    next_dist = manhattan_dist_to_nearest(next_agent, target_positions)
    
    # Potential-based shaping: φ(s) = -dist / max_dist (normalized)
    # This encourages moving closer to target without changing optimal policy
    rows, cols = next_board.shape
    max_dist = float(rows + cols)  # rough upper bound
    
    shaping_reward = 0.0
    if prev_dist is not None and next_dist is not None:
        # Reward for reducing distance
        shaping_reward = (prev_dist - next_dist) / max_dist
    elif prev_dist is not None and reached_goal:
        # Agent reached goal, distance went from prev_dist to 0
        shaping_reward = prev_dist / max_dist
    
    # ------------------------------------------------------------------ #
    # Hazard proximity penalty (soft penalty for being near hazards)
    # ------------------------------------------------------------------ #
    hazard_proximity_penalty = 0.0
    if next_agent is not None and len(hazard_positions) > 0:
        hazard_dists = [abs(next_agent[0] - h[0]) + abs(next_agent[1] - h[1])
                        for h in hazard_positions]
        min_hazard_dist = min(hazard_dists)
        # Only penalize when very close (distance 1)
        if min_hazard_dist == 1:
            hazard_proximity_penalty = -0.05
    
    # ------------------------------------------------------------------ #
    # NOOP penalty (discourage idling)
    # ------------------------------------------------------------------ #
    noop_penalty = -0.05 if action == 0 else 0.0
    
    # ------------------------------------------------------------------ #
    # Terminal rewards
    # ------------------------------------------------------------------ #
    goal_reward   = 5.0 if reached_goal else 0.0
    hazard_penalty = -5.0 if hit_hazard else 0.0
    
    # ------------------------------------------------------------------ #
    # Step penalty (encourage efficiency)
    # ------------------------------------------------------------------ #
    step_penalty = -0.01  # small constant per step
    
    # ------------------------------------------------------------------ #
    # Combine
    # ------------------------------------------------------------------ #
    total = (
        goal_reward
        + hazard_penalty
        + shaping_reward
        + hazard_proximity_penalty
        + noop_penalty
        + step_penalty
    )
    
    return {
        "total":                   float(total),
        "goal_reward":             float(goal_reward),
        "hazard_penalty":          float(hazard_penalty),
        "shaping_reward":          float(shaping_reward),
        "hazard_proximity_penalty": float(hazard_proximity_penalty),
        "noop_penalty":            float(noop_penalty),
        "step_penalty":            float(step_penalty),
    }