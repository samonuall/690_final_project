import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Reach the extraction point (Y) - large positive reward
    2. Avoid chemical spills (X) - large negative reward
    3. Move efficiently toward the target - shaped reward via distance reduction
    4. Penalize unnecessary steps (NOOP and general step cost)
    5. Penalize hitting walls (staying in place when not doing NOOP)
    """
    
    # Tile encoding constants
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0
    
    # Extract the board (shape: 1, 7, 9)
    prev_board = prev_obs[0]  # shape (7, 9)
    next_board = next_obs[0]  # shape (7, 9)
    
    # --- Locate key positions ---
    def find_tile(board, value):
        positions = np.argwhere(np.isclose(board, value))
        if len(positions) > 0:
            return positions[0]  # (row, col)
        return None
    
    prev_agent_pos = find_tile(prev_board, AGENT)
    next_agent_pos = find_tile(next_board, AGENT)
    target_pos     = find_tile(prev_board, TARGET)
    
    # If target not found in prev_board, check next_board (agent may be on it)
    if target_pos is None:
        target_pos = find_tile(next_board, TARGET)
    
    # --- Terminal condition detection ---
    env_reward = info.get("env_reward", 0.0)
    
    # Detect if agent reached the target (Y disappears from board or env_reward > 0)
    reached_target = False
    hit_hazard = False
    
    if next_agent_pos is None:
        # Agent tile is gone — episode ended
        if env_reward > 0:
            reached_target = True
        else:
            hit_hazard = True
    
    # Also check env_reward directly for robustness
    if env_reward > 0:
        reached_target = True
    elif env_reward < 0:
        hit_hazard = True
    
    # --- Compute Manhattan distance to target ---
    def manhattan(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    distance_reward = 0.0
    if target_pos is not None and prev_agent_pos is not None and next_agent_pos is not None:
        prev_dist = manhattan(prev_agent_pos, target_pos)
        next_dist = manhattan(next_agent_pos, target_pos)
        # Reward for getting closer, penalize for moving away
        distance_reward = float(prev_dist - next_dist) * 0.3
    
    # --- Goal reward ---
    goal_reward = 0.0
    if reached_target:
        goal_reward = 10.0
    
    # --- Hazard penalty ---
    hazard_penalty = 0.0
    if hit_hazard:
        hazard_penalty = -10.0
    
    # --- Proximity penalty: penalize being adjacent to hazards ---
    proximity_penalty = 0.0
    if next_agent_pos is not None:
        r, c = next_agent_pos
        neighbors = [
            (r - 1, c), (r + 1, c),
            (r, c - 1), (r, c + 1)
        ]
        hazard_neighbors = 0
        rows, cols = next_board.shape
        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols:
                if np.isclose(next_board[nr, nc], HAZARD):
                    hazard_neighbors += 1
        proximity_penalty = -0.05 * hazard_neighbors
    
    # --- Step penalty to encourage efficiency ---
    step_penalty = -0.02
    
    # --- NOOP penalty: discourage standing still ---
    noop_penalty = 0.0
    if action == 0:
        noop_penalty = -0.05
    
    # --- Wall bump penalty: penalize ineffective moves ---
    wall_penalty = 0.0
    if (action != 0 and 
        prev_agent_pos is not None and 
        next_agent_pos is not None and
        not reached_target and not hit_hazard):
        if np.array_equal(prev_agent_pos, next_agent_pos):
            wall_penalty = -0.1
    
    # --- Total reward ---
    total = (
        goal_reward
        + hazard_penalty
        + distance_reward
        + proximity_penalty
        + step_penalty
        + noop_penalty
        + wall_penalty
    )
    
    return {
        "total":             float(total),
        "goal_reward":       float(goal_reward),
        "hazard_penalty":    float(hazard_penalty),
        "distance_reward":   float(distance_reward),
        "proximity_penalty": float(proximity_penalty),
        "step_penalty":      float(step_penalty),
        "noop_penalty":      float(noop_penalty),
        "wall_penalty":      float(wall_penalty),
    }