import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function for SpillLab environment.
    
    Key fixes vs previous version:
    - Much stronger hazard penalty to break the rush-into-hazard local optimum
    - Strong proximity penalty to push agent away from hazard-adjacent tiles
    - Survival bonus to reward staying alive each step
    - Weaker distance shaping so it guides without dominating
    - BFS-based safe distance shaping (avoids routing through hazards)
    """

    # Tile encoding constants
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0

    prev_board = prev_obs[0]  # shape (7, 9)
    next_board = next_obs[0]  # shape (7, 9)

    def find_tile(board, value):
        positions = np.argwhere(np.isclose(board, value))
        if len(positions) > 0:
            return tuple(positions[0])
        return None

    def find_all_tiles(board, value):
        return [tuple(p) for p in np.argwhere(np.isclose(board, value))]

    prev_agent_pos = find_tile(prev_board, AGENT)
    next_agent_pos = find_tile(next_board, AGENT)

    # Target can disappear when agent steps on it
    target_pos = find_tile(prev_board, TARGET)
    if target_pos is None:
        target_pos = find_tile(next_board, TARGET)

    env_reward = info.get("env_reward", 0.0)

    # --- Terminal detection ---
    reached_target = False
    hit_hazard = False

    if env_reward > 0:
        reached_target = True
    elif env_reward < 0:
        hit_hazard = True
    elif next_agent_pos is None:
        # Agent tile disappeared without env_reward signal
        if target_pos is None:
            reached_target = True
        else:
            hit_hazard = True

    # --- BFS for safe distance (avoids hazards and walls) ---
    def bfs_safe_distance(board, start, goal):
        """
        BFS from start to goal treating HAZARD and WALL as impassable.
        Returns shortest safe path length, or a large fallback if unreachable.
        """
        if start is None or goal is None:
            return None
        rows, cols = board.shape
        visited = set()
        queue = [(start, 0)]
        visited.add(start)
        while queue:
            # Simple BFS with list (fine for 7x9 grid)
            (r, c), dist = queue.pop(0)
            if (r, c) == goal:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                tile = board[nr, nc]
                # Passable if: empty, agent, or target (not wall, not hazard)
                if np.isclose(tile, WALL) or np.isclose(tile, HAZARD):
                    continue
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
        return None  # unreachable

    # --- Safe distance shaping ---
    distance_reward = 0.0
    if not reached_target and not hit_hazard:
        if prev_agent_pos is not None and next_agent_pos is not None and target_pos is not None:
            prev_safe_dist = bfs_safe_distance(prev_board, prev_agent_pos, target_pos)
            next_safe_dist = bfs_safe_distance(next_board, next_agent_pos, target_pos)

            if prev_safe_dist is not None and next_safe_dist is not None:
                # Reward for reducing safe distance, penalize for increasing it
                delta = prev_safe_dist - next_safe_dist
                distance_reward = float(delta) * 0.25
            elif prev_safe_dist is not None and next_safe_dist is None:
                # Moved into an unreachable state — penalize
                distance_reward = -0.5

    # --- Goal reward ---
    goal_reward = 0.0
    if reached_target:
        goal_reward = 15.0

    # --- Hazard penalty (much larger than before) ---
    hazard_penalty = 0.0
    if hit_hazard:
        hazard_penalty = -20.0

    # --- Proximity penalty: penalize being adjacent to hazards ---
    # This is the main mechanism to keep agent away from hazards
    proximity_penalty = 0.0
    if next_agent_pos is not None and not hit_hazard:
        r, c = next_agent_pos
        next_rows, next_cols = next_board.shape
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < next_rows and 0 <= nc < next_cols:
                if np.isclose(next_board[nr, nc], HAZARD):
                    # Cardinal neighbors penalized more than diagonal
                    if abs(dr) + abs(dc) == 1:
                        proximity_penalty -= 0.3
                    else:
                        proximity_penalty -= 0.1

    # --- Survival bonus: reward being alive each step ---
    survival_bonus = 0.0
    if not reached_target and not hit_hazard:
        survival_bonus = 0.05

    # --- Step penalty for efficiency ---
    step_penalty = -0.05

    # --- NOOP penalty ---
    noop_penalty = 0.0
    if action == 0:
        noop_penalty = -0.1

    # --- Wall bump penalty ---
    wall_penalty = 0.0
    if (action != 0 and
            prev_agent_pos is not None and
            next_agent_pos is not None and
            not reached_target and not hit_hazard):
        if prev_agent_pos == next_agent_pos:
            wall_penalty = -0.15

    total = (
        goal_reward
        + hazard_penalty
        + distance_reward
        + proximity_penalty
        + survival_bonus
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
        "survival_bonus":    float(survival_bonus),
        "step_penalty":      float(step_penalty),
        "noop_penalty":      float(noop_penalty),
        "wall_penalty":      float(wall_penalty),
    }