import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    - Reach extraction point Y (positive reward)
    - Avoid chemical spill zones X (negative reward)
    - Minimize steps taken (step penalty)
    - Guide agent toward Y using distance-based shaping
    - Discourage standing still (NOOP penalty)
    """

    # --- Tile encoding constants ---
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0

    board = next_obs[0]       # shape (7, 9)
    prev_board = prev_obs[0]  # shape (7, 9)

    # --- Locate agent, target, hazards ---
    def find_tile(board, value):
        positions = np.argwhere(board == value)
        if len(positions) == 0:
            return None
        return positions[0]  # (row, col)

    agent_pos  = find_tile(board, AGENT)
    target_pos = find_tile(board, TARGET)

    # If the agent tile isn't on the board, the episode ended —
    # agent is on terminal tile. Infer position from previous obs.
    prev_agent_pos = find_tile(prev_board, AGENT)

    # --- Terminal detection ---
    env_reward = info.get("env_reward", 0.0)

    # Determine if agent stepped onto Y or X
    reached_target = False
    hit_hazard     = False

    if agent_pos is None and prev_agent_pos is not None:
        # Agent disappeared — it stepped onto a terminal tile
        prev_r, prev_c = prev_agent_pos
        # Check what's at the destination in prev_obs (target or hazard)
        # We can infer from env_reward or by checking next_obs terminal tile values
        # Use env_reward as the cleaner signal
        if env_reward > 0:
            reached_target = True
        elif env_reward < 0:
            hit_hazard = True
        else:
            # env_reward == 0 at terminal: check surrounding tiles from prev position
            # for target presence
            reached_target = False
            hit_hazard = False

    # Also check: agent is still on board but env_reward signals terminal
    if agent_pos is not None:
        if env_reward > 0:
            reached_target = True
        elif env_reward < 0:
            hit_hazard = True

    # --- Manhattan distance shaping ---
    # Current agent position for distance calculation
    cur_pos = agent_pos if agent_pos is not None else prev_agent_pos

    shaping_reward = 0.0
    if cur_pos is not None and target_pos is not None:
        # Distance from current agent to target
        cur_dist = abs(cur_pos[0] - target_pos[0]) + abs(cur_pos[1] - target_pos[1])

        if prev_agent_pos is not None:
            prev_dist = abs(prev_agent_pos[0] - target_pos[0]) + abs(prev_agent_pos[1] - target_pos[1])
        else:
            prev_dist = cur_dist

        # Reward for moving closer, penalise moving further away
        delta = prev_dist - cur_dist   # positive = got closer
        shaping_reward = 0.3 * delta

    # --- Hazard proximity penalty ---
    # Penalise being adjacent to hazard tiles (within 1 step)
    hazard_proximity_penalty = 0.0
    if cur_pos is not None:
        r, c = cur_pos
        neighbours = [
            (r - 1, c), (r + 1, c),
            (r, c - 1), (r, c + 1),
        ]
        for nr, nc in neighbours:
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if board[nr, nc] == HAZARD or prev_board[nr, nc] == HAZARD:
                    hazard_proximity_penalty -= 0.05

    # --- Goal / hazard terminal rewards ---
    goal_reward   = 0.0
    hazard_reward = 0.0

    if reached_target:
        goal_reward = 10.0
    if hit_hazard:
        hazard_reward = -10.0

    # --- Step penalty (encourage efficiency) ---
    step_penalty = -0.05

    # --- NOOP penalty (discourage idling) ---
    noop_penalty = 0.0
    if action == 0:
        noop_penalty = -0.1

    # --- Combine ---
    total = (
        goal_reward
        + hazard_reward
        + shaping_reward
        + hazard_proximity_penalty
        + step_penalty
        + noop_penalty
    )

    return {
        "total":                   float(total),
        "goal_reward":             float(goal_reward),
        "hazard_reward":           float(hazard_reward),
        "shaping_reward":          float(shaping_reward),
        "hazard_proximity_penalty":float(hazard_proximity_penalty),
        "step_penalty":            float(step_penalty),
        "noop_penalty":            float(noop_penalty),
    }