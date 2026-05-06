import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function for SpillLab.

    Key insight from training analysis:
    - Agent was consistently walking into top-row hazards (Manhattan distance
      shaping naively pulls right toward Y, through XXX on row 1).
    - Need to discourage the direct right path and encourage going around
      via the lower rows.
    - Removed hazard proximity penalty (was conflicting with shaping).
    - Use potential-based shaping with a safe detour waypoint to guide
      the agent around hazards.
    """

    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0

    board      = next_obs[0]   # shape (7, 9)
    prev_board = prev_obs[0]

    def find_tile(b, value):
        positions = np.argwhere(b == value)
        if len(positions) == 0:
            return None
        return positions[0]  # (row, col)

    agent_pos      = find_tile(board, AGENT)
    target_pos     = find_tile(board, TARGET)
    prev_agent_pos = find_tile(prev_board, AGENT)

    env_reward = info.get("env_reward", 0.0)

    # --- Terminal detection ---
    reached_target = False
    hit_hazard     = False

    if env_reward > 0:
        reached_target = True
    elif env_reward < 0:
        hit_hazard = True

    # --- Current / previous positions ---
    cur_pos  = agent_pos  if agent_pos  is not None else prev_agent_pos
    prev_pos = prev_agent_pos

    # ---------------------------------------------------------------
    # Potential-based shaping using a safe waypoint.
    #
    # The layout is:
    #   Row 0: walls
    #   Row 1: A . X X X . Y   (hazards at cols 3,4,5)
    #   Row 2: empty corridor
    #   Row 3: empty corridor
    #   Row 4: empty corridor
    #   Row 5: . . X X X . .   (hazards at cols 3,4,5)
    #   Row 6: walls
    #
    # Safe path: go DOWN from row 1 to row 2-4, then RIGHT to col 7,
    # then UP back to row 1 for Y.
    #
    # We use a two-phase potential:
    #   Phase 1: if agent col <= 5 (left of / at hazard zone), encourage
    #            moving to the safe corridor (row 3, any col).
    #            Potential = -(distance to row 3)
    #   Phase 2: if agent is in safe rows (row 2-4), use straight
    #            Manhattan distance to target.
    #   Phase 3: if agent col > 5 (right of hazard zone), use
    #            Manhattan distance to target.
    # ---------------------------------------------------------------

    TARGET_ROW = 1
    TARGET_COL = 7
    SAFE_ROW   = 3   # middle of the open corridor

    # Hazard column range on row 1 and row 5
    HAZARD_COLS = {3, 4, 5}

    def safe_potential(pos):
        """
        Estimate of how close the agent is to safely reaching the target.
        Lower (more negative) = farther away.
        """
        if pos is None:
            return 0.0
        r, c = int(pos[0]), int(pos[1])

        # If on hazard, worst potential
        if board[r, c] == HAZARD or prev_board[r, c] == HAZARD:
            return -20.0

        # If agent is on row 1 (same row as start/target) and in/left of hazard zone
        if r == 1 and c <= 5:
            # Encourage going down first to safe row
            dist_to_safe = abs(r - SAFE_ROW)
            # Then from safe row at same column, go right, then up to target
            dist_safe_to_target = abs(SAFE_ROW - TARGET_ROW) + abs(c - TARGET_COL)
            return -(dist_to_safe + dist_safe_to_target)

        # If in the safe corridor rows (2-4) or already past the hazard zone
        if 2 <= r <= 4:
            # Go right to col 7, then up to target row
            dist_right = abs(c - TARGET_COL)
            dist_up    = abs(r - TARGET_ROW)
            return -(dist_right + dist_up)

        # If on row 1 col > 5 (right of hazards) — almost there
        if r == 1 and c > 5:
            return -(abs(r - TARGET_ROW) + abs(c - TARGET_COL))

        # Row 5 with hazards — treat as dangerous, go up first
        if r == 5:
            dist_up_safe = abs(r - SAFE_ROW)
            dist_safe_to_target = abs(SAFE_ROW - TARGET_ROW) + abs(c - TARGET_COL)
            return -(dist_up_safe + dist_safe_to_target + 2)  # extra penalty

        # Generic fallback
        return -(abs(r - TARGET_ROW) + abs(c - TARGET_COL))

    # Potential-based shaping: F(s', a, s) = gamma * Phi(s') - Phi(s)
    gamma = 0.99
    phi_cur  = safe_potential(cur_pos)
    phi_prev = safe_potential(prev_pos) if prev_pos is not None else phi_cur
    shaping_reward = gamma * phi_cur - phi_prev

    # --- Goal / hazard terminal rewards ---
    goal_reward   = 15.0 if reached_target else 0.0
    hazard_reward = -5.0 if hit_hazard     else 0.0

    # --- Step penalty ---
    step_penalty = -0.05

    # --- NOOP penalty ---
    noop_penalty = -0.15 if action == 0 else 0.0

    # --- Extra penalty for moving directly right on row 1 when near hazards ---
    # Detect: prev position was row 1, col in {1,2} and action was Right (4)
    # This directly discourages the naive "go right" strategy into hazards
    dangerous_move_penalty = 0.0
    if prev_pos is not None:
        pr, pc = int(prev_pos[0]), int(prev_pos[1])
        if pr == 1 and pc in {1, 2} and action == 4:  # Right on hazard row
            dangerous_move_penalty = -0.5

    # --- Combine ---
    total = (
        goal_reward
        + hazard_reward
        + shaping_reward
        + step_penalty
        + noop_penalty
        + dangerous_move_penalty
    )

    return {
        "total":                   float(total),
        "goal_reward":             float(goal_reward),
        "hazard_reward":           float(hazard_reward),
        "shaping_reward":          float(shaping_reward),
        "step_penalty":            float(step_penalty),
        "noop_penalty":            float(noop_penalty),
        "dangerous_move_penalty":  float(dangerous_move_penalty),
    }