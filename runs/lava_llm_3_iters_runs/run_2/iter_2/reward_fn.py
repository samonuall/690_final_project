import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function v3 for SpillLab.

    Lessons from v2:
    - Agent learned the path but oscillates — hazard penalty too weak relative
      to step costs, causing the agent to sometimes prefer dying to navigating.
    - Episode with 50 hazard hits + 47 NOOPs = agent stuck at step limit,
      meaning hazard didn't terminate OR hazard penalty was acceptable cost.
    - Simplify: strong goal reward, strong hazard penalty, mild shaping,
      tiny step penalty, survival bonus to keep agent moving safely.
    """

    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0

    board      = next_obs[0]   # shape (7, 9)
    prev_board = prev_obs[0]   # shape (7, 9)

    def find_tile(b, value):
        positions = np.argwhere(b == value)
        return positions[0] if len(positions) > 0 else None

    agent_pos      = find_tile(board, AGENT)
    prev_agent_pos = find_tile(prev_board, AGENT)
    target_pos     = find_tile(board, TARGET)

    # If target disappeared from next_obs, find it in prev_obs
    if target_pos is None:
        target_pos = find_tile(prev_board, TARGET)

    env_reward = info.get("env_reward", 0.0)

    # --- Terminal detection ---
    reached_target = env_reward > 0
    hit_hazard     = env_reward < 0

    # --- Current position for shaping ---
    cur_pos  = agent_pos if agent_pos is not None else prev_agent_pos
    prev_pos = prev_agent_pos

    # ---------------------------------------------------------------
    # Safe-path potential function.
    #
    # Layout reminder:
    #   Row 1: A _ X X X _ Y   (start col=1, hazards cols 3-5, target col=7)
    #   Rows 2-4: open corridor
    #   Row 5: _ _ X X X _ _   (hazards cols 3-5)
    #
    # Optimal safe path: down to row 2-4, right to col 7, up to row 1 (Y).
    # We define a potential that rewards:
    #   1. Getting off row 1 (if col <= 5)
    #   2. Moving right in safe rows
    #   3. Getting to col 7
    #   4. Moving up to target
    #
    # Use a simple but safe distance metric:
    #   If on row 1 and col <= 5: must detour — cost = (to safe row) + (right) + (up)
    #   Otherwise: min Manhattan via safe corridor
    # ---------------------------------------------------------------

    TARGET_ROW = 1
    TARGET_COL = 7
    SAFE_ROW   = 2   # First safe row below the top hazards

    def safe_dist_to_target(pos):
        """
        Estimated safe distance to target.
        Returns a non-negative integer (lower = closer).
        """
        if pos is None:
            return 20  # large default
        r, c = int(pos[0]), int(pos[1])

        # Already at or past the hazard column range on the target row
        if r == TARGET_ROW and c > 5:
            return abs(TARGET_COL - c)

        # On the target row but blocked by hazards ahead (cols 3-5)
        if r == TARGET_ROW and c <= 5:
            # Must go down to safe row, right, then up
            detour = (r - SAFE_ROW) + abs(TARGET_COL - c) + (SAFE_ROW - TARGET_ROW)
            # detour down = SAFE_ROW - r (positive, going down increases row index)
            down_steps  = SAFE_ROW - r       # row 1 -> row 2 = 1 step
            right_steps = abs(TARGET_COL - c)
            up_steps    = SAFE_ROW - TARGET_ROW  # row 2 -> row 1 = 1 step
            return down_steps + right_steps + up_steps

        # In safe rows (2, 3, 4) — go right then up
        if 2 <= r <= 4:
            right_steps = abs(TARGET_COL - c)
            up_steps    = abs(r - TARGET_ROW)
            return right_steps + up_steps

        # Row 5 (lower hazard row) — go up first to row 4, then right, then up
        if r == 5:
            up_to_safe  = r - 4             # 1 step up to row 4
            right_steps = abs(TARGET_COL - c)
            up_to_target= 4 - TARGET_ROW    # row 4 -> row 1 = 3 steps
            return up_to_safe + right_steps + up_to_target

        # Generic fallback
        return abs(r - TARGET_ROW) + abs(c - TARGET_COL)

    # --- Potential-based shaping (Ng et al. 1999) ---
    # F(s,a,s') = gamma * Phi(s') - Phi(s)
    # Phi(s) = -safe_dist * scale  (higher potential = closer to goal)
    SHAPING_SCALE = 1.0
    gamma = 0.99

    cur_dist  = safe_dist_to_target(cur_pos)
    prev_dist = safe_dist_to_target(prev_pos) if prev_pos is not None else cur_dist

    phi_cur  = -SHAPING_SCALE * cur_dist
    phi_prev = -SHAPING_SCALE * prev_dist

    shaping_reward = gamma * phi_cur - phi_prev

    # --- Terminal rewards ---
    # Make goal reward clearly dominant; hazard penalty severe but not astronomically
    # large (to keep gradient scale manageable for PPO).
    goal_reward   = 20.0 if reached_target else 0.0
    hazard_reward = -20.0 if hit_hazard    else 0.0

    # --- Step penalty — very small to avoid "avoid everything" trap ---
    step_penalty = -0.02

    # --- NOOP penalty — small, just to discourage idling ---
    noop_penalty = -0.05 if action == 0 else 0.0

    # --- Combine ---
    total = (
        goal_reward
        + hazard_reward
        + shaping_reward
        + step_penalty
        + noop_penalty
    )

    return {
        "total":          float(total),
        "goal_reward":    float(goal_reward),
        "hazard_reward":  float(hazard_reward),
        "shaping_reward": float(shaping_reward),
        "step_penalty":   float(step_penalty),
        "noop_penalty":   float(noop_penalty),
    }