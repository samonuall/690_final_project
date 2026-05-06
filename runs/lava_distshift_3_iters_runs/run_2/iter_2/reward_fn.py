import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function for SpillLab - v3.
    
    Key insight from v2 failure: agent was dying in exactly 2 steps every episode
    with BFS distance never firing. Fix:
    - Balanced reward scale (hazard/goal ~5x step, not 400x)
    - Strong per-step survival bonus that always fires
    - Simple robust Manhattan distance shaping (not BFS which was silently failing)
    - Hazard penalty sized to be instructive, not catastrophic vs exploration
    """

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

    # Locate agent and target
    prev_agent = find_tile(prev_board, AGENT)
    next_agent = find_tile(next_board, AGENT)
    
    # Target position - check both boards (disappears when reached)
    target_pos = find_tile(prev_board, TARGET)
    if target_pos is None:
        target_pos = find_tile(next_board, TARGET)

    # Hazard positions for proximity checks
    hazard_positions = find_all_tiles(prev_board, HAZARD)

    env_reward = info.get("env_reward", 0.0)

    # --- Terminal detection (robust) ---
    reached_target = env_reward > 0
    hit_hazard = env_reward < 0

    # Fallback: if agent tile disappears
    if next_agent is None and not reached_target and not hit_hazard:
        if target_pos is None:
            reached_target = True
        else:
            hit_hazard = True

    # ----------------------------------------------------------------
    # 1. GOAL REWARD — large positive for reaching extraction point
    # ----------------------------------------------------------------
    goal_reward = 5.0 if reached_target else 0.0

    # ----------------------------------------------------------------
    # 2. HAZARD PENALTY — negative but not overwhelming
    #    Key: must be LESS than cumulative survival bonus for a full
    #    episode, so surviving is always preferred to dying early
    # ----------------------------------------------------------------
    hazard_penalty = -3.0 if hit_hazard else 0.0

    # ----------------------------------------------------------------
    # 3. SURVIVAL BONUS — fires every non-terminal step
    #    This is the primary signal keeping the agent alive
    #    Sized so a full episode (~30 steps) ≈ 3.0, beating hazard
    # ----------------------------------------------------------------
    survival_bonus = 0.0
    if not reached_target and not hit_hazard:
        survival_bonus = 0.1

    # ----------------------------------------------------------------
    # 4. DISTANCE SHAPING — Manhattan distance to target
    #    Simple and robust; always computable
    #    Scaled small so it guides without overriding survival
    # ----------------------------------------------------------------
    distance_reward = 0.0
    if prev_agent is not None and target_pos is not None:
        prev_dist = abs(prev_agent[0] - target_pos[0]) + abs(prev_agent[1] - target_pos[1])
        
        if next_agent is not None and not reached_target and not hit_hazard:
            next_dist = abs(next_agent[0] - target_pos[0]) + abs(next_agent[1] - target_pos[1])
            # Positive when moving closer, negative when moving away
            distance_reward = float(prev_dist - next_dist) * 0.15
        elif reached_target:
            # Bonus proportional to distance covered
            distance_reward = float(prev_dist) * 0.1

    # ----------------------------------------------------------------
    # 5. HAZARD PROXIMITY PENALTY — avoid getting close to hazards
    #    Scaled relative to survival bonus: 1 adjacent hazard costs
    #    ~half the survival bonus to keep it informative not crippling
    # ----------------------------------------------------------------
    proximity_penalty = 0.0
    if next_agent is not None and not hit_hazard:
        r, c = next_agent
        rows, cols = next_board.shape
        for hr, hc in hazard_positions:
            dist_to_hazard = abs(r - hr) + abs(c - hc)
            if dist_to_hazard == 1:  # directly adjacent (cardinal)
                proximity_penalty -= 0.05
            elif dist_to_hazard == 2:  # two steps away
                proximity_penalty -= 0.01

    # ----------------------------------------------------------------
    # 6. EFFICIENCY PENALTIES — small, don't dominate
    # ----------------------------------------------------------------
    step_penalty = -0.02  # always active, encourage speed

    noop_penalty = -0.05 if action == 0 else 0.0  # discourage standing still

    # Wall bump: agent didn't move despite non-NOOP action
    wall_penalty = 0.0
    if (action != 0 and
            prev_agent is not None and
            next_agent is not None and
            prev_agent == next_agent and
            not reached_target and not hit_hazard):
        wall_penalty = -0.05

    # ----------------------------------------------------------------
    # TOTAL
    # ----------------------------------------------------------------
    total = (
        goal_reward
        + hazard_penalty
        + survival_bonus
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
        "survival_bonus":    float(survival_bonus),
        "distance_reward":   float(distance_reward),
        "proximity_penalty": float(proximity_penalty),
        "step_penalty":      float(step_penalty),
        "noop_penalty":      float(noop_penalty),
        "wall_penalty":      float(wall_penalty),
    }