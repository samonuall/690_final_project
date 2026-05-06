import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Strongly reward reaching the extraction point (Y)
    2. Strongly penalize entering chemical spill zones (X)
    3. Provide dense shaping via distance-to-goal to guide exploration
    4. Small step penalty to encourage efficiency
    5. Penalize NOOP to discourage idling
    """

    # -----------------------------------------------------------------------
    # Tile value constants
    # -----------------------------------------------------------------------
    AGENT_VAL  = 2.0
    TARGET_VAL = 3.0
    HAZARD_VAL = 4.0

    # -----------------------------------------------------------------------
    # Helper: locate a tile value in the observation (shape: 1 x 7 x 9)
    # -----------------------------------------------------------------------
    def find_tile(obs, value):
        """Return (row, col) of the first cell matching `value`, or None."""
        board = obs[0]  # shape (7, 9)
        positions = np.argwhere(np.isclose(board, value))
        if len(positions) > 0:
            return tuple(positions[0])
        return None

    # -----------------------------------------------------------------------
    # Locate agent and target in both observations
    # -----------------------------------------------------------------------
    prev_agent  = find_tile(prev_obs, AGENT_VAL)
    next_agent  = find_tile(next_obs, AGENT_VAL)
    target_pos  = find_tile(prev_obs, TARGET_VAL)

    # Fallback: target might be occluded if agent stands on it (episode ends)
    if target_pos is None:
        target_pos = find_tile(next_obs, TARGET_VAL)

    # -----------------------------------------------------------------------
    # Detect terminal events
    # -----------------------------------------------------------------------
    env_reward = info.get("env_reward", 0.0)

    # Agent reached extraction point: agent tile disappears, env gives +reward
    reached_goal = (next_agent is None) and (env_reward > 0)

    # Agent hit a hazard: agent tile disappears, env gives negative / 0 reward
    hit_hazard   = (next_agent is None) and (env_reward <= 0)

    # -----------------------------------------------------------------------
    # 1. Sparse terminal rewards
    # -----------------------------------------------------------------------
    goal_reward   = 10.0 if reached_goal else 0.0
    hazard_penalty = -10.0 if hit_hazard else 0.0

    # -----------------------------------------------------------------------
    # 2. Dense distance-based shaping (potential-based)
    #    phi(s) = -manhattan_distance(agent, target)
    #    shaping = gamma * phi(s') - phi(s)
    # -----------------------------------------------------------------------
    gamma = 0.99
    shaping_reward = 0.0

    if target_pos is not None and prev_agent is not None:
        prev_dist = abs(prev_agent[0] - target_pos[0]) + abs(prev_agent[1] - target_pos[1])
        phi_prev  = -float(prev_dist)

        if next_agent is not None:
            next_dist = abs(next_agent[0] - target_pos[0]) + abs(next_agent[1] - target_pos[1])
            phi_next  = -float(next_dist)
        elif reached_goal:
            # Agent is on the goal, distance = 0
            phi_next = 0.0
        else:
            # Hit a hazard — no shaping benefit
            phi_next = phi_prev  # net shaping = 0

        shaping_reward = gamma * phi_next - phi_prev

    # -----------------------------------------------------------------------
    # 3. Step penalty — discourage wasting time
    # -----------------------------------------------------------------------
    step_penalty = -0.05

    # -----------------------------------------------------------------------
    # 4. NOOP penalty — discourage idling (action index 0)
    # -----------------------------------------------------------------------
    noop_penalty = -0.1 if action == 0 else 0.0

    # -----------------------------------------------------------------------
    # 5. Proximity-to-hazard penalty (optional soft deterrent)
    #    Penalise being adjacent to an X tile so the agent learns to avoid
    #    the hazard zone even before stepping into it.
    # -----------------------------------------------------------------------
    proximity_penalty = 0.0
    if next_agent is not None:
        board = next_obs[0]
        r, c  = next_agent
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if np.isclose(board[nr, nc], HAZARD_VAL):
                    proximity_penalty -= 0.2

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    total = (
        goal_reward
        + hazard_penalty
        + shaping_reward
        + step_penalty
        + noop_penalty
        + proximity_penalty
    )

    return {
        "total":             total,
        "goal_reward":       goal_reward,
        "hazard_penalty":    hazard_penalty,
        "shaping_reward":    shaping_reward,
        "step_penalty":      step_penalty,
        "noop_penalty":      noop_penalty,
        "proximity_penalty": proximity_penalty,
    }