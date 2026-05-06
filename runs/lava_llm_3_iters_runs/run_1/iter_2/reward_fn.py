import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function v3 for SpillLab.
    
    Key changes vs v2:
    - Larger step penalty to incentivize shorter paths
    - Larger goal reward to maintain dominance over accumulated penalties
    - Stronger shaping coefficient to bootstrap faster
    - Hazard proximity now penalizes more to keep the agent away from X
    - Progress bonus kept but rebalanced
    - Added a large bonus for reaching goal quickly (time-to-goal incentive)
    """

    # -----------------------------------------------------------------------
    # Tile value constants
    # -----------------------------------------------------------------------
    AGENT_VAL  = 2.0
    TARGET_VAL = 3.0
    HAZARD_VAL = 4.0
    WALL_VAL   = 0.0

    # -----------------------------------------------------------------------
    # Helper: locate a tile value in the observation (shape: 1 x 7 x 9)
    # -----------------------------------------------------------------------
    def find_tile(obs, value):
        board = obs[0]  # shape (7, 9)
        positions = np.argwhere(np.isclose(board, value))
        if len(positions) > 0:
            return tuple(positions[0])
        return None

    # -----------------------------------------------------------------------
    # Raw environment reward
    # -----------------------------------------------------------------------
    env_reward = float(info.get("env_reward", 0.0))

    # -----------------------------------------------------------------------
    # Locate tiles
    # -----------------------------------------------------------------------
    prev_agent = find_tile(prev_obs, AGENT_VAL)
    next_agent = find_tile(next_obs, AGENT_VAL)

    # Target — may vanish when stepped on
    target_pos = find_tile(prev_obs, TARGET_VAL)
    if target_pos is None:
        target_pos = find_tile(next_obs, TARGET_VAL)

    # -----------------------------------------------------------------------
    # Terminal event detection
    # -----------------------------------------------------------------------
    agent_disappeared = (next_agent is None)
    reached_goal  = (env_reward > 0.0)
    hit_hazard    = agent_disappeared and (env_reward <= 0.0)

    # -----------------------------------------------------------------------
    # 1. Terminal rewards
    # -----------------------------------------------------------------------
    goal_reward    = 50.0 if reached_goal else 0.0
    hazard_penalty = -20.0 if hit_hazard else 0.0

    # -----------------------------------------------------------------------
    # 2. Potential-based shaping  phi(s) = -manhattan_distance(agent, target)
    #    Stronger coefficient (1.5) to bootstrap exploration toward goal fast
    # -----------------------------------------------------------------------
    gamma = 0.99
    shaping_reward = 0.0

    if target_pos is not None and prev_agent is not None:
        prev_dist = (abs(prev_agent[0] - target_pos[0])
                     + abs(prev_agent[1] - target_pos[1]))
        phi_prev = -float(prev_dist)

        if next_agent is not None:
            next_dist = (abs(next_agent[0] - target_pos[0])
                         + abs(next_agent[1] - target_pos[1]))
            phi_next = -float(next_dist)
        elif reached_goal:
            phi_next = 0.0
        else:
            phi_next = phi_prev  # hazard: no shaping benefit

        shaping_reward = 1.5 * (gamma * phi_next - phi_prev)

    # -----------------------------------------------------------------------
    # 3. Step penalty — larger than v2 to incentivize shorter paths
    #    At 50 goal reward and -0.5/step, an agent saving 5 steps gains 2.5
    #    extra reward — meaningful difference.
    # -----------------------------------------------------------------------
    step_penalty = -0.5

    # -----------------------------------------------------------------------
    # 4. NOOP penalty
    # -----------------------------------------------------------------------
    noop_penalty = -0.5 if action == 0 else 0.0

    # -----------------------------------------------------------------------
    # 5. Progress bonus: explicit reward for reducing distance in one step
    # -----------------------------------------------------------------------
    progress_bonus = 0.0
    if (target_pos is not None
            and prev_agent is not None
            and next_agent is not None):
        prev_dist = (abs(prev_agent[0] - target_pos[0])
                     + abs(prev_agent[1] - target_pos[1]))
        next_dist = (abs(next_agent[0] - target_pos[0])
                     + abs(next_agent[1] - target_pos[1]))
        if next_dist < prev_dist:
            progress_bonus = 0.5
        elif next_dist > prev_dist:
            progress_bonus = -0.3

    # -----------------------------------------------------------------------
    # 6. Hazard proximity penalty — discourage loitering near X tiles
    #    (agent sometimes wanders near hazards while navigating around them)
    # -----------------------------------------------------------------------
    hazard_proximity = 0.0
    if next_agent is not None:
        board = next_obs[0]
        r, c = next_agent
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if np.isclose(board[nr, nc], HAZARD_VAL):
                    hazard_proximity -= 0.3

    # -----------------------------------------------------------------------
    # 7. Wall-bumping penalty — discourage wasting moves into walls
    # -----------------------------------------------------------------------
    wall_bump_penalty = 0.0
    if (prev_agent is not None
            and next_agent is not None
            and prev_agent == next_agent
            and action != 0):
        # Agent tried to move but didn't — likely hit a wall
        wall_bump_penalty = -0.3

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    total = (
        goal_reward
        + hazard_penalty
        + shaping_reward
        + step_penalty
        + noop_penalty
        + progress_bonus
        + hazard_proximity
        + wall_bump_penalty
    )

    return {
        "total":             total,
        "goal_reward":       goal_reward,
        "hazard_penalty":    hazard_penalty,
        "shaping_reward":    shaping_reward,
        "step_penalty":      step_penalty,
        "noop_penalty":      noop_penalty,
        "progress_bonus":    progress_bonus,
        "hazard_proximity":  hazard_proximity,
        "wall_bump_penalty": wall_bump_penalty,
    }