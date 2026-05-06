import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function for SpillLab environment.
    
    Key fixes vs. v1:
    - Much larger goal reward so finishing strictly dominates any shaping
    - Use env_reward directly as a reliable terminal signal
    - Stronger step penalty to prevent hovering
    - Simplified shaping to avoid local-optima exploitation
    - Removed proximity_penalty (was blocking near-goal movement)
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
        board = obs[0]  # shape (7, 9)
        positions = np.argwhere(np.isclose(board, value))
        if len(positions) > 0:
            return tuple(positions[0])
        return None

    # -----------------------------------------------------------------------
    # Raw environment reward — ground truth signal
    # -----------------------------------------------------------------------
    env_reward = float(info.get("env_reward", 0.0))

    # -----------------------------------------------------------------------
    # Locate tiles
    # -----------------------------------------------------------------------
    prev_agent = find_tile(prev_obs, AGENT_VAL)
    next_agent = find_tile(next_obs, AGENT_VAL)

    # Target position — read from prev_obs; may vanish when agent reaches it
    target_pos = find_tile(prev_obs, TARGET_VAL)
    if target_pos is None:
        target_pos = find_tile(next_obs, TARGET_VAL)

    # -----------------------------------------------------------------------
    # Terminal event detection
    # Use BOTH tile-based detection AND env_reward as cross-checks
    # -----------------------------------------------------------------------
    agent_disappeared = (next_agent is None)

    # Primary method: env_reward > 0 means success
    reached_goal = env_reward > 0.0

    # Secondary tile-based check: agent gone AND env_reward positive
    # (belt-and-suspenders)
    if agent_disappeared and env_reward > 0.0:
        reached_goal = True

    # Hazard: agent disappeared but env_reward is not positive
    hit_hazard = agent_disappeared and (env_reward <= 0.0)

    # -----------------------------------------------------------------------
    # 1. Terminal rewards — must dominate everything else
    # -----------------------------------------------------------------------
    # The maximum possible cumulative shaping over an episode is bounded by
    # the grid diameter (~14 steps * gamma discounting). With shaping ~6.4
    # plateau seen in training, goal_reward must be >> 6.4 to motivate
    # the final step.
    goal_reward    = 25.0 if reached_goal else 0.0
    hazard_penalty = -15.0 if hit_hazard else 0.0

    # -----------------------------------------------------------------------
    # 2. Potential-based distance shaping
    #    phi(s) = -manhattan_distance(agent, target)
    #    shaping = gamma * phi(s') - phi(s)
    #
    #    Scaled down vs v1 so shaping NEVER exceeds goal_reward in total.
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
            phi_next = 0.0   # distance = 0 at goal
        else:
            phi_next = phi_prev  # hit hazard: no shaping bonus

        # Scale shaping by 0.5 so its total accumulated value stays well
        # below goal_reward (25), preventing hovering exploitation
        shaping_reward = 0.5 * (gamma * phi_next - phi_prev)

    # -----------------------------------------------------------------------
    # 3. Step penalty — stronger than v1 to fight the hovering plateau
    # -----------------------------------------------------------------------
    step_penalty = -0.1  # per step

    # -----------------------------------------------------------------------
    # 4. NOOP penalty — discourage idling
    # -----------------------------------------------------------------------
    noop_penalty = -0.2 if action == 0 else 0.0

    # -----------------------------------------------------------------------
    # 5. Bonus for actually moving closer (on top of shaping)
    #    Directly reward progress steps to reinforce good transitions.
    # -----------------------------------------------------------------------
    progress_bonus = 0.0
    if (target_pos is not None
            and prev_agent is not None
            and next_agent is not None):
        prev_dist = abs(prev_agent[0] - target_pos[0]) + abs(prev_agent[1] - target_pos[1])
        next_dist = abs(next_agent[0] - target_pos[0]) + abs(next_agent[1] - target_pos[1])
        if next_dist < prev_dist:
            progress_bonus = 0.3   # moved closer
        elif next_dist > prev_dist:
            progress_bonus = -0.1  # moved farther

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
    )

    return {
        "total":           total,
        "goal_reward":     goal_reward,
        "hazard_penalty":  hazard_penalty,
        "shaping_reward":  shaping_reward,
        "step_penalty":    step_penalty,
        "noop_penalty":    noop_penalty,
        "progress_bonus":  progress_bonus,
    }