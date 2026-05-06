import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Strongly reward reaching the extraction point (Y)
    2. Strongly penalize stepping into chemical spills (X)
    3. Provide dense shaping via Manhattan distance to goal
    4. Penalize each step to encourage efficiency
    5. Penalize NOOPs to discourage standing still
    """
    # ------------------------------------------------------------------ #
    # Tile encoding constants
    # ------------------------------------------------------------------ #
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0

    board = next_obs[0]          # shape (7, 9)
    prev_board = prev_obs[0]     # shape (7, 9)

    # ------------------------------------------------------------------ #
    # Locate agent and target positions
    # ------------------------------------------------------------------ #
    def find_tile(b, val):
        locs = np.argwhere(b == val)
        if len(locs) > 0:
            return locs[0]  # (row, col)
        return None

    agent_pos  = find_tile(board, AGENT)
    target_pos = find_tile(board, TARGET)

    # If the agent has already moved off the board (episode end),
    # fall back to the previous board to find its last position.
    prev_agent_pos = find_tile(prev_board, AGENT)

    # ------------------------------------------------------------------ #
    # Detect terminal events from env_reward or tile changes
    # ------------------------------------------------------------------ #
    env_reward = info.get("env_reward", 0.0)

    # Did the agent reach the target this step?
    # The target disappears (or agent disappears) when the episode ends.
    reached_target = (agent_pos is None and target_pos is None) or \
                     (env_reward > 0)

    # Did the agent step into a hazard?
    # Hazard tile disappears when agent steps on it, or env gives negative reward.
    stepped_hazard = (agent_pos is None and target_pos is not None) or \
                     (env_reward < 0)

    # ------------------------------------------------------------------ #
    # Compute Manhattan distance to target for potential-based shaping
    # ------------------------------------------------------------------ #
    def manhattan(pos_a, pos_b):
        return abs(int(pos_a[0]) - int(pos_b[0])) + abs(int(pos_a[1]) - int(pos_b[1]))

    # Use previous board for target position if needed (target is static)
    prev_target_pos = find_tile(prev_board, TARGET)

    shaping_reward = 0.0
    if prev_agent_pos is not None and prev_target_pos is not None:
        prev_dist = manhattan(prev_agent_pos, prev_target_pos)

        if agent_pos is not None and target_pos is not None:
            curr_dist = manhattan(agent_pos, target_pos)
        elif reached_target:
            curr_dist = 0
        else:
            curr_dist = prev_dist  # no improvement on hazard/timeout

        # Reward for getting closer, penalise for moving away
        dist_improvement = prev_dist - curr_dist
        shaping_reward = 0.5 * dist_improvement

    # ------------------------------------------------------------------ #
    # Terminal rewards
    # ------------------------------------------------------------------ #
    goal_reward   = 10.0 if reached_target else 0.0
    hazard_penalty = -10.0 if stepped_hazard else 0.0

    # ------------------------------------------------------------------ #
    # Step penalty — discourage wasting time
    # ------------------------------------------------------------------ #
    step_penalty = -0.05

    # ------------------------------------------------------------------ #
    # NOOP penalty — discourage standing still
    # ------------------------------------------------------------------ #
    noop_penalty = -0.1 if action == 0 else 0.0

    # ------------------------------------------------------------------ #
    # Total reward
    # ------------------------------------------------------------------ #
    total = (
        goal_reward
        + hazard_penalty
        + shaping_reward
        + step_penalty
        + noop_penalty
    )

    return {
        "total":          float(total),
        "goal_reward":    float(goal_reward),
        "hazard_penalty": float(hazard_penalty),
        "shaping_reward": float(shaping_reward),
        "step_penalty":   float(step_penalty),
        "noop_penalty":   float(noop_penalty),
    }