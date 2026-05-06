import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Reach the extraction point (Y) as quickly as possible
    2. Avoid chemical spill zones (X)
    3. Discourage wasting time (step penalty)
    4. Provide dense shaping via distance-to-goal guidance
    """
    
    # Tile encoding constants
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0

    # ---------- helper: locate tiles ----------
    def find_tile(obs, value):
        """Return list of (row, col) positions matching value in obs[0]."""
        board = obs[0]
        positions = list(zip(*np.where(board == value)))
        return positions

    def agent_pos(obs):
        positions = find_tile(obs, AGENT)
        if positions:
            return positions[0]  # (row, col)
        return None

    def target_pos(obs):
        positions = find_tile(obs, TARGET)
        if positions:
            return positions[0]
        return None

    def hazard_positions(obs):
        return find_tile(obs, HAZARD)

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------- terminal detection ----------
    # If the agent tile disappears from next_obs, it means the agent stepped
    # onto a terminal tile (Y or X).
    prev_agent = agent_pos(prev_obs)
    next_agent = agent_pos(next_obs)

    # Check if target / hazard tile disappeared (agent consumed it by stepping on it)
    prev_target  = target_pos(prev_obs)
    next_target  = target_pos(next_obs)
    prev_hazards = hazard_positions(prev_obs)
    next_hazards = hazard_positions(next_obs)

    # Determine terminal outcomes
    reached_goal   = False
    reached_hazard = False

    if next_agent is None:
        # Agent tile gone — episode ended
        # If target also disappeared, agent reached the goal
        if prev_target is not None and next_target is None:
            reached_goal = True
        # If a hazard disappeared (or env_reward is very negative), stepped on hazard
        elif len(next_hazards) < len(prev_hazards):
            reached_hazard = True
        else:
            # Fall back to env_reward sign to distinguish
            env_r = info.get("env_reward", 0.0)
            if env_r > 0:
                reached_goal = True
            else:
                reached_hazard = True

    # ---------- reward components ----------

    # 1. Goal reward: large positive for reaching extraction point
    goal_reward = 10.0 if reached_goal else 0.0

    # 2. Hazard penalty: large negative for stepping into spill
    hazard_penalty = -10.0 if reached_hazard else 0.0

    # 3. Step penalty: small constant to encourage efficiency
    step_penalty = -0.05

    # 4. NOOP penalty: discourage standing still
    noop_penalty = -0.05 if action == 0 else 0.0

    # 5. Distance-based shaping: reward getting closer to the target
    #    Uses potential-based shaping: F = gamma * Phi(s') - Phi(s)
    #    Phi(s) = -distance_to_goal  (closer is higher potential)
    distance_shaping = 0.0
    if not reached_goal and not reached_hazard:
        # Both observations should have agent and target
        if prev_agent is not None and prev_target is not None:
            prev_dist = manhattan(prev_agent, prev_target)
            if next_agent is not None and next_target is not None:
                next_dist = manhattan(next_agent, next_target)
            elif next_agent is None and next_target is not None:
                # Agent just disappeared without terminal — shouldn't happen normally
                next_dist = prev_dist
            else:
                next_dist = prev_dist

            # Potential-based shaping (gamma=1 for undiscounted, but use slight discount)
            gamma = 0.99
            phi_prev = -float(prev_dist)
            phi_next = -float(next_dist)
            distance_shaping = gamma * phi_next - phi_prev

    # 6. Hazard proximity penalty: softly discourage being near hazards
    #    This helps the agent learn to avoid spill zones proactively
    hazard_proximity_penalty = 0.0
    if next_agent is not None and not reached_hazard:
        hazards = hazard_positions(next_obs)
        if hazards:
            min_hazard_dist = min(manhattan(next_agent, h) for h in hazards)
            # Penalize if within 2 steps of a hazard
            if min_hazard_dist == 1:
                hazard_proximity_penalty = -0.15
            elif min_hazard_dist == 2:
                hazard_proximity_penalty = -0.05

    # ---------- combine ----------
    total = (
        goal_reward
        + hazard_penalty
        + step_penalty
        + noop_penalty
        + distance_shaping
        + hazard_proximity_penalty
    )

    return {
        "total":                   float(total),
        "goal_reward":             float(goal_reward),
        "hazard_penalty":          float(hazard_penalty),
        "step_penalty":            float(step_penalty),
        "noop_penalty":            float(noop_penalty),
        "distance_shaping":        float(distance_shaping),
        "hazard_proximity_penalty": float(hazard_proximity_penalty),
    }