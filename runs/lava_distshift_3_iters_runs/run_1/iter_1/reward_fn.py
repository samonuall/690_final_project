import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function for SpillLab environment.
    
    Key fixes:
    - Use env_reward as authoritative terminal signal
    - Much larger goal reward to overcome shaping exploitation
    - Stronger step penalty to prevent hovering
    - Simpler, more reliable shaping
    """

    # Tile encoding constants
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0

    # ---------- helpers ----------
    def find_tile(obs, value):
        board = obs[0]
        positions = list(zip(*np.where(board == value)))
        return [tuple(int(x) for x in p) for p in positions]

    def agent_pos(obs):
        positions = find_tile(obs, AGENT)
        return positions[0] if positions else None

    def target_pos(obs):
        positions = find_tile(obs, TARGET)
        return positions[0] if positions else None

    def hazard_positions(obs):
        return find_tile(obs, HAZARD)

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------- use env_reward as ground truth for terminal events ----------
    env_reward = float(info.get("env_reward", 0.0))

    # Detect terminal outcomes from env_reward
    # Positive env_reward => reached goal, negative => hit hazard
    reached_goal   = env_reward > 0.5
    reached_hazard = env_reward < -0.5
    is_terminal    = reached_goal or reached_hazard

    # ---------- reward components ----------

    # 1. Goal reward: large positive — must dominate all shaping
    goal_reward = 50.0 if reached_goal else 0.0

    # 2. Hazard penalty: large negative
    hazard_penalty = -25.0 if reached_hazard else 0.0

    # 3. Step penalty: meaningful penalty to drive urgency
    #    Make it larger so hovering near goal isn't profitable
    step_penalty = -0.5

    # 4. NOOP penalty: extra discouragement for standing still
    noop_penalty = -0.5 if action == 0 else 0.0

    # 5. Distance-based potential shaping
    #    Only apply during non-terminal steps to avoid confounding terminal signal
    distance_shaping = 0.0
    if not is_terminal:
        prev_agent  = agent_pos(prev_obs)
        prev_target = target_pos(prev_obs)
        next_agent_p = agent_pos(next_obs)
        next_target = target_pos(next_obs)

        if (prev_agent is not None and prev_target is not None and
                next_agent_p is not None and next_target is not None):
            prev_dist = manhattan(prev_agent, prev_target)
            next_dist = manhattan(next_agent_p, next_target)
            # Reward for getting closer, penalize for moving away
            # Simple delta shaping (gamma=1 approximation)
            distance_shaping = float(prev_dist - next_dist) * 1.5

    # 6. Hazard proximity penalty: discourage loitering near spills
    #    Only when not terminal (terminal hazard already penalized above)
    hazard_proximity_penalty = 0.0
    if not is_terminal:
        next_agent_p = agent_pos(next_obs)
        if next_agent_p is not None:
            hazards = hazard_positions(next_obs)
            if hazards:
                min_hazard_dist = min(manhattan(next_agent_p, h) for h in hazards)
                if min_hazard_dist == 1:
                    hazard_proximity_penalty = -0.3
                elif min_hazard_dist == 2:
                    hazard_proximity_penalty = -0.1

    # 7. Efficiency bonus: reward for reaching goal quickly
    #    Use step count from info if available
    efficiency_bonus = 0.0
    if reached_goal:
        # Bonus that decays with steps taken — encourages fast solutions
        steps_taken = float(info.get("step_count", 50))
        efficiency_bonus = max(0.0, 20.0 - steps_taken * 0.5)

    # ---------- combine ----------
    total = (
        goal_reward
        + hazard_penalty
        + step_penalty
        + noop_penalty
        + distance_shaping
        + hazard_proximity_penalty
        + efficiency_bonus
    )

    return {
        "total":                    float(total),
        "goal_reward":              float(goal_reward),
        "hazard_penalty":           float(hazard_penalty),
        "step_penalty":             float(step_penalty),
        "noop_penalty":             float(noop_penalty),
        "distance_shaping":         float(distance_shaping),
        "hazard_proximity_penalty": float(hazard_proximity_penalty),
        "efficiency_bonus":         float(efficiency_bonus),
        "env_reward_passthrough":   env_reward,
    }